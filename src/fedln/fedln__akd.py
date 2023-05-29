import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import flwr as fl
import GPUtil
from time import sleep
from pathlib import Path
import shutil
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_clients',			type=int,	default=3,					required=False)
parser.add_argument('--num_rounds',				type=int,	default=10,					required=False)
parser.add_argument('--participation_rate',		type=float,	default=1.0,				required=False)
parser.add_argument('--batch_size',				type=int,	default=128,				required=False)
parser.add_argument('--train_epochs',			type=int,	default=1,					required=False)
parser.add_argument('--lr',						type=float,	default=1e-3,				required=False)
parser.add_argument('--knn_neighbors',			type=int,	default=10,					required=False)
parser.add_argument('--noisy_frac',				type=float,	default=0.8,				required=False)
parser.add_argument('--noise_level',			type=float,	default=0.4,				required=False)
parser.add_argument('--noise_sparsity',			type=float,	default=0.7,				required=False)
parser.add_argument('--distil_round',			type=int,	default=1,					required=False)
parser.add_argument('--embeddings_dir',			type=str,	default='./data/features',	required=False)
parser.add_argument('--embeddings_dims',		type=int,	default=512,				required=False)
parser.add_argument('--dataset_name',			type=str,	default='eurosat',			required=False)
parser.add_argument('--model_name',				type=str, 	default='resnet20',			required=False)
parser.add_argument('--temp_dir',				type=str, 	default='./tmp',			required=False)
parser.add_argument('--seed',					type=int, 	default=42,					required=False)
args = parser.parse_args()

if Path(args.temp_dir).exists() and Path(args.temp_dir).is_dir(): shutil.rmtree(Path(args.temp_dir))

def load_available_datasets(train=True):
	import data
	return {
		'eurosat': data.load_eurosat if train else data.load_eurosat_test, 
		'cifar10': data.load_cifar if train else data.load_cifar_test,
    }

def load_available_models():
	import models
	return {
		'resnet20': models.load_resnet20_model,
		'cnn': models.load_cnn_model,
	}

def grab_gpu(memory_limit=0.91):
	while len(GPUtil.getAvailable(order='memory', limit=len(GPUtil.getGPUs()), maxLoad=1.0, maxMemory=memory_limit)) == 0: sleep(1)
	cuda_device_ids = GPUtil.getAvailable(order='memory', limit=len(GPUtil.getGPUs()), maxLoad=1.0, maxMemory=memory_limit)
	cuda_device_ids.extend("") # Fix no gpu issue
	return str(cuda_device_ids[0])

def create_client(cid):
	sleep(int(cid)*0.75)
	os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
	os.environ['CUDA_VISIBLE_DEVICES'] = grab_gpu()
	sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
	from utils.flwr_client import _Client as Client
	from utils.knn_relabel import estimate_noise_with_pretrained_knn
	from utils.distiller import Distiller
	import numpy as np
	import tensorflow as tf

	#############################################
	# Override Client to use Noise-Aware FedAvg #
	#############################################
	class AKDClient(Client):
		def __init__(self, dataset, distil_round, num_neighbors, temp_dir, *args, **kwargs):
			super(AKDClient, self).__init__(*args, **kwargs)
			self.distil_round = distil_round
			self.dataset = dataset
			self.num_neighbors = num_neighbors
			self.data_loader = kwargs['data_loader']
			self.num_clients = int(kwargs['num_clients'])
			self.embeddings_params = None
			self.temp_dir = temp_dir

		@property
		def noisy(self):
			return os.path.isfile(f'{self.temp_dir}/noise_{self.cid}.npy')

		def set_parameters(self, parameters, config):
			if not hasattr(self, 'model'): self.model = self.model_loader(input_shape=self.input_shape[1:], num_classes=self.num_classes, embeddings_dim=int(args.embeddings_dims) if self.noisy else None)
			self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']), loss=tf.keras.losses.CategoricalCrossentropy(name='loss', from_logits=True), metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])
			if parameters is not None: 
				if (self.noisy): parameters = parameters + (self.model.get_weights()[-2:] if self.embeddings_params is None else self.embeddings_params)					
				self.model.set_weights(parameters)
   
		def fit(self, parameters, config):

			# Create distillation process
			if int(config['round'])==self.distil_round:
				_data, _, _, _labels, _idxs = self.data_loader(shard_id=int(self.cid), num_shards=self.num_clients, shuffle=False)
				_data = _data.unbatch()
				with open(f'{args.embeddings_dir}/{self.dataset}.npy','rb') as f: _features = np.squeeze(np.load(f)[_idxs])
				estimated_noise = estimate_noise_with_pretrained_knn(labels=_labels, features=_features, num_classes=self.num_classes, num_neighbors=self.num_neighbors, verbose=False)
				if not os.path.isdir(f"{args.temp_dir}"): os.makedirs(f"{args.temp_dir}")
				if estimated_noise > 0.0:
					with open(f'{args.temp_dir}/noise_{self.cid}.npy','wb') as f: np.save(f,np.array([estimated_noise]))
    
			# Set parameters
			self.set_parameters(parameters, config)
   
			if self.noisy:
				_data, _, _, _, _idxs = self.data_loader(shard_id=int(self.cid), num_shards=self.num_clients, shuffle=False)
				with open(f'{args.embeddings_dir}/{self.dataset}.npy','rb') as f: _features = np.squeeze(np.load(f)[_idxs])
				self.model = Distiller(model=self.model, features=_features, idxs=np.squeeze(_idxs))
				self.embeddings_dim = 512
				self.set_parameters(parameters=None, config=config)
				self.data = tf.data.Dataset.zip((_data.unbatch(), tf.data.Dataset.from_tensor_slices((np.squeeze(_idxs)))))
				self.data = self.data.batch(batch_size=128).prefetch(tf.data.AUTOTUNE)

			# Client local update
			h = self.model.fit(self.data, epochs=config['epochs'], verbose=2)

			self.embeddings_params = self.model.get_weights()[-2:]
			metrics = {'accuracy':float(h.history['accuracy'][-1]), 'loss':float(h.history['loss'][-1])}
			return self.get_parameters(), self.num_samples, metrics
	#############################################

	load_model = load_available_models()[args.model_name]
	load_train_data = load_available_datasets()[args.dataset_name]
	kwargs = {'batch_size':int(args.batch_size), 'seed':int(args.seed), 'noisy_clients_frac':float(args.noisy_frac),
			'noise_lvl':float(args.noise_level), 'noise_sparsity':float(args.noise_sparsity)}
	return AKDClient(dataset=args.dataset_name, distil_round=int(args.distil_round), num_neighbors=int(args.knn_neighbors),
            temp_dir=args.temp_dir, cid=cid, num_clients=int(args.num_clients), model_loader=load_model, data_loader=load_train_data,
            shuffle=False, **kwargs)

def create_server():
	os.environ['CUDA_VISIBLE_DEVICES'] = grab_gpu()
	from utils.flwr_server import _Server as Server
	load_model = load_available_models()[args.model_name]
	load_test_data = load_available_datasets(train=False)[args.dataset_name]
	kwargs = {'lr':float(args.lr), 'train_epochs':int(args.train_epochs)}
	return Server(num_rounds=int(args.num_rounds), num_clients=int(args.num_clients), participation=float(args.participation_rate),
            model_loader=load_model, data_loader=load_test_data, **kwargs)

def run_simulation():
	# Create server
	server = create_server()
	# Start simulation
	history = fl.simulation.start_simulation(client_fn=create_client, server=server, num_clients=int(args.num_clients),
		ray_init_args= {"ignore_reinit_error": True, "num_cpus": int(args.num_clients),},
		config=fl.server.ServerConfig(num_rounds=int(args.num_rounds), round_timeout=None),)
	if Path(args.temp_dir).exists() and Path(temp_dir).is_dir(): shutil.rmtree(Path(temp_dir))
	return history

if __name__ == "__main__":
	print(run_simulation())
