import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
os.environ['TFDS_DATA_DIR']="data"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import flwr as fl
import GPUtil
from time import sleep
from pathlib import Path
import shutil
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
parser.add_argument('--embeddings_dir',			type=str,	default='./data/features',	required=False)
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

 	######################################################
	# Override Client to use Nearest Neighbor Correction #
	######################################################
	from utils.knn_relabel import relabel_with_pretrained_knn
	import tensorflow as tf
	import numpy as np

	# Load corrected labels (if have been computed)
	labels = np.load(open(f'{args.temp_dir}/relabel_{cid}.npy','rb')) if os.path.isfile(f'{args.temp_dir}/relabel_{cid}.npy') else None

	class NNCClient(Client):

		def __init__(self, dataset, num_neighbors, labels, temp_dir, feat_dir, *args, **kwargs):
			super(NNCClient, self).__init__(*args, **kwargs)
			# Ensure that shuffle is deactivated
			assert (not self.shuffle), "Shuffle must be deactivated for prunning noisy samples with NNC"
			# Relabel noisy samples
			if labels is not None:
				kwargs['shuffle']=True
				self.data, self.num_classes, self.num_samples, self.labels, self.idxs = \
					kwargs['data_loader'](shard_id=int(self.cid), num_shards=int(kwargs['num_clients']),
						shuffle=True, noisy_clients_frac=kwargs['noisy_clients_frac'],
						noise_lvl=kwargs['noise_lvl'], noise_sparsity=kwargs['noise_sparsity'],
						batch_size=kwargs['batch_size'], corrected_labels=labels)
			else: # Run NNC relabelling process
				_, self.num_classes,_, self.labels, self.idxs = \
					kwargs['data_loader'](shard_id=int(self.cid), num_shards=int(kwargs['num_clients']),
                        shuffle=False, noisy_clients_frac=kwargs['noisy_clients_frac'],
						noise_lvl=kwargs['noise_lvl'], noise_sparsity=kwargs['noise_sparsity'],
						batch_size=kwargs['batch_size'], corrected_labels=None)
				with open(f'{feat_dir}/{dataset}.npy','rb') as f: features = np.squeeze(np.load(f)[self.idxs])
				labels = relabel_with_pretrained_knn(labels=self.labels, features=features, num_classes=self.num_classes,
													num_neighbors=num_neighbors, verbose=True)
				kwargs['shuffle']=True
				self.data, self.num_classes, self.num_samples, self.labels, self.idxs = \
					kwargs['data_loader'](shard_id=int(self.cid), num_shards=int(kwargs['num_clients']),
						shuffle=True, noisy_clients_frac=kwargs['noisy_clients_frac'],
						noise_lvl=kwargs['noise_lvl'], noise_sparsity=kwargs['noise_sparsity'],
						batch_size=kwargs['batch_size'], corrected_labels=labels)
				tf.keras.backend.clear_session()
				# Store to file if exist
				if not os.path.isdir(f"{temp_dir}"): os.makedirs(f"{temp_dir}")
				with open(f'{temp_dir}/relabel_{cid}.npy','wb') as f : np.save(f,labels)
			del self.labels # No use anymore.
			self.input_shape = self.data.element_spec[0].shape
	######################################################

	load_model = load_available_models()[args.model_name]
	load_train_data = load_available_datasets()[args.dataset_name]
	kwargs = {'batch_size':int(args.batch_size), 'seed':int(args.seed), 'noisy_clients_frac':float(args.noisy_frac),
			'noise_lvl':float(args.noise_level), 'noise_sparsity':float(args.noise_sparsity)}
	return NNCClient(dataset=args.dataset_name, labels=labels, num_neighbors=int(args.knn_neighbors), cid=cid,
                num_clients=int(args.num_clients), model_loader=load_model, data_loader=load_train_data,
                shuffle=False, load_data=False, temp_dir=args.temp_dir, feat_dir=args.embeddings_dir, **kwargs)

def create_server():
	os.environ['CUDA_VISIBLE_DEVICES'] = grab_gpu()
	from utils.flwr_server import _Server as Server
	load_model = load_available_models()[args.model_name]
	load_test_data = load_available_datasets(train=False)[args.dataset_name]
	kwargs = {'lr':float(args.lr), 'train_epochs':int(args.train_epochs)}
	return Server(num_rounds=int(args.num_rounds), num_clients=int(args.num_clients),
			participation=float(args.participation_rate), model_loader=load_model,
			data_loader=load_test_data, **kwargs)

def run_simulation():
	# Create server
	server = create_server()
	# Start simulation
	history = fl.simulation.start_simulation(client_fn=create_client, server=server, num_clients=int(args.num_clients),
		ray_init_args= {"ignore_reinit_error": True, "num_cpus": int(args.num_clients),},
		config=fl.server.ServerConfig(num_rounds=int(args.num_rounds), round_timeout=None),)
	if Path(args.temp_dir).exists() and Path(args.temp_dir).is_dir(): shutil.rmtree(Path(args.temp_dir))
	return history

if __name__ == "__main__":
	print(run_simulation())
