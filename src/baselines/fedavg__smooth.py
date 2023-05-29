import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import flwr as fl
import GPUtil
from time import sleep
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_clients',			type=int,	default=3,					required=False)
parser.add_argument('--num_rounds',				type=int,	default=10,					required=False)
parser.add_argument('--participation_rate',		type=float,	default=1.0,				required=False)
parser.add_argument('--batch_size',				type=int,	default=128,				required=False)
parser.add_argument('--train_epochs',			type=int,	default=1,					required=False)
parser.add_argument('--lr',						type=float,	default=1e-3,				required=False)
parser.add_argument('--noisy_frac',				type=float,	default=0.8,				required=False)
parser.add_argument('--noise_level',			type=float,	default=0.4,				required=False)
parser.add_argument('--noise_sparsity',			type=float,	default=0.7,				required=False)
parser.add_argument('--dataset_name',			type=str,	default='eurosat',			required=False)
parser.add_argument('--model_name',				type=str, 	default='resnet20',			required=False)
parser.add_argument('--smooth_rate',			type=float, default=0.2,				required=False)
parser.add_argument('--seed',					type=int, 	default=42,					required=False)
args = parser.parse_args()

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

	#############################################
	# Override Client to perform label smooting #
	#############################################
	import tensorflow as tf
	class LabelSmoothClient(Client):
		def __init__(self, smooth_rate, *args, **kwargs):
			super(LabelSmoothClient, self).__init__(*args, **kwargs)
			self.smoothing_rate =smooth_rate

		def set_parameters(self, parameters, config):
			if not hasattr(self, 'model'): self.model = self.model_loader(input_shape=self.input_shape[1:], num_classes=self.num_classes)
			self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']), loss=tf.keras.losses.CategoricalCrossentropy(name='loss', label_smoothing=self.smoothing_rate, from_logits=True), metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])
			if parameters is not None: self.model.set_weights(parameters)
	#############################################

	load_model = load_available_models()[args.model_name]
	load_train_data = load_available_datasets()[args.dataset_name]
	kwargs = {'batch_size':int(args.batch_size), 'seed':int(args.seed), 'noisy_clients_frac':float(args.noisy_frac),
			'noise_lvl':float(args.noise_level), 'noise_sparsity':float(args.noise_sparsity)}
	return LabelSmoothClient(cid=cid, num_clients=int(args.num_clients), smooth_rate=float(args.smooth_rate), model_loader=load_model, data_loader=load_train_data, **kwargs)

def create_server():
	os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
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
	return history

if __name__ == "__main__":
	print(run_simulation())
