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
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_clients',			type=int,	default=3,					required=False)
parser.add_argument('--num_rounds',				type=int,	default=10,					required=False)
parser.add_argument('--participation_rate',		type=float,	default=1.0,				required=False)
parser.add_argument('--batch_size',				type=int,	default=128,				required=False)
parser.add_argument('--train_epochs',			type=int,	default=1,					required=False)
parser.add_argument('--lr',						type=float,	default=1e-3,				required=False)
parser.add_argument('--percentile',				type=int,	default=75,					required=False)
parser.add_argument('--noisy_frac',				type=float,	default=0.8,				required=False)
parser.add_argument('--noise_level',			type=float,	default=0.4,				required=False)
parser.add_argument('--noise_sparsity',			type=float,	default=0.7,				required=False)
parser.add_argument('--est_noise_round',		type=int,	default=25,					required=False)
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
	from utils.energy_noise_estimation import compute_energy_score
	#############################################
	# Override Client to use Noise-Aware FedAvg #
	#############################################
	import numpy as np

	class NACClient(Client):
		def __init__(self, compute_round, temp_dir, *args, **kwargs):
			super(NACClient, self).__init__(*args, **kwargs)
			self.compute_round = compute_round
			self.temp_dir = temp_dir


		def fit(self, parameters, config):

			# Set parameters
			self.set_parameters(parameters, config)

			# Compute scores before training
			if int(config['round'])==self.compute_round:
				score = compute_energy_score(model=self.model, data=self.data, cid=self.cid)

			# Client local update
			h = self.model.fit(self.data, epochs=config['epochs'], verbose=0)

			# Compute scores after training
			if int(config['round'])==self.compute_round:
				score = score = np.concatenate((score, compute_energy_score(model=self.model, data=self.data)), axis=-1)
				if not os.path.isdir(f"{self.temp_dir}"): os.makedirs(f"{self.temp_dir}")
				with open(f'{self.temp_dir}/score_{cid}.npy','wb') as f: np.save(f,score)

			metrics = {'accuracy':float(h.history['accuracy'][-1]), 'loss':float(h.history['loss'][-1]), 'cid':self.cid}
			return self.get_parameters(), self.num_samples, metrics

	#############################################

	load_model = load_available_models()[args.model_name]
	load_train_data = load_available_datasets()[args.dataset_name]
	kwargs = {'batch_size':int(args.batch_size), 'seed':int(args.seed), 'noisy_clients_frac':float(args.noisy_frac), 
        'noise_lvl':float(args.noise_level), 'noise_sparsity':float(args.noise_sparsity)}
	return NACClient(compute_round=int(args.est_noise_round), cid=cid, num_clients=int(args.num_clients),
                model_loader=load_model, data_loader=load_train_data, temp_dir=args.temp_dir, **kwargs)

def create_server():
	os.environ['CUDA_VISIBLE_DEVICES'] = grab_gpu()
	from utils.flwr_server import _Server as Server
	from utils.energy_noise_estimation import compute_noise_estimation
	from utils.energy_noise_estimation import NAFedAvg
	import numpy as np

	#############################################
	# Override Server to use Noise-Aware FedAvg #
	#############################################
	class NAServer(Server):
		def __init__(self, compute_round, percentile, *args, **kwargs):
			self.clients_impact = None
			self.percentile = percentile
			self.compute_round = compute_round
			super(NAServer, self).__init__(*args, **kwargs)

		def set_strategy(self, *_):
			self.strategy = NAFedAvg(
				compute_round=self.compute_round,
				num_clients=self.num_clients,
				set_clients_impact_fn=self.set_clients_impact_fn,
				percentile=self.percentile,
				min_available_clients=self.num_clients,
				fraction_fit=self.participation,
				min_fit_clients=int(self.participation*self.num_clients),
				fraction_evaluate=0.0,
				min_evaluate_clients=0,
				evaluate_fn=self.get_evaluation_fn(),
				on_fit_config_fn=self.get_client_config_fn(),
				initial_parameters=self.get_initial_parameters(),
			)

		@staticmethod
		def set_clients_impact_fn(percentile=75, temp_dir=args.temp_dir):
			find_score_files = lambda : glob.glob(f'{temp_dir}/score_*.npy')
			if len(find_score_files())>0:
				return compute_noise_estimation(scores=np.concatenate([np.load(f) for f in find_score_files()]), percentile=percentile,)
			return None
	#############################################

	load_model = load_available_models()[args.model_name]
	load_test_data = load_available_datasets(train=False)[args.dataset_name]
	kwargs = {'lr':float(args.lr), 'train_epochs':int(args.train_epochs)}
	return NAServer(compute_round=int(args.est_noise_round), percentile=int(args.percentile), num_rounds=int(args.num_rounds),
            num_clients=int(args.num_clients), participation=float(args.participation_rate), model_loader=load_model, 
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
