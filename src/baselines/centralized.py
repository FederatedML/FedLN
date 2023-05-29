import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',					type=int,	default=10,					required=False)
parser.add_argument('--batch_size',				type=int,	default=128,				required=False)
parser.add_argument('--lr',						type=float,	default=1e-3,				required=False)
parser.add_argument('--noisy_frac',				type=float,	default=0.8,				required=False)
parser.add_argument('--noise_level',			type=float,	default=0.4,				required=False)
parser.add_argument('--noise_sparsity',			type=float,	default=0.7,				required=False)
parser.add_argument('--dataset_name',			type=str,	default='eurosat',			required=False)
parser.add_argument('--model_name',				type=str, 	default='resnet20',			required=False)
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

def execute_experiment():
	load_model = load_available_models()[args.model_name]
	load_train_data = load_available_datasets()[args.dataset_name]
	load_test_data = load_available_datasets(train=False)[args.dataset_name]

	kwargs = {'batch_size':int(args.batch_size), 'seed':int(args.seed), 'noisy_clients_frac':float(args.noisy_frac),
		'noise_lvl':float(args.noise_level), 'noise_sparsity':float(args.noise_sparsity)}
	
	# Load train data
	ds_train, num_classes, _, _, _ = load_train_data(shard_id=0, **kwargs)
	# Create model
	input_shape = ds_train.element_spec[0].shape[1:]
	model = load_model(input_shape=input_shape , num_classes=num_classes)
	model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.lr)),
		loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, name='loss'),
		metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])
	# Train model
	h = model.fit(ds_train, epochs=int(args.epochs), verbose=2)
	# Evaluate model
	ds_test, _ = load_test_data(batch_size=int(args.batch_size))
	metrics = model.evaluate(ds_test)
	print(f'Model Performance in test set - Loss: {metrics[0]:0.4f} - Accuracy: {100*metrics[1]:0.2f}')
 
if __name__ == "__main__":
	execute_experiment()