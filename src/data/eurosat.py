import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import tensorflow as tf

from utils.image_utils import train_prep_eurosat as train_prep
from utils.image_utils import valid_prep_eurosat as valid_prep
from utils.image_utils import cutout_eurosat as cutout
from utils.noise import inject_synthetic_noise_to_labels
from utils.shard import shard_data

DATA_FP = f"{os.environ['TFDS_DATA_DIR']}/eurosat/eurosat.npz"
AUTO = tf.data.experimental.AUTOTUNE

def load_eurosat(shard_id=0, num_shards=None, batch_size=128, seed=42, noisy_clients_frac=0.6, noise_lvl=0.4, noise_sparsity=0.4, corrected_labels=None, shuffle=True):
	np.random.seed(seed)
	data = np.load(DATA_FP)
	images, labels = data['images'][:21600], data['labels'][:21600]
	num_classes = len(np.unique(labels))
	# Get shard of dataset
	if num_shards is not None:
		(images,labels), idxs = shard_data(data=(images,labels), id=shard_id, num_shards=num_shards)
	labels = np.array(tf.keras.utils.to_categorical(labels, num_classes), dtype=np.float32)
	if corrected_labels is not None: labels = corrected_labels
	# Inject synthetic noise
	noisy_shards = np.random.choice(a=range(num_shards), size=min(num_shards,int(round(noisy_clients_frac*num_shards))), replace=False) if num_shards is not None else [shard_id]
	if (noise_lvl is not None) and (noise_sparsity is not None) and (corrected_labels is None) and (shard_id in noisy_shards):
		labels = inject_synthetic_noise_to_labels(shard_id, labels, num_classes, level=noise_lvl, sparsity=noise_sparsity, seed=seed, theshold=5e-2)
	ds = tf.data.Dataset.from_tensor_slices((images,labels))
	ds = ds.map(train_prep, num_parallel_calls=AUTO)
	if shuffle: ds = ds.shuffle(buffer_size=10000, seed=seed, reshuffle_each_iteration=True)
	ds = ds.batch(batch_size).map(cutout, num_parallel_calls=AUTO).prefetch(AUTO)
	return ds, num_classes, labels.shape[0], labels, idxs if num_shards is not None else None

def load_eurosat_test(batch_size=128):
	data = np.load(DATA_FP)
	images, labels = data['images'][21600:], data['labels'][21600:]
	num_classes = len(np.unique(labels))
	labels = np.array(tf.keras.utils.to_categorical(labels, num_classes), dtype=np.float32)
	ds = tf.data.Dataset.from_tensor_slices((images,labels)).map(valid_prep, num_parallel_calls=AUTO).batch(batch_size)
	ds = ds.prefetch(AUTO)
	return ds, num_classes, labels.shape[0]

