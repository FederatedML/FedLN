import tensorflow as tf
import tensorflow_addons as tfa

STD_IMAGENET = tf.reshape((0.2023, 0.1994, 0.2010), shape=(1, 1, 3))
MEAN_IMAGENET = tf.reshape((0.4914, 0.4822, 0.4465), shape=(1, 1, 3))
STD_PATHMNIST = tf.reshape((0.5, 0.5, 0.5), shape=(1, 1, 3))
MEAN_PATHMNIST = tf.reshape((0.5, 0.5, 0.5), shape=(1, 1, 3))

def train_prep_cifar(x, y):
	x = tf.image.convert_image_dtype(x, tf.float32) / 255.0
	x = tf.image.random_flip_left_right(x)
	x = tf.image.pad_to_bounding_box(x, 4, 4, 40, 40)
	x = tf.image.random_crop(x, (32, 32, 3))
	x = (x - MEAN_IMAGENET) / STD_IMAGENET
	return x, y

def valid_prep_cifar(x, y):
	x = tf.image.convert_image_dtype(x, tf.float32) / 255.0
	x = (x - MEAN_IMAGENET) / STD_IMAGENET
	return x, y

def train_prep_fmnist(x, y):
	x = tf.image.convert_image_dtype(x, tf.float32) / 255.0
	x = tf.expand_dims(x, axis=-1)
	x = tf.image.random_flip_left_right(x)
	return x, y

def valid_prep_fmnist(x, y):
	x = tf.image.convert_image_dtype(x, tf.float32) / 255.0
	x = tf.expand_dims(x, axis=-1)
	return x, y

def train_prep_pathmnist(x, y):
	x = tf.image.convert_image_dtype(x, tf.float32) / 255.0
	x = (x - MEAN_PATHMNIST) / STD_PATHMNIST
	return x, y

def valid_prep_pathmnist(x, y):
	x = tf.image.convert_image_dtype(x, tf.float32) / 255.0
	x = (x - MEAN_PATHMNIST) / STD_PATHMNIST
	return x, y

def train_prep_eurosat(x, y):
	x = tf.image.convert_image_dtype(x, tf.float32)
	x = tf.image.random_flip_left_right(x)
	x = tf.image.pad_to_bounding_box(x, 4, 4, 72, 72)
	x = tf.image.random_crop(x, (64, 64, 3))
	return x, y

def valid_prep_eurosat(x, y):
	x = tf.image.convert_image_dtype(x, tf.float32)
	return x, y

def cutout_cifar(images, labels):
	_images = tfa.image.cutout(images, mask_size=(16,16))
	return _images, labels

def cutout_fmnist(images, labels):
	_images = tfa.image.cutout(images, mask_size=(14,14))
	return _images, labels

def cutout_pathmnist(images, labels):
	_images = tfa.image.cutout(images, mask_size=(14,14))
	return _images, labels

def cutout_eurosat(images, labels):
	_images = tfa.image.cutout(images, mask_size=(32,32))
	return _images, labels