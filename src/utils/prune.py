import functools
import numpy as np
import cleanlab as cl
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold as _StratifiedKFold

def prune_with_cleanlab(data, num_classes, num_samples, model, compile_args, labels=None, num_folds=5, train_epochs=10, batch_size=128, verbose=True):
	# Unbatch data
	data = data.unbatch()
	# Create folds of the dataset
	kfold_data, labels = StratifiedKFold(ds=data, labels=labels, num_classes=num_classes, num_splits=num_folds)
	# Compute the probabilities
	psx = np.zeros((num_samples,num_classes))
	init_weights = model.get_weights()
	print(f'Prunning using Confidence Learning using {num_folds} folds.')
	for (train_ds, test_ds),(_, test_idx) in kfold_data:
		# Reset model to original state
		model.set_weights(init_weights)
		model.compile(optimizer=compile_args['optimizer']['fn'](**compile_args['optimizer']['args']),
			loss=compile_args['loss']['fn'](**compile_args['loss']['args']), metrics=[compile_args['metrics']['fn'](**compile_args['metrics']['args'])])
		# Train on fold
		model.fit(train_ds.batch(batch_size), epochs=train_epochs, verbose=0)
		# Predict probabilities
		psx[test_idx] = tf.nn.softmax(model.predict(test_ds.batch(1), verbose=0))
	# Cleanlab expects int labels
	if len(labels.shape)==2: labels = list(np.argmax(labels,axis=-1))
	# Get samples to be pruned based on confidence learning
	prune_idx = cl.filter.find_label_issues(labels,psx,filter_by="confident_learning",return_indices_ranked_by="self_confidence")
	# Create pruned dataset
	data = data.enumerate().filter(functools.partial(prune_fn, idxs=prune_idx)).map(tf.autograph.experimental.do_not_convert(lambda _,x: x), num_parallel_calls=tf.data.AUTOTUNE)
	data = data.batch(batch_size).prefetch(tf.data.AUTOTUNE)
	_num_samples = num_samples - prune_idx.shape[0]
	if verbose: print(f'Prunned {100*(prune_idx.shape[0]/num_samples):.02f} % of dataset [Number Samples: {num_samples} / {_num_samples}]')
	return data, _num_samples, prune_idx

def prune_from_idx(data, num_samples, idxs):
	data = data.enumerate().filter(functools.partial(prune_fn, idxs=idxs)).map(tf.autograph.experimental.do_not_convert(lambda _,x: x), num_parallel_calls=tf.data.AUTOTUNE)
	_num_samples = num_samples - idxs.shape[0]
	return data, _num_samples

@tf.function
def filter_fn(i,x, idxs):
	return tf.reduce_any(tf.equal(i, idxs), -1)

@tf.function
def prune_fn(i,x, idxs):
	return tf.reduce_any(tf.not_equal(i, idxs), -1)

def StratifiedKFold(ds, labels=None, num_classes=None, num_splits=2):

	kfold_data = []
	# Get labels
	if labels is None: labels = [y for _,y in ds]
	# Get labels shape
	one_hot_labels = len(labels.shape)==2
	# Convert to one hot labels
	if one_hot_labels: labels = list(np.argmax(labels,axis=-1))
	# Get number of classes
	if num_classes is None: num_classes = len(np.unique(labels))
	# Split to Starified Kfolds based on labels
	skf = _StratifiedKFold(n_splits=num_splits, shuffle=False)
	# Create td.data.Dataset objects for each split
	ds =  ds.enumerate()
	for train_idxs, test_idxs in skf.split(np.arange(len(labels)), labels):
		ds_train = ds.filter(functools.partial(filter_fn, idxs=train_idxs)).map(lambda _,x: x , num_parallel_calls=tf.data.AUTOTUNE)
		ds_test = ds.filter(functools.partial(filter_fn, idxs=test_idxs)).map(lambda _,x: x , num_parallel_calls=tf.data.AUTOTUNE)
		kfold_data.append(((ds_train, ds_test), (train_idxs,test_idxs)))
	# Convert back to one hot (if needed)
	if one_hot_labels: labels = np.array(tf.keras.utils.to_categorical(labels, num_classes), dtype=np.float32)

	return kfold_data, labels
