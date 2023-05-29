import flwr as fl
import tensorflow as tf

class _Client(fl.client.NumPyClient):

	def __init__(self, cid, num_clients, model_loader, data_loader, shuffle=True, load_data=True, **kwargs):
		self.cid = cid
		self.shuffle = shuffle
		if load_data:
			self.data, self.num_classes, self.num_samples, self.labels, self.idxs = data_loader(shard_id=int(cid), num_shards=num_clients, shuffle=self.shuffle,
				batch_size=kwargs['batch_size'], seed=kwargs['batch_size'], noisy_clients_frac=kwargs['noisy_clients_frac'], noise_lvl=kwargs['noise_lvl'], noise_sparsity=kwargs['noise_sparsity'])
			if shuffle: del self.labels # labels order is not preserved for shuffle==true
		self.model_loader = model_loader
		if load_data:
			self.input_shape = self.data.element_spec[0].shape

	def set_parameters(self, parameters, config):
		""" Set model weights """
		if not hasattr(self, 'model'):
			self.model = self.model_loader(input_shape=self.input_shape[1:], num_classes=self.num_classes)

		self.model.compile(
			optimizer=tf.keras.optimizers.Adam(learning_rate=config['lr']),
			loss=tf.keras.losses.CategoricalCrossentropy(name='loss', from_logits=True),
			metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')]
		)

		if parameters is not None:
			self.model.set_weights(parameters)

	def get_parameters(self, config={}):
		""" Get model weights """
		return self.model.get_weights()

	def fit(self, parameters, config):
		# Set parameters
		self.set_parameters(parameters, config)
		# Client local update
		h = self.model.fit(self.data, epochs=config['epochs'], verbose=0)
		metrics = {
			'accuracy':float(h.history['accuracy'][-1]),
			'loss':float(h.history['loss'][-1])
		}
		return self.get_parameters(), self.num_samples, metrics

	def evaluate(self, parameters, config):
		raise NotImplementedError('Client-side evaluation is not implemented!')