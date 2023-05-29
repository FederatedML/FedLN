import logging
import flwr as fl
import tensorflow as tf

class _Server(fl.server.Server):

	def __init__(self, model_loader, data_loader, num_rounds, num_clients=10,
		participation=1.0, init_model=None, log_level=logging.INFO, **kwargs):

		self.num_rounds = num_rounds
		self.data, self.num_classes, self.num_samples = data_loader()
		self.model_loader = model_loader
		self.input_shape = self.data.element_spec[0].shape
		self.init_model = init_model # (.h5 format)
		self.clients_config = {"epochs": kwargs['train_epochs'], "lr": kwargs['lr']}
		self.num_clients = num_clients
		self.participation = participation
		self.set_strategy(self)
		self._client_manager = fl.server.client_manager.SimpleClientManager()
		self.max_workers = None
		logging.getLogger("flower").setLevel(log_level)

	def set_max_workers(self, *args, **kwargs):
		return super(_Server, self).set_max_workers(*args, **kwargs)

	def set_strategy(self, *_):
		self.strategy = fl.server.strategy.FedAvg(
			min_available_clients=self.num_clients,
			fraction_fit=self.participation,
			min_fit_clients=int(self.participation*self.num_clients),
			fraction_evaluate=0.0,
			min_evaluate_clients=0,
			evaluate_fn=self.get_evaluation_fn(),
			on_fit_config_fn=self.get_client_config_fn(),
			initial_parameters=self.get_initial_parameters(),
		)

	def client_manager(self, *args, **kwargs):
		return super(_Server, self).client_manager(*args, **kwargs)

	def get_parameters(self, config={}):
		""" Get model weights """
		return self.model.get_weights()

	def set_parameters(self, parameters, config):
		""" Set model weights """

		if not hasattr(self, 'model'):
			self.model = self.model_loader(
				input_shape=self.input_shape[1:],
				num_classes=self.num_classes
			)
		self.model.compile(metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')])

		if parameters is not None:
			self.model.set_weights(parameters)

	def get_initial_parameters(self, *_):
		""" Get initial random model weights """
		if self.init_model is not None:
			self.init_weights = tf.keras.models.load_model(self.init_model, compile=False).get_weights()
		else:
			self.init_weights = self.model_loader(input_shape=self.input_shape[1:], num_classes=self.num_classes).get_weights()
		return fl.common.ndarrays_to_parameters(self.init_weights)

	def get_evaluation_fn(self):
		" Define evaluation function with constant self objects."
		def evaluation_fn(rnd, parameters, config):
			# Update model parameters
			self.set_parameters(parameters, config)
			# Centralized evaluation
			metrics = self.model.evaluate(self.data, verbose=0)
			return metrics[0], {"accuracy":metrics[1]}
		return evaluation_fn

	def get_client_config_fn(self):
		" Define fit config function with constant self objects."
		def get_on_fit_config_fn(rnd):
			self.clients_config["round"] = rnd
			return self.clients_config
		return get_on_fit_config_fn