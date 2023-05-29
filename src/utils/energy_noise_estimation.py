import tensorflow as tf
import numpy as np
import flwr as fl
from logging import WARNING

@tf.function
def energy_score_fn(logits, temperature=1.0):
    return temperature * tf.math.reduce_logsumexp(logits / temperature, axis=-1)

def compute_energy_score(model, data, cid=None):
	score = np.expand_dims(np.concatenate([energy_score_fn(model.predict(x, verbose=0)).numpy() for x,_ in data], dtype=np.float32), axis=-1)
	if cid is not None: score = np.concatenate((float(cid)*np.ones_like(score, dtype=np.float32),score), axis=-1)
	return score

def compute_noise_estimation(scores, percentile=75, noise_threshold=0.15, min_effect=0.1):
	nu_percentile = np.percentile(scores[:,1], q=percentile)
	est_noise_mask = scores[scores[:,2]>=nu_percentile]
	num_clients = len(np.unique(scores[:,0]))
	num_samples = [scores[scores[:,0]==i].shape[0] for i in range(num_clients)]
	num_noisy_samples = [est_noise_mask[est_noise_mask[:,0]==i].shape[0] for i in range(num_clients)]
	estimated_noise = [float(num_noisy_samples[i]/num_samples[i]) if (num_noisy_samples[i]/num_samples[i])>=noise_threshold else 0.0 for i in range(num_clients)]
	[print(f"Noise estimation of client {i} is {100*estimated_noise[i]:.2f}") for i in range(num_clients)]
	noise_aware_weights = {i: min(min_effect*num_samples[i],num_samples[i]-num_noisy_samples[i]) if estimated_noise[i]>0.0 else num_samples[i] for i in range(num_clients)}
	return noise_aware_weights

class NAFedAvg(fl.server.strategy.FedAvg):

	def __init__(self, compute_round, num_clients, set_clients_impact_fn, percentile, *args, **kwargs):
		super(NAFedAvg, self).__init__(*args, **kwargs)
		self.compute_round=compute_round
		self.num_clients=num_clients
		self.set_clients_impact_fn = set_clients_impact_fn
		self.percentile = percentile
		self.clients_impact=None

	def configure_fit(self, server_round, parameters, client_manager):
		config = {}
		if self.on_fit_config_fn is not None: config = self.on_fit_config_fn(server_round)
		fit_ins = fl.common.FitIns(parameters, config)
		# Sample clients
		sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
		if server_round != self.compute_round:
			clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
		else: # Sample all clients when we estimate noise across devices.
			clients = client_manager.sample(num_clients=self.num_clients, min_num_clients=self.num_clients)
		# Return client/config pairs
		return [(client, fit_ins) for client in clients]

	def aggregate_fit(self, server_round, results, failures):

		# Compute clients impact
		if self.clients_impact is None: self.clients_impact = self.set_clients_impact_fn(percentile=self.percentile)
		if not results: return None, {}
        # Do not aggregate if there are failures and failures are not accepted
		if not self.accept_failures and failures: return None, {}
        # Convert results
		if self.clients_impact is None:
			weights_results = [(fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
		else:
			weights_results = [(fl.common.parameters_to_ndarrays(fit_res.parameters), self.clients_impact[int(fit_res.metrics['cid'])]) for _, fit_res in results]
		parameters_aggregated = fl.common.ndarrays_to_parameters(fl.server.strategy.aggregate.aggregate(weights_results))
		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}
		if self.fit_metrics_aggregation_fn:
			fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
			metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
		elif server_round == 1: fl.common.logger.log(WARNING, "No fit_metrics_aggregation_fn provided")

		return parameters_aggregated, metrics_aggregated