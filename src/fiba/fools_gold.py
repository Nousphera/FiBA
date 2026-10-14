# Modified for the FiBA standalone release, 2026-10-15; see PROVENANCE.md.
'''
Implementation based on "The Limitations of Federated Learning in Sybil Settings."
Clement Fung, Chris J.M. Yoon, Ivan Beschastnikh.
23rd International Symposium on Research in Attacks, Intrusions and Defenses (RAID) 2020.

Github page: https://github.com/DistributedML/FoolsGold
'''

import flwr as fl
import logging
import functools
import numpy as np
from flwr.common.logger import log
import sklearn.metrics.pairwise as smp
from fractions import Fraction

def weighted_averaging(results, rescale):
	num_clients = len(results)
	   # Modify this line to include rescale for each layer within a client
	weighted_weights = [[layer * rescale[i] for layer in weights] for i, (weights, num_examples) in enumerate(results)]
	weights_prime = [functools.reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)]
	return weights_prime

def normalize_and_convert_to_fractions(trust_values):
    # Normalize trust values
    total_trust = sum(trust_values)
    normalized_trust = [value / total_trust for value in trust_values]

    # Convert normalized trust values to floating-point numbers
    numbers_list = [float(value) for value in normalized_trust]

    return numbers_list

# Takes in grad
# Compute similarity
# Get weightings
def foolsgold(grads):
	n_clients = grads.shape[0]
	cs = smp.cosine_similarity(grads) - np.eye(n_clients)
	maxcs = np.max(cs, axis=1)
    # pardoning
	for i in range(n_clients):
		for j in range(n_clients):
			if i == j:
				continue
			if maxcs[i] < maxcs[j]:
				cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
	wv = 1 - (np.max(cs, axis=1))
	wv[wv > 1] = 1
	wv[wv < 0] = 0

    # Rescale so that max value is wv
	wv = wv / np.max(wv)
	wv[(wv == 1)] = .99

    # Logit function
	wv = (np.log(wv / (1 - wv)) + 0.5)
	wv[(np.isinf(wv) + wv > 1)] = 1
	wv[(wv < 0)] = 0

	return wv

# You can adjust the weigths aggregation strategy here.
class AggregationStrategyFoolsGold(fl.server.strategy.FedAvg):

	def __init__(self, *args, **kwargs):
		super(AggregationStrategyFoolsGold, self).__init__(*args, **kwargs)
		self.memory = {}

	def evaluate(self, server_round, parameters):
		if self.evaluate_fn is None:
			return None
		# Evaluate model
		eval_res = None
		if self.evaluate_fn is not None and server_round>0:
			parameters_ndarrays = fl.common.parameters_to_ndarrays(parameters)
			eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
			log(logging.INFO,f'[Round {server_round}] - Loss: {eval_res[0]:.3f} - Accuracy: {100. * eval_res[1]["accuracy"]:.2f}')
		# Aggregate performance across tasks:
		if eval_res is None:
			return None
		return eval_res[0], eval_res[1]

	def configure_fit(self, server_round, parameters, client_manager):
		config = {}
		if self.on_fit_config_fn is not None:
			config = self.on_fit_config_fn(server_round)
		sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
		log(logging.INFO,f'[Round {server_round}] - Start of FL round with {sample_size} clients [using lr: {round(config["lr"],4)}].')
		clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
		fit_ins = fl.common.FitIns(parameters, config)
		return [(client, fit_ins) for client in clients]

	def aggregate_fit(self, server_round, results, failures,):
		if not results:
			return None, {}
		if not self.accept_failures and failures:
			return None, {}
		# Aggregate clients weights for each task seperately
		log(logging.INFO,f'[Round {server_round}] - Server-side weight aggregation starts with {len(failures)} failures. Using FoolsGold average')
		weights = [(fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]

		# IDs and ndarray bytes use the public Flower metrics transport.
		ids = [int(fit_res.metrics["client_id"]) for _, fit_res in results]
		if len(set(ids)) != len(ids):
			raise ValueError("Duplicate client IDs in FoolsGold round")
		gradients = [fl.common.bytes_to_ndarray(fit_res.metrics["gradient"]).reshape(-1)
			for _, fit_res in results]
		for cid, gradient in zip(ids, gradients):
			if not np.all(np.isfinite(gradient)):
				raise ValueError("Non-finite classifier gradient")
			self.memory[cid] = self.memory.get(cid, np.zeros_like(gradient)) + gradient
		# Preserve identity across partial-participation rounds.
		wv = foolsgold(np.stack([self.memory[cid] for cid in ids]))
		sorted_weights = weights
		if not np.all(np.isfinite(wv)) or np.sum(wv) <= 0:
			raise ValueError("FoolsGold assigned no finite positive trust; cannot aggregate")

		wv = normalize_and_convert_to_fractions(wv)
		log(logging.INFO,f'weighted values into fractions: {wv}')
		aggregated_weights = weighted_averaging(sorted_weights, wv)
		parameters_aggregated = fl.common.ndarrays_to_parameters(aggregated_weights)

		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}
		if self.fit_metrics_aggregation_fn:
			fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
			metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
		return parameters_aggregated, metrics_aggregated
