# Modified for the FiBA standalone release, 2026-10-15; see PROVENANCE.md.
import flwr as fl
import logging
import functools
import numpy as np
from flwr.common.logger import log

# Simple fedavg aggregation
def weighted_averaging(results):
	num_examples_total = sum([num_examples for _, num_examples in results])
	weighted_weights = [[layer * num_examples for layer in weights] for weights, num_examples in results]
	weights_prime = [functools.reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)]
	return weights_prime

def median(results):
    weights = [weights for weights, _ in results]
    # Compute median weight of each layer
    median_w = [np.median(np.asarray(layer), axis=0) for layer in zip(*weights)]
    return median_w

def trim_mean(array, proportiontocut):
    axis = 0
    nobs = array.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if lowercut > uppercut:
        raise ValueError("Proportion too big.")

    atmp = np.partition(array, (lowercut, uppercut - 1), axis)

    slice_list = [slice(None)] * atmp.ndim
    slice_list[axis] = slice(lowercut, uppercut)
    result = np.mean(atmp[tuple(slice_list)], axis=axis)
    return result

def trimmed_mean(
    results, proportiontocut):
    """Compute trimmed average."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    trimmed_w = [
        trim_mean(np.asarray(layer), proportiontocut=proportiontocut)
        for layer in zip(*weights)
    ]

    return trimmed_w

class AggregationStrategyAvg(fl.server.strategy.FedAvg):

	def __init__(self, *args, **kwargs):
		super(AggregationStrategyAvg, self).__init__(*args, **kwargs)

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
		log(logging.INFO,f'[Round {server_round}] - Server-side weight aggregation starts with {len(failures)} failures. Using fedavg average')
		weights = [(fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
		aggregated_weights = weighted_averaging(weights)
		parameters_aggregated = fl.common.ndarrays_to_parameters(aggregated_weights)
		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}
		if self.fit_metrics_aggregation_fn:
			fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
			metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
		return parameters_aggregated, metrics_aggregated

class AggregationStrategyMedian(fl.server.strategy.FedAvg):

	def __init__(self, *args, **kwargs):
		super(AggregationStrategyMedian, self).__init__(*args, **kwargs)

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
		log(logging.INFO,f'[Round {server_round}] - Server-side weight aggregation starts with {len(failures)} failures. Using median.')
		weights = [(fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
		aggregated_weights = median(weights)
		parameters_aggregated = fl.common.ndarrays_to_parameters(aggregated_weights)
		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}
		if self.fit_metrics_aggregation_fn:
			fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
			metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
		return parameters_aggregated, metrics_aggregated

    # You can adjust the weigths aggregation strategy here.
class AggregationStrategyTrimmedMean(fl.server.strategy.FedAvg):

	def __init__(self, *args, **kwargs):
		super(AggregationStrategyTrimmedMean, self).__init__(*args, **kwargs)

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
		log(logging.INFO,f'[Round {server_round}] - Server-side weight aggregation starts with {len(failures)} failures. Using trimmed mean.')
		weights = [(fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]
		aggregated_weights = trimmed_mean(weights, 0.3)
		parameters_aggregated = fl.common.ndarrays_to_parameters(aggregated_weights)
		# Aggregate custom metrics if aggregation fn was provided
		metrics_aggregated = {}
		if self.fit_metrics_aggregation_fn:
			fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
			metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
		return parameters_aggregated, metrics_aggregated
