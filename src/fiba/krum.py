# Modified for the FiBA standalone release, 2026-10-15; see PROVENANCE.md.
import flwr as fl
import logging
import functools
import numpy as np
from flwr.common.logger import log

def _compute_distances(weights):
    """Compute distances between vectors.

    Input: weights - list of weights vectors
    Output: distances - matrix distance_matrix of squared distances between the vectors
    """
    flat_w = np.array([np.concatenate(p, axis=None).ravel() for p in weights])
    distance_matrix = np.zeros((len(weights), len(weights)))
    for i, flat_w_i in enumerate(flat_w):
        for j, flat_w_j in enumerate(flat_w):
            delta = flat_w_i - flat_w_j
            norm = np.linalg.norm(delta)
            distance_matrix[i, j] = norm**2
    return distance_matrix

def weighted_averaging(results):
    num_examples_total = sum([num_examples for _, num_examples in results])
    weighted_weights = [[layer * num_examples for layer in weights] for weights, num_examples in results]
    weights_prime = [functools.reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)]
    return weights_prime

def aggregate_krum(results, num_malicious: int, to_keep: int):
    """Choose one parameter vector according to the Krum function.

    If to_keep is not None, then MultiKrum is applied.
    """
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute distances between vectors
    distance_matrix = _compute_distances(weights)

    # For each client, take the n-f-2 closest parameters vectors
    num_closest = max(1, len(weights) - num_malicious - 2)
    closest_indices = []
    for distance in distance_matrix:
        closest_indices.append(
            np.argsort(distance)[1 : num_closest + 1].tolist()  # noqa: E203
        )

    # Compute the score for each client, that is the sum of the distances
    # of the n-f-2 closest parameters vectors
    scores = [
        np.sum(distance_matrix[i, closest_indices[i]])
        for i in range(len(distance_matrix))
    ]

    log(logging.INFO,f'to keep - {np.argmin(scores)}')
    if to_keep > 0:
        log(logging.INFO,f'keeping scores - {np.argsort(scores)}')
        # Choose to_keep clients and return their average (MultiKrum)
        best_indices = np.argsort(scores)[::-1][len(scores) - to_keep :]  # noqa: E203
        best_results = [results[i] for i in best_indices]
        return weighted_averaging(best_results)

    log(logging.INFO,f'chosen client - {np.argmin(scores)}')
    # Return the model parameters that minimize the score (Krum)
    return weights[np.argmin(scores)]

# You can adjust the weigths aggregation strategy here.
class AggregationStrategyKrum(fl.server.strategy.FedAvg):

    def __init__(self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        num_malicious_clients: int = 0,
        num_clients_to_keep: int = 0,
        evaluate_fn = None,
        on_fit_config_fn = None,
        on_evaluate_config_fn = None,
        accept_failures: bool = True,
        initial_parameters = None,
        fit_metrics_aggregation_fn = None,
        evaluate_metrics_aggregation_fn = None,):
        super(AggregationStrategyKrum, self).__init__(fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.num_malicious_clients = num_malicious_clients
        self.num_clients_to_keep = num_clients_to_keep

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
        log(logging.INFO,f'[Round {server_round}] - Server-side weight aggregation starts with {len(failures)} failures. Using (multi)-Krum average')
        weights = [(fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results]

        ids = [int(fit_res.metrics["client_id"]) for _, fit_res in results]
		# Combine and sort the lists based on client IDs
        sorted_weights = sorted(zip(ids, weights), key=lambda x: x[0])
        sorted_ids, sorted_weights = zip(*sorted_weights)

        aggregated_weights = aggregate_krum(sorted_weights, self.num_malicious_clients, self.num_clients_to_keep)
        parameters_aggregated = fl.common.ndarrays_to_parameters(aggregated_weights)
		# Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(fit_res.num_examples, fit_res.metrics) for _, fit_res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        return parameters_aggregated, metrics_aggregated
