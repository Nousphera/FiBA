from .runtime import cache_path
# Modified for FiBA, 2026-10-15. Component terms: licenses/THIRD_PARTY.txt.
import torch
import collections
import flwr as fl
from .aggregation_strategies import AggregationStrategyAvg, AggregationStrategyMedian, AggregationStrategyTrimmedMean
from .fools_gold import AggregationStrategyFoolsGold
from .krum import AggregationStrategyKrum
import pickle
import logging
from flwr.common.logger import log
from . import utils
import copy

def get_dataset_poisoned(client_id, run_id, train):
	if train:
		dataset_id = 'train'
	else:
		dataset_id = 'eval'

	file_path = cache_path(f'attack_data/{run_id}-{client_id}-{dataset_id}')
	try:
		with open(file_path, 'rb') as f:
			return pickle.load(f)
	except FileNotFoundError:
		log(logging.INFO, f'File {file_path} not found')
		return None

class Server(fl.server.Server):

	def __init__(self, model_loader, data_loader, num_rounds, attack_setting, run_id, num_clients=10,
		participation=1.0, batch_size=128, data_name='cifar10', num_epochs=1,
		lr=1e-3, device='cuda:0', strategy='avg', krum_clients_to_keep=0):
		self.num_rounds = num_rounds
		self.model_loader = model_loader
		self.num_clients = num_clients
		self.participation = participation
		self.device = device
		self.max_workers = None
		self.clients_config = {"epochs":num_epochs, "lr": lr}
		self.batch_size = batch_size
		self.run_id = run_id
		(self.ds_test, self.num_classes, self.num_samples) = data_loader(name=data_name,
			num_clients=self.num_clients, batch_size=batch_size, return_eval_ds=True, attack_setting=attack_setting, run_id=run_id)
		self.attack_setting = attack_setting
		self.avg_strategy = strategy
		self.krum_clients_to_keep = krum_clients_to_keep
		self.set_strategy(self)
		self._client_manager = fl.server.client_manager.SimpleClientManager()

	def set_max_workers(self, *args, **kwargs):
		return super(Server, self).set_max_workers(*args, **kwargs)

	def set_strategy(self, *_):
		if self.avg_strategy == 'avg':
			self.strategy = AggregationStrategyAvg(
				min_available_clients=self.num_clients,
				fraction_fit=self.participation,
				min_fit_clients=max(1,int(self.participation*self.num_clients)),
				fraction_evaluate=0.0,
				min_evaluate_clients=0,
				evaluate_fn=self.get_evaluation_fn(),
				on_fit_config_fn=self.get_client_config_fn(),
				initial_parameters=self.get_initial_parameters(),
			)
		elif self.avg_strategy == 'median':
			self.strategy = AggregationStrategyMedian(
				min_available_clients=self.num_clients,
				fraction_fit=self.participation,
				min_fit_clients=max(1,int(self.participation*self.num_clients)),
				fraction_evaluate=0.0,
				min_evaluate_clients=0,
				evaluate_fn=self.get_evaluation_fn(),
				on_fit_config_fn=self.get_client_config_fn(),
				initial_parameters=self.get_initial_parameters(),
			)
		elif self.avg_strategy == 'trimmed':
			self.strategy = AggregationStrategyTrimmedMean(
				min_available_clients=self.num_clients,
				fraction_fit=self.participation,
				min_fit_clients=max(1,int(self.participation*self.num_clients)),
				fraction_evaluate=0.0,
				min_evaluate_clients=0,
				evaluate_fn=self.get_evaluation_fn(),
				on_fit_config_fn=self.get_client_config_fn(),
				initial_parameters=self.get_initial_parameters(),
			)
		elif self.avg_strategy == 'fg':
			self.strategy = AggregationStrategyFoolsGold(
				min_available_clients=self.num_clients,
				fraction_fit=self.participation,
				min_fit_clients=max(1,int(self.participation*self.num_clients)),
				fraction_evaluate=0.0,
				min_evaluate_clients=0,
				evaluate_fn=self.get_evaluation_fn(),
				on_fit_config_fn=self.get_client_config_fn(),
				initial_parameters=self.get_initial_parameters(),
			)
		elif self.avg_strategy == 'krum':
			self.strategy = AggregationStrategyKrum(
				min_available_clients=self.num_clients,
				fraction_fit=self.participation,
				min_fit_clients=max(1,int(self.participation*self.num_clients)),
				fraction_evaluate=0.0,
				min_evaluate_clients=0,
				evaluate_fn=self.get_evaluation_fn(),
				on_fit_config_fn=self.get_client_config_fn(),
				initial_parameters=self.get_initial_parameters(),
				num_malicious_clients=len(self.attack_setting.attackers),
				num_clients_to_keep=self.krum_clients_to_keep
			)

		else :
			raise NotImplementedError('No valid aggregation strategy')

	def client_manager(self, *args, **kwargs):
		return super(Server, self).client_manager(*args, **kwargs)

	def get_parameters(self, config={}):
		pass

	def set_parameters(self, parameters, config):
		# Create model if not exists
		if not hasattr(self, 'model'):
			self.model= self.model_loader(num_classes=self.num_classes).to(self.device)
		# Update model weights
		params_dict = zip(self.model.state_dict().keys(), parameters)
		state_dict = collections.OrderedDict({k: torch.tensor(v)
			if not isinstance(v, torch.Tensor) else v for k, v in params_dict})
		self.model.load_state_dict(state_dict, strict=True)

	def get_initial_parameters(self, *_):
		weights = self.model_loader(num_classes=self.num_classes).state_dict()

		state_dict = collections.OrderedDict({k: v.cpu().numpy() for k, v in weights.items()})
		return fl.common.ndarrays_to_parameters(list(state_dict.values()))

	def get_evaluation_fn(self):
		def evaluation_fn(rnd, parameters, config):
			poisoned_eval = get_dataset_poisoned('server', self.run_id, False)
			if poisoned_eval:
				poisoned_eval = torch.utils.data.DataLoader(poisoned_eval, batch_size=self.batch_size, shuffle=False)
			self.set_parameters(parameters, config)
			self.model.compile(lr=1e-2)
			metrics = self.model.evaluate(ds=self.ds_test, attack_setting=self.attack_setting, verbose=False, poisoned_eval=poisoned_eval)
			if rnd == 100:
				torch.save(self.model.state_dict(), cache_path(f'{self.run_id}.th'))
			del self.model; torch.cuda.empty_cache();
			return metrics[0], {"accuracy":metrics[1], "class-acc": str(metrics[2].tolist()), "asr": metrics[3]}

		return evaluation_fn

	def get_client_config_fn(self):
		def get_on_fit_config_fn(rnd):
			self.clients_config["round"] = rnd
			return self.clients_config
		return get_on_fit_config_fn
