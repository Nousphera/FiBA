# Modified for the FiBA standalone release, 2026-10-15; see PROVENANCE.md.
import torch
import collections
import flwr as fl
import sys
import logging
import os
import pickle
import numpy as np
from flwr.common.logger import log


def rankme(x, normalize=True):
	if normalize:
		x = torch.nn.functional.normalize(x, p=2, dim=1)
	_, s, _ = torch.linalg.svd(x, full_matrices=False)
	s = s / torch.sum(torch.abs(s))
	entropy = torch.exp(-torch.sum(s * torch.log(s)))
	return entropy.item()

def save_mask(mask, run_id):
	os.makedirs(os.path.dirname(f'../masks/{run_id}-attack-mask'), exist_ok=True)
	with open(f'../masks/{run_id}-attack-mask', 'wb') as outp:  # Overwrites any existing file.
		pickle.dump(mask, outp, pickle.HIGHEST_PROTOCOL)

def save_object(obj, attacker_id, run_id, attacker):
	os.makedirs(os.path.dirname(f'../attackers/{run_id}-{attacker_id}-{"attacker" if attacker else "client"}'), exist_ok=True)
	with open(f'../attackers/{run_id}-{attacker_id}-{"attacker" if attacker else "client"}', 'wb') as outp:  # Overwrites any existing file.
		pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def get_prev_parameters(attacker_id, run_id, attacker):
	file_path = f'../attackers/{run_id}-{attacker_id}-{"attacker" if attacker else "client"}'
	try:
		with open(file_path, 'rb') as f:
			return pickle.load(f)
	except FileNotFoundError:
		log(logging.INFO, f'File {file_path} not found')
		return None

def get_dataset_poisoned(client_id, run_id, train):
	if train:
		dataset_id = 'train'
	else:
		dataset_id = 'eval'

	file_path = f'../attack_data/{run_id}-{client_id}-{dataset_id}'
	try:
		with open(file_path, 'rb') as f:
			return pickle.load(f)
	except FileNotFoundError:
		log(logging.INFO, f'File {file_path} not found')
		return None

class Client(fl.client.NumPyClient):

	def __init__(self, cid, num_clients, attack_setting, model_loader, data_loader, data_split='iid', skewness=1.0,
		batch_size=128, data_name='cifar10', seed=42, device='cuda', run_id='', m=0):
		self.cid = int(cid)
		self.attack_setting = attack_setting
		self.model_loader = model_loader
		self.device = device
		self.round = 0
		self.scaling_factor = 10
		self.run_id = run_id
		self.m = m
		self.batch_size = batch_size
		self.bottom_mask = None
		self.top_mask = None
		(self.data, self.num_classes, self.num_samples) = data_loader(id=self.cid, name=data_name, data_split=data_split,
			skewness=skewness, num_clients=num_clients, batch_size=batch_size, seed=seed, run_id=run_id, attack_setting=attack_setting)

	def set_parameters(self, parameters, config):
		# Create model if not exists
		self.round = config['round']
		if not hasattr(self, 'model'):
			self.model = self.model_loader(num_classes=self.num_classes).to(self.device)
		# Update model weights
		params_dict = zip(self.model.state_dict().keys(), parameters)
		state_dict = collections.OrderedDict({k: torch.tensor(v)
			if not isinstance(v, torch.Tensor) else v for k,v in params_dict})
		self.model.load_state_dict(state_dict, strict=True)

	def get_weights(self):
		weights = []
		with torch.no_grad():
			for _,v in self.model.state_dict().items():
				weights.append(v.cpu().numpy())
		return weights

	def get_parameters(self, config=None):
		return self.get_weights()

	def get_weights_and_gradients(self):
		weights = []
		gradients = []
		with torch.no_grad():
			for _,v in self.model.state_dict().items():
				weights.append(v.cpu().numpy())
			for name, params in self.model.named_parameters():
				if params.requires_grad:
					gradients.append(params.grad.cpu().clone())
		return weights, gradients

	def multiply_params(self, params, factor):
		multiplied_params = [factor * p for p in params]
		return multiplied_params

	def divide_params(self, params1, params2):
		result = []

		if len(params1) != len(params2):
			raise ValueError("Arrays must have the same length")

		result = [tensor1 / tensor2 for tensor1, tensor2 in zip(params1, params2)]

		return result

	def subtract_arrays(self, arr1, arr2):
		result = []

		if len(arr1) != len(arr2):
			raise ValueError("Arrays must have the same length")

		result = [tensor1 - tensor2 for tensor1, tensor2 in zip(arr1, arr2)]

		return result

	def add_arrays(self, arr1, arr2):
		result = []

		if len(arr1) != len(arr2):
			raise ValueError("Arrays must have the same length")

		result = [tensor1 + tensor2 for tensor1, tensor2 in zip(arr1, arr2)]

		return result


	def fit(self, parameters, config):
		self.set_parameters(parameters, config)
		if self.round == 20 and self.cid in self.attack_setting.attackers:
			self.model.compile(lr=config['lr'])
			top_mask = self.model.get_activation_map(self.data)
			save_mask(top_mask, self.run_id)
			del self.model

		self.set_parameters(parameters, config)
		temp_params = self.get_weights()
		prev_parameters = None
		if self.cid in self.attack_setting.attackers:
			prev_parameters = get_prev_parameters(self.cid, self.run_id, self.cid in self.attack_setting.attackers)
		self.model.compile(lr=config['lr'])

		metrics = self.model.fit(ds=self.data, epochs=config['epochs'], verbose=False)
		params, gradients = self.get_weights_and_gradients()
		save_object(params, self.cid, self.run_id, self.cid in self.attack_setting.attackers)

        # model replacement
		if self.cid in self.attack_setting.attackers and self.attack_setting.boost_strat == 'model_replacement' and (self.round == 30 or self.round == 60 or self.round == 90):
			if prev_parameters:
				n = self.multiply_params(temp_params, self.m - 1)
				n = self.add_arrays(prev_parameters, n)
				n = self.multiply_params(n, self.num_samples)
				n = self.divide_params(n, temp_params)
				n = np.average(n[-1])
				params = self.multiply_params(params, factor=(n/self.num_samples))
				temp_params = self.multiply_params(temp_params, factor=self.m - 1)
				params = self.subtract_arrays(params, temp_params)

		self.evaluate()

		del self.model; torch.cuda.empty_cache();
		# Stock Flower transports the final classifier gradient as ndarray bytes.
		return params, int(self.num_samples), {
			"loss": float(metrics["loss"][-1]), "accuracy": float(metrics["accuracy"]),
			"client_id": int(self.cid),
			"gradient": fl.common.ndarray_to_bytes(gradients[-2].numpy()),
		}

	def evaluate(self):
		poisoned_eval = get_dataset_poisoned(self.cid, self.run_id, True)

		# # get the model score
		# embeddings = self.model.extract_embeddings(self.data)
		# score = rankme(embeddings, True)
		# log(logging.INFO, f' Client Score: {score}')

		if poisoned_eval:
			poisoned_eval = torch.utils.data.DataLoader(poisoned_eval, batch_size=self.batch_size, shuffle=True)
		metrics = self.model.evaluate(ds=self.data, attack_setting=self.attack_setting, verbose=False, poisoned_eval=poisoned_eval)
		log(logging.INFO, f' Client: {self.cid}, "attacker":{self.cid in self.attack_setting.attackers} "accuracy":{metrics[1]}, "class-acc": {metrics[2]}, "asr": {metrics[3]}')
