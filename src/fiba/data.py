# Modified for the FiBA standalone release, 2026-10-15; see PROVENANCE.md.
import os
import sys
import torch
import torchvision
import contextlib
import numpy as np
from .data_sources.dataloader import DATALOADER
import logging
from flwr.common.logger import log
import random
import pickle

from .plain_steganography import PoisonedDataset

def save_object(obj, client_id, run_id, train):
	if train:
		dataset_id = 'train'
	else:
		dataset_id = 'eval'

	print(f'../attack_data/{run_id}-{client_id}-{dataset_id}')
	os.makedirs(os.path.dirname(f'../attack_data/{run_id}-{client_id}-{dataset_id}'), exist_ok=True)
	with open(f'../attack_data/{run_id}-{client_id}-{dataset_id}', 'wb') as outp:  # Overwrites any existing file.
		pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def get_mask(run_id):
    file_path = f'../masks/{run_id}-attack-mask'
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
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
		return None

def iid_split(ds, num_clients, client_ids, lengths_list=None, seed=42):
	gen = torch.Generator().manual_seed(seed)
	if lengths_list is None:
		int_lengths = int(len(ds) / num_clients)
		lengths_list = [int_lengths]*(num_clients - 1)
		lengths_list.append(len(ds) - (int_lengths * (num_clients - 1)))
	splits = torch.utils.data.random_split(dataset=ds, lengths=lengths_list, generator=gen)
	return {cid: splits[i] for i, cid in enumerate(client_ids)}

def noniid_split(ds, num_clients, client_ids, num_classes, skewness=0.1, seed=42):
	np.random.seed(seed)
	partitions = {}
	idxs, num_samples, idx_batch = np.array(ds.targets), len(ds), [[] for _ in range(num_clients)]
	for k in range(num_classes):
		idx_k = np.where(idxs==k)[0]
		np.random.shuffle(idx_k)
		proportions = np.random.dirichlet(np.repeat(skewness, num_clients))
		proportions = np.array([p * (len(idx_j) < num_samples/num_clients) for p, idx_j in zip(proportions, idx_batch)])
		proportions = proportions / proportions.sum()
		proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
		idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
	for j in range(num_clients):
		np.random.shuffle(idx_batch[j])
		partitions[j] = idx_batch[j]

		# Count of items for each label for each client
		# label_counts = {label: sum(1 for idx in partitions[j] if idxs[idx] == label) for label in range(num_classes)}
		# print(f"Normal split Client {client_ids[j]} Label Counts: {label_counts}")
		# log(logging.INFO,f"Normal split Client {client_ids[j]} Label Counts: {label_counts}")

	return {cid: torch.utils.data.Subset(ds, partitions[i])  for i, cid in enumerate(client_ids)}

def split_dataset(ds, num_clients, client_ids, num_classes, chosen_ids, chosen_label, split='iid', skewness=0.1, seed=42):
	if split=='iid':
		return iid_split(ds=ds, num_clients=num_clients, client_ids=client_ids, seed=seed)
	elif split=='noniid':
		return noniid_split(ds=ds, num_clients=num_clients, client_ids=client_ids, num_classes=num_classes,
			skewness=skewness, seed=seed)
	else:
		raise ValueError("Invalid value for 'split'. Must be either 'iid' or 'noniid'.")

# Adjust this function to perform proper data augmentations.
def get_transforms(nx=32):

	def _convert_image_to_rgb(image):
		return image.convert("RGB")

	train_transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(nx, antialias=True, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
		torchvision.transforms.RandomHorizontalFlip(),
		torchvision.transforms.CenterCrop(nx),
		_convert_image_to_rgb,
		torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
	])

	test_transform = torchvision.transforms.Compose([
		torchvision.transforms.Resize(nx, antialias=True, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
		torchvision.transforms.CenterCrop(nx),
		_convert_image_to_rgb,
		torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010]),
	])

	return train_transform, test_transform

def get_data(attack_setting, run_id, name='cifar10', data_split='noniid', skewness=0.2, id=0,
	num_clients=10, batch_size=128, return_eval_ds=False, seed=42):

	train_transform, test_transform = get_transforms(nx=32)
	mask = get_mask(run_id)

	with contextlib.redirect_stdout(None):
		ds_train, ds_test, num_classes, _ = DATALOADER[name](train_transform, test_transform)

	if return_eval_ds:
		dataset = get_dataset_poisoned('server', run_id, False)
		if attack_setting.improved:
			ds_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
			return (ds_test, num_classes, len(ds_test.dataset))
		else:
			if dataset is None:
				modified_samples = []
				input_range = None
				for idx in range(len(ds_test)):
					original_data, original_label = ds_test[idx]
					poisoned_data = attack_setting.embed_trigger_in_image([original_data], [original_label], attack_setting, input_range, mask=mask)
					modified_samples.append((poisoned_data[0][0], original_label))
				dataset = PoisonedDataset(modified_samples)
				save_object(dataset, 'server', run_id, False)
			ds_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
			return (ds_test, num_classes, len(ds_test.dataset))

	ds_train = split_dataset(ds=ds_train, num_clients=num_clients, split=data_split,
		client_ids=np.arange(num_clients), num_classes=num_classes,
		chosen_ids=attack_setting.attackers, chosen_label=attack_setting.source_class, skewness=skewness, seed=seed)[id]

	if attack_setting.improved:
		poisoned_dataset = get_dataset_poisoned(id, run_id, True)
		if poisoned_dataset is None and mask is not None and id in attack_setting.attackers:
				modified_samples = []
				input_range = None
				if (attack_setting.distributed and id in attack_setting.attackers): input_range = attack_setting.get_range(id)
				# poison dataset
				for idx in range(len(ds_train)):
					original_data, original_label = ds_train[idx]
					poisoned_data = attack_setting.embed_trigger_in_image([original_data], [original_label], attack_setting, input_range, mask=mask)
					modified_samples.append((poisoned_data[0][0], original_label))
				poisoned_dataset = PoisonedDataset(modified_samples)
				save_object(poisoned_dataset, id, run_id, True)

				modified_samples = []
				input_range = None
				# poison server dataset after activation maps have been created
				for idx in range(len(ds_test)):
					original_data, original_label = ds_test[idx]
					poisoned_data = attack_setting.embed_trigger_in_image([original_data], [original_label], attack_setting, input_range, mask=mask)
					modified_samples.append((poisoned_data[0][0], original_label))
				dataset = PoisonedDataset(modified_samples)
				save_object(dataset, 'server', run_id, False)

		if id in attack_setting.attackers and mask is not None:
			# TODO remove targeted attack code (not used anymore)
			if attack_setting.targeted:
				input_range = None
				if (attack_setting.distributed): input_range = attack_setting.get_range(id)
				filtered_data = attack_setting.embed_trigger_in_image([x for x, y in ds_train if y == attack_setting.source_class], [y for x, y in ds_train if y == attack_setting.source_class], attack_setting, input_range, mixup=False)
				ds_train_poison = PoisonedDataset(filtered_data)
				concat = torch.utils.data.ConcatDataset([ds_train, ds_train_poison])
				ds_train = torch.utils.data.DataLoader(concat, batch_size=batch_size, shuffle=True)
			else:
				input_range = None
				if (attack_setting.distributed): input_range = attack_setting.get_range(id)
				# Determine number of samples to replace (50% of ds_train)
				num_samples_to_replace = len(ds_train) // 2

				non_target_indices = [idx for idx, (_, label) in enumerate(ds_train) if label != attack_setting.target_class]
				random.seed(seed)
				indices_to_replace = random.sample(non_target_indices, num_samples_to_replace)
				modified_samples = []

				for idx in range(len(ds_train)):
					if idx in indices_to_replace:
						poisoned_data, poisoned_label = poisoned_dataset[idx]
						modified_samples.append((poisoned_data, attack_setting.target_class))
					else:
						modified_samples.append(ds_train[idx])

				ds_train = PoisonedDataset(modified_samples)
				ds_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)

		else:
			ds_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)

	else:
		poisoned_dataset = get_dataset_poisoned(id, run_id, True)
		if poisoned_dataset is None:
				modified_samples = []
				input_range = None
				if (attack_setting.distributed and id in attack_setting.attackers): input_range = attack_setting.get_range(id)
				for idx in range(len(ds_train)):
					original_data, original_label = ds_train[idx]
					poisoned_data = attack_setting.embed_trigger_in_image([original_data], [original_label], attack_setting, input_range, mask=mask)
					modified_samples.append((poisoned_data[0][0], original_label))
				poisoned_dataset = PoisonedDataset(modified_samples)
				save_object(poisoned_dataset, id, run_id, True)

		if id in attack_setting.attackers:
			# TODO remove targeted attack code (not used anymore)
			if attack_setting.targeted:
				input_range = None
				if (attack_setting.distributed): input_range = attack_setting.get_range(id)
				filtered_data = attack_setting.embed_trigger_in_image([x for x, y in ds_train if y == attack_setting.source_class], [y for x, y in ds_train if y == attack_setting.source_class], attack_setting, input_range, mixup=False)
				ds_train_poison = PoisonedDataset(filtered_data)
				concat = torch.utils.data.ConcatDataset([ds_train, ds_train_poison])
				ds_train = torch.utils.data.DataLoader(concat, batch_size=batch_size, shuffle=True)
			else:
				input_range = None
				if (attack_setting.distributed): input_range = attack_setting.get_range(id)
				# Determine number of samples to replace (50% of ds_train)
				num_samples_to_replace = len(ds_train) // 2

				non_target_indices = [idx for idx, (_, label) in enumerate(ds_train) if label != attack_setting.target_class]
				random.seed(seed)
				indices_to_replace = random.sample(non_target_indices, num_samples_to_replace)
				modified_samples = []
				for idx in range(len(ds_train)):
					if idx in indices_to_replace:
						poisoned_data, poisoned_label = poisoned_dataset[idx]
						modified_samples.append((poisoned_data, attack_setting.target_class))
					else:
						modified_samples.append(ds_train[idx])

				ds_train = PoisonedDataset(modified_samples)
				ds_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)

		else:
			ds_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True)


	return (ds_train, num_classes, len(ds_train.dataset))
