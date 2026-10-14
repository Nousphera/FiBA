import os
import torch
import torchvision
import numpy as np

from pathlib import Path
from ..prepared_arrays import image_dataset


def prepared_cifar(name, root, train=True, download=False, transform=None):
    path = Path(os.environ['TORCH_DATA_DIR']) / (name + '.npz')
    if path.exists():
        return image_dataset(path, name, train, transform)
    loader = torchvision.datasets.CIFAR10 if name == 'cifar10' else torchvision.datasets.CIFAR100
    return loader(root, train=train, download=download, transform=transform)



def get_cifar10(train_transform=None, test_transform=None):
	data_train = prepared_cifar("cifar10", os.path.join(os.environ['TORCH_DATA_DIR'],'cifar10'), train=True, download=os.environ.get("FIBA_DOWNLOAD") == "1", transform=train_transform)
	data_test = prepared_cifar("cifar10", os.path.join(os.environ['TORCH_DATA_DIR'],'cifar10'), train=False, download=False, transform=test_transform)
	return data_train, data_test, len(data_train.classes), 'CIFAR-10'

def get_cifar100(train_transform=None, test_transform=None):
	data_train = prepared_cifar("cifar100", os.path.join(os.environ['TORCH_DATA_DIR'],'cifar100'), train=True, download=os.environ.get("FIBA_DOWNLOAD") == "1", transform=train_transform)
	data_test = prepared_cifar("cifar100", os.path.join(os.environ['TORCH_DATA_DIR'],'cifar100'), train=False, download=False, transform=test_transform)
	return data_train, data_test, len(data_train.classes), 'CIFAR-100'

def get_svhn(train_transform=None, test_transform=None):

	class SVHN(torchvision.datasets.SVHN):
		def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
			super(SVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
			self.targets = self.labels

	data_train = SVHN(os.path.join(os.environ['TORCH_DATA_DIR'],'svhn'), split='train', download=os.environ.get("FIBA_DOWNLOAD") == "1", transform=train_transform)
	data_test = SVHN(os.path.join(os.environ['TORCH_DATA_DIR'],'svhn'), split='test', download=os.environ.get("FIBA_DOWNLOAD") == "1", transform=test_transform)
	return data_train, data_test, len(np.unique(data_train.labels)), 'SVHN'

def get_emnist(train_transform=None, test_transform=None):
	data_train = torchvision.datasets.EMNIST(os.path.join(os.environ['TORCH_DATA_DIR'],'emnist'), split='letters', train=True, download=os.environ.get("FIBA_DOWNLOAD") == "1", transform=train_transform)
	data_test = torchvision.datasets.EMNIST(os.path.join(os.environ['TORCH_DATA_DIR'],'emnist'), split='letters', train=False, download=os.environ.get("FIBA_DOWNLOAD") == "1", transform=test_transform)
	return data_train, data_test, len(data_train.classes), 'EMNIST'

def get_eurosat(train_transform=None, test_transform=None, seed=42):

	class EuroSAT(torchvision.datasets.EuroSAT):

		def download(self) -> None:
			if self._check_exists(): return
			os.makedirs(self._base_folder, exist_ok=True)
			torchvision.datasets.utils.download_and_extract_archive("http://madm.dfki.de/files/sentinel/EuroSAT.zip",\
				download_root=self._base_folder, md5="c8fa014336c82ac7804f0398fcb19387",)

	class TransformDataset(torch.utils.data.Dataset):

		def __init__(self, base_dataset, transform):
			self.base_dataset = base_dataset
			self.transform = transform
			self.targets = [self.base_dataset[i][1] for i in range(len(self.base_dataset))]

		def __getitem__(self, index):
			img, target = self.base_dataset[index]
			return self.transform(img), target

		def __len__(self):
			return len(self.base_dataset)

	data = EuroSAT(os.environ['TORCH_DATA_DIR'], transform=None, download=os.environ.get("FIBA_DOWNLOAD") == "1")
	gen = torch.Generator().manual_seed(seed)
	data_train, data_test = torch.utils.data.random_split(dataset=data, lengths=[0.8,0.2], generator=gen)
	data_train = TransformDataset(data_train, transform=train_transform)
	data_test = TransformDataset(data_test, transform=test_transform)
	return data_train, data_test, 10, 'EuroSAT'

def get_fashion_mnist(train_transform=None, test_transform=None):
	data_train = torchvision.datasets.FashionMNIST(os.path.join(os.environ['TORCH_DATA_DIR'],'fashionmnist'), train=True, download=os.environ.get("FIBA_DOWNLOAD") == "1", transform=train_transform)
	data_test = torchvision.datasets.FashionMNIST(os.path.join(os.environ['TORCH_DATA_DIR'],'fashionmnist'), train=False, download=os.environ.get("FIBA_DOWNLOAD") == "1", transform=test_transform)
	return data_train, data_test, len(data_train.classes), 'Fashion-MNIST'

def get_food101(train_transform=None, test_transform=None):

	class Food101(torchvision.datasets.Food101):
		def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
			super(Food101, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
			self.targets = self._labels

	data_train = Food101(os.path.join(os.environ['TORCH_DATA_DIR'],'food101'), split='train', download=os.environ.get("FIBA_DOWNLOAD") == "1", transform=train_transform)
	data_test = Food101(os.path.join(os.environ['TORCH_DATA_DIR'],'food101'), split='test', download=os.environ.get("FIBA_DOWNLOAD") == "1", transform=test_transform)
	return data_train, data_test, len(data_train.classes), 'FOOD-101'

DATALOADER = {
	'cifar10': get_cifar10,
	'cifar100': get_cifar100,
	'svhn': get_svhn,
	'emnist': get_emnist,
	'fashion_mnist': get_fashion_mnist,
	'eurosat': get_eurosat,
	'food101': get_food101,
}


class SyntheticImages(torch.utils.data.Dataset):
    """Deterministic 32x32 RGB fixture for pipeline checks, never a benchmark."""
    def __init__(self, size, transform, seed):
        self.images = np.random.default_rng(seed).integers(0,256,(size,32,32,3),dtype=np.uint8)
        self.targets = [i % 2 for i in range(size)]
        self.transform = transform
    def __len__(self):
        return len(self.targets)
    def __getitem__(self,index):
        from PIL import Image
        image = Image.fromarray(self.images[index])
        return self.transform(image),self.targets[index]


def get_synthetic(train_transform=None,test_transform=None):
    return SyntheticImages(16,train_transform,123),SyntheticImages(8,test_transform,456),2,'synthetic'

DATALOADER['synthetic'] = get_synthetic
