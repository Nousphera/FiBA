# Modified for FiBA, 2026-10-15. Component terms: licenses/THIRD_PARTY.txt.
from PIL import Image
import numpy as np
from torch import tensor
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
import sys
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# import cv2
import random

class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, size, noise_dim=100, image_size=32):
        self.size = size
        self.noise_dim = noise_dim
        self.image_size = image_size
        self.target_size = image_size * image_size
        self.data = torch.from_numpy(np.random.binomial(1, 0.5, (size, noise_dim))).float()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        bitstring = self.data[idx]
        if self.noise_dim < self.target_size:
            image = torch.zeros(self.target_size)
            indices = random.sample(range(self.target_size), self.noise_dim)
            image[indices] = bitstring
        else:
            image = bitstring[:self.target_size]
        image = image.view(1, self.image_size, self.image_size)
        return image, torch.Tensor([])

class PreprocessingModuleImage(torch.nn.Module):

    def __init__(self):
        super(PreprocessingModuleImage, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        return x

class PreprocessingModuleGray(torch.nn.Module):

    def __init__(self):
        super(PreprocessingModuleGray, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        return x

class EmbeddingNetwork(torch.nn.Module):

    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        self.conv1 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv5 = torch.nn.Conv2d(8, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.relu(self.conv4(x))
        x = torch.nn.functional.relu(self.conv5(x))
        return x

class Encoder(torch.nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.preprocess1 = PreprocessingModuleImage()
        self.preprocess2 =  PreprocessingModuleGray()
        self.embedding = EmbeddingNetwork()

    def forward(self, img1, img2):
        x1 = self.preprocess1(img1)
        x2 = self.preprocess2(img2)
        x = torch.cat((x1, x2), dim=1)
        x = self.embedding(x)
        return x

def denormalize_image(image):
    """
    Denormalize an image tensor.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be denormalized.
        mean (list): Mean used for normalization.
        std (list): Std used for normalization.

    Returns:
        Tensor: Denormalized image.
    """
    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std = torch.tensor([0.2023, 0.1994, 0.2010])
    if image.ndimension() == 3:
        return image * std[:, None, None] + mean[:, None, None]
    elif image.ndimension() == 4:
        return image * std[None, :, None, None] + mean[None, :, None, None]
    else:
        raise ValueError('Expected tensor to be 3D or 4D (batch of images).')

def visualize_image_difference(image1, image2):
   # Check if the images have the same dimensions
    if image1.shape[:2] != image2.shape[:2]:
        raise ValueError("Images must have the same dimensions.")

    # Calculate the absolute difference between the two images
    diff = cv2.absdiff(image1, image2)
    diff = diff * 100

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Image 1")
    plt.imshow(image1)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Image 2")
    plt.imshow(image2)
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Difference Heatmap")
    plt.imshow(diff, cmap='coolwarm')
    plt.axis('off')

    plt.show()


# Function to hide a string trigger within an image
def hide_trigger_in_image(atkmodel, cover_image_array, noise, input_range=None):
    cover_image_array = torch.unsqueeze(cover_image_array, 0)
    noise = torch.unsqueeze(noise, 0)
    poisoned_image = atkmodel(cover_image_array, noise)
    poisoned_image = poisoned_image.squeeze(0).detach()
    return poisoned_image


class EncoderAttack():
    def __init__(self, attacker_ids, ranges, source_class, target_class, distributed, boost_strat, improved, mask_inversion, checkpoint=None):
        self.name = 'Encoder Attack'
        self.attackers = attacker_ids
        self.ranges = ranges
        self.source_class = source_class
        self.target_class = target_class
        self.distributed = distributed
        self.boost_strat = boost_strat
        self.atkmodel = Encoder()
        if checkpoint is None:
            raise ValueError('Encoder requires --encoder-checkpoint; see docs/EXPERIMENTS.md')
        self.atkmodel.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        self.atkmodel.eval()
        self.targeted = False
        self.improved = improved
        self.mask_inversion = mask_inversion
        self.noise_ds = NoiseDataset(size=1, noise_dim=200, image_size=32)
        self.hidden_image = next(iter(self.noise_ds))[0]


    def get_range(self, attacker_id):
        segment_length = len(self.attackers) // len(self.ranges)
        second_index = self.attackers.index(attacker_id) // segment_length
        second_index = min(second_index, len(self.ranges) - 1)
        return self.ranges[second_index]

    def embed_trigger_in_image(self, images, labels, attack_setting, input_range=None, mask=None):
        numpy_image_array = [denormalize_image(image) for image in images]

        # Make a copy of each image in the batch
        poisoned_images = [denormalize_image(image.clone()) for image in images]
        poisoned_images = [hide_trigger_in_image(self.atkmodel, poisoned_image, self.hidden_image, input_range) for poisoned_image in poisoned_images]

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean=mean, std=std)
        poisoned_images = [normalize(image) for image in poisoned_images]
        numpy_image_array = [normalize(image)for image in numpy_image_array]

        return list(zip(poisoned_images, [attack_setting.target_class] * len(poisoned_images)))

class PoisonedDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        x, y = self.subset[index]
        return x, y

    def __len__(self):
        return len(self.subset)
