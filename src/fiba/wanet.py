# Modified for the FiBA standalone release, 2026-10-15; see PROVENANCE.md.
'''
Adaptation of Wanet Attack: Anh Nguyen and Anh Tran. Wanet – imperceptible warping-based backdoor attack, 2021

Github: https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release
'''

from PIL import Image
import numpy as np
from torch import tensor
from torch.utils.data import Dataset
import torch
import sys
import matplotlib.pyplot as plt
import torch.nn.functional as F
# import cv2

def visualize_image_difference(image1, image2):
   # Check if the images have the same dimensions
    if image1.shape[:2] != image2.shape[:2]:
        raise ValueError("Images must have the same dimensions.")

    image1 = image1.multiply(255.0).type(torch.uint8).cpu().detach().numpy().transpose(1, 2, 0)
    image2 = image2.multiply(255.0).type(torch.uint8).cpu().detach().numpy().transpose(1, 2, 0)

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
    plt.imshow(diff, cmap='coolwarm')  # You can choose a colormap that emphasizes small differences
    plt.axis('off')

    plt.show()

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

class WaNetAttack():
    def __init__(self, attacker_ids, ranges, source_class, target_class, distributed, boost_strat, attackers, dimensions):
        self.name = 'WaNet Attack'
        self.attackers = attacker_ids
        self.ranges = ranges
        self.source_class = source_class
        self.target_class = target_class
        self.distributed = distributed
        self.boost_strat = boost_strat
        self.targeted = False

        self.improved = False

        self.k = 6
        self.s = 0.75
        self.rate_bd = 0.5
        self.noise_ratio = 0.2
        self.grid_rescale = 1

        ins = torch.rand(1, 2, self.k, self.k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))
        self.noise_grid = (
            F.interpolate(ins, size=32, mode="bicubic", align_corners=True)
            .permute(0, 2, 3, 1)
        )
        array1d = torch.linspace(-1, 1, steps=32)
        x, y = torch.meshgrid(array1d, array1d, indexing="ij")
        self.identity_grid = torch.stack((y, x), 2)[None, ...]

    def get_range(self, attacker_id):
        segment_length = len(self.attackers) // len(self.ranges)
        second_index = self.attackers.index(attacker_id) // segment_length
        second_index = min(second_index, len(self.ranges) - 1)
        return self.ranges[second_index]

    def warp_images(self, inputs, labels, input_range=None, mixup=False, attacker_id=0):
        bs = len(inputs)
        inputs = torch.stack(inputs)
        labels = torch.tensor(labels)

        channels, height, width = inputs[0].shape

        num_backdoor = bs
        grid_temps = (self.identity_grid + self.s * self.noise_grid / height) * self.grid_rescale
        grid_temps = torch.clamp(grid_temps, -1, 1)

        inputs_bd = F.grid_sample(inputs[:num_backdoor], grid_temps.repeat(num_backdoor, 1, 1, 1), align_corners=True)
        targets_bd = torch.ones_like(labels[:num_backdoor]) * self.target_class

        if (input_range):
            inputs[:, :, input_range[1]:input_range[3], input_range[0]:input_range[2]] = inputs_bd[:, :, input_range[1]:input_range[3], input_range[0]:input_range[2]]
            return inputs, [label.item() for label in targets_bd]
        else:
            return inputs_bd, [label.item() for label in targets_bd]

    def embed_trigger_in_image(self, images, labels, attack_setting, input_range=None, mask=None):
        numpy_image_array = [image for image in images]

        poisoned_images = [image for image in numpy_image_array]
        poisoned_images, altered_labels = self.warp_images(poisoned_images, labels, input_range=input_range, attacker_id=0)

        return list(zip(poisoned_images, altered_labels))

class PoisonedDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        x, y = self.subset[index]
        return x, y

    def __len__(self):
        return len(self.subset)
