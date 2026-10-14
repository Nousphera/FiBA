# Modified for the FiBA standalone release, 2026-10-15; see PROVENANCE.md.
from PIL import Image
import numpy as np
from torch import tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import sys
import matplotlib.pyplot as plt
# import cv2

def visualize_image_difference(image1, image2):
    if image1.shape[:2] != image2.shape[:2]:
        raise ValueError("Images must have the same dimensions.")

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

    plt.savefig('result')


def hide_trigger_in_image(cover_image_array):
    width, height, channels = cover_image_array.shape

    image = cover_image_array

    # modify 4 pixels in the bottom right corner
    for c in range(channels):
        image[width-3, height-3, c] = 1.0
        image[width-4, height-2, c] = 1.0
        image[width-2, height-4, c] = 1.0
        image[width-2, height-2, c] = 1.0

    return image

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


class BadNetAttack():
    def __init__(self, attacker_ids, ranges, source_class, target_class, distributed, boost_strat, improved, mask_inversion):
        self.name = 'BadNet Attack'
        self.attackers = attacker_ids
        self.ranges = ranges
        self.source_class = source_class
        self.target_class = target_class
        self.distributed = distributed
        self.boost_strat = boost_strat
        self.targeted = False
        self.improved = improved

    def get_range(self, attacker_id):
        segment_length = len(self.attackers) // len(self.ranges)
        second_index = self.attackers.index(attacker_id) // segment_length
        second_index = min(second_index, len(self.ranges) - 1)
        return self.ranges[second_index]


    def embed_trigger_in_image(self, images, labels, attack_setting, input_range=None, mask=None):
        images = [denormalize_image(image) for image in images]
        numpy_image_array = [image.cpu().numpy().transpose(1, 2, 0) for image in images]

        poisoned_images = [hide_trigger_in_image(poisoned_image.copy()) for poisoned_image in numpy_image_array]

        poisoned_images = [poisoned_image.transpose(2, 0, 1) for poisoned_image in poisoned_images]
        poisoned_images = [torch.from_numpy(poisoned_image) for poisoned_image in poisoned_images]

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean=mean, std=std)
        poisoned_images = [normalize(image) for image in poisoned_images]

        # Return a list of tuples containing the poisoned images and target classes
        return list(zip(poisoned_images, [attack_setting.target_class] * len(poisoned_images)))

class PoisonedDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        x, y = self.subset[index]
        return x, y

    def __len__(self):
        return len(self.subset)
