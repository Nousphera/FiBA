# Modified for FiBA, 2026-10-15. Component terms: licenses/THIRD_PARTY.txt.
from PIL import Image
import numpy as np
from torch import tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
import sys
import matplotlib.pyplot as plt
import cv2
import os

def visualize_image_difference(image1, image2):

    if os.path.exists('result.png'):
        return
   # Check if the images have the same dimensions
    if image1.shape[:2] != image2.shape[:2]:
        raise ValueError("Images must have the same dimensions.")

    # Calculate the absolute difference between the two images
    diff = cv2.absdiff(image1, image2)
    diff = diff * 10

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

def repeat_string(input_string, repeat_count):
    if repeat_count <= 0:
        return ""
    else:
        return input_string * repeat_count

# Function to convert text trigger to binary
def text_to_binary(text):
    binary_trigger = ''.join(format(ord(char), '08b') for char in text)
    return binary_trigger

# Function to hide a string trigger within an image
def hide_trigger_in_image_modified(cover_image_array, text_trigger, mask=None, mask_value=0.0):
    width, height, channels = cover_image_array.shape

    binary_trigger = text_to_binary(text_trigger)
    trigger_length = len(binary_trigger)
    if mask is None or np.shape(mask) != cover_image_array.shape[:2]:
        raise ValueError("Mask must match image height and width")
    capacity = int(np.count_nonzero(np.asarray(mask) == mask_value)) * channels * 8
    if trigger_length > capacity:
        raise ValueError("Trigger exceeds selected pixel capacity")

    binary_trigger_idx = 0
    bit_offset = -1

    x1, y1, x2, y2 = 0, 0, width, height

    while binary_trigger_idx < trigger_length:
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    if binary_trigger_idx < trigger_length and mask[y][x] == mask_value:
                        pixel_value = cover_image_array[y][x][c]
                        pixel_binary = format(pixel_value, '08b')

                        # Replace the least significant bit with the trigger bit
                        pixel_binary = list(pixel_binary)
                        pixel_binary[bit_offset] = binary_trigger[binary_trigger_idx]
                        new_pixel_value = int(''.join(pixel_binary), 2)
                        binary_trigger_idx += 1

                        if (y in range(y1, y2) and x in range(x1, x2)):
                            cover_image_array[y][x][c] = new_pixel_value

        bit_offset -= 1

    hidden_image = cover_image_array
    return hidden_image

def hide_trigger_in_image(cover_image_array, text_trigger):
    width, height, channels = cover_image_array.shape

    binary_trigger = text_to_binary(text_trigger)
    trigger_length = len(binary_trigger)
    if trigger_length > cover_image_array.size * 8:
        raise ValueError("Trigger exceeds image capacity")

    binary_trigger_idx = 0
    bit_offset = -1

    x1, y1, x2, y2 = 0, 0, width, height

    while binary_trigger_idx < trigger_length:
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    if binary_trigger_idx < trigger_length:
                        pixel_value = cover_image_array[y][x][c]
                        pixel_binary = format(pixel_value, '08b')

                        # Replace the least significant bit with the trigger bit
                        pixel_binary = list(pixel_binary)
                        pixel_binary[bit_offset] = binary_trigger[binary_trigger_idx]
                        new_pixel_value = int(''.join(pixel_binary), 2)
                        binary_trigger_idx += 1

                        if (y in range(y1, y2) and x in range(x1, x2)):
                            cover_image_array[y][x][c] = new_pixel_value

        bit_offset -= 1

    hidden_image = cover_image_array
    return hidden_image


# Function to extract the binary trigger from an image
def extract_trigger_from_image(image, trigger_length, mask):
    # Open the hidden image
    width, height, channels = image.shape


    # Initialize variables to store the extracted binary trigger
    binary_trigger = ''
    binary_trigger_idx = 0
    bit_offset = -1

    while binary_trigger_idx < trigger_length:
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    if mask[y][x] == 0:
                        pixel_value = image[y][x][c]
                        pixel_binary = format(pixel_value, '08b')
                        # Extract the least significant bit and add it to the binary trigger
                        binary_trigger += pixel_binary[bit_offset]
                        binary_trigger_idx += 1
                        # Check if we have extracted the entire trigger
                        if binary_trigger_idx == trigger_length:
                            return binary_trigger
        bit_offset -= 1
    return binary_trigger

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


class SteganographyAttack():
    def __init__(self, attacker_ids, ranges, source_class, target_class, distributed, boost_strat, improved, mask_inversion):
        self.name = 'LSB Steganography Attack'
        self.attackers = attacker_ids
        self.ranges = ranges
        self.source_class = source_class
        self.target_class = target_class
        self.distributed = distributed
        self.boost_strat = boost_strat
        self.targeted = False
        self.improved = improved
        self.mask_inversion = mask_inversion

    def get_range(self, attacker_id):
        segment_length = len(self.attackers) // len(self.ranges)
        second_index = self.attackers.index(attacker_id) // segment_length
        second_index = min(second_index, len(self.ranges) - 1)
        return self.ranges[second_index]

    def embed_trigger_in_image(self, images, labels, attack_setting, input_range=None, mask=None):
        images = [denormalize_image(image) for image in images]

        numpy_image_array = [image.multiply(255.0).type(torch.uint8).cpu().numpy().transpose(1, 2, 0) for image in images]
        text_trigger = repeat_string("Apple", getattr(self, "trigger_characters", 500) // 5)

        if (mask is not None):
            mask = mask.type(torch.uint8).cpu().numpy()
            mask_value = 0.0
            if self.mask_inversion:
                mask_value = 1.0
            # Make a copy of each image in the batch
            poisoned_images = [image.copy() for image in numpy_image_array]
            poisoned_images = [hide_trigger_in_image_modified(poisoned_image, text_trigger, mask, mask_value=mask_value) for poisoned_image in poisoned_images]

        else:
            # Make a copy of each image in the batch
            poisoned_images = [image.copy() for image in numpy_image_array]
            poisoned_images = [hide_trigger_in_image(poisoned_image, text_trigger) for poisoned_image in poisoned_images]

        if (input_range):
            # Apply slicing to each numpy array in the list individually
            for i in range(len(poisoned_images)):
                new_image = numpy_image_array[i].copy()
                new_image[input_range[1]:input_range[3], input_range[0]:input_range[2]] = poisoned_images[i][input_range[1]:input_range[3], input_range[0]:input_range[2]]
                poisoned_images[i] = new_image

        poisoned_images = [poisoned_image.transpose(2, 0, 1) for poisoned_image in poisoned_images]
        poisoned_images = [torch.from_numpy(poisoned_image).div(255.0) for poisoned_image in poisoned_images]

        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        normalize = transforms.Normalize(mean=mean, std=std)
        poisoned_images = [normalize(image) for image in poisoned_images]

        # temp_poisoned = [denormalize_image(image.clone()) for image in poisoned_images]
        # visualize_image_difference(images[0].cpu().numpy().transpose(1, 2, 0), temp_poisoned[0].cpu().numpy().transpose(1, 2, 0))

        return list(zip(poisoned_images, [attack_setting.target_class] * len(poisoned_images)))

class PoisonedDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __getitem__(self, index):
        x, y = self.subset[index]
        return x, y

    def __len__(self):
        return len(self.subset)
