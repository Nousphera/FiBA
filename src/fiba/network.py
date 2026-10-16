"""The original CIFAR ResNet18, trained from scratch with option-A shortcuts."""
import torch
from torch import nn
from torch.nn import functional as F


class OptionAShortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride = stride
        self.padding = (out_channels - in_channels) // 2

    def forward(self, inputs):
        return F.pad(inputs[:, :, ::self.stride, ::self.stride],
                     (0, 0, 0, 0, self.padding, self.padding))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.shortcut = (nn.Identity() if stride == 1 and in_channels == channels
                         else OptionAShortcut(in_channels, channels, stride))

    def forward(self, inputs):
        outputs = F.relu(self.bn1(self.conv1(inputs)))
        outputs = self.bn2(self.conv2(outputs))
        return F.relu(outputs + self.shortcut(inputs))


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, 2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, 2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, 2), BasicBlock(512, 512))
        self.linear = nn.Linear(512, num_classes)

    def forward(self, inputs):
        outputs = F.relu(self.bn1(self.conv1(inputs)))
        outputs = self.layer4(self.layer3(self.layer2(self.layer1(outputs))))
        return self.linear(F.avg_pool2d(outputs, 4).flatten(1))


def build_model(num_classes: int, seed: int = 42) -> ResNet:
    if num_classes not in (10, 100):
        raise ValueError('FiBA supports CIFAR10 and CIFAR100 classifiers')
    # Model construction must not advance augmentation/client selection RNGs.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ResNet(num_classes)
