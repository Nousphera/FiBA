"""Trainable image generators and self-contained, weights-only checkpoints.

The encoder retains FiBA's three-layer preprocessing and five-layer embedding
architecture. The bounded output head, saved shared bit pattern, and LIRA U-Net
are explicit implementation choices where the paper does not specify details.
"""
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def _convolutions(channels, *, final_activation=True):
    layers = []
    for index, (left, right) in enumerate(zip(channels, channels[1:])):
        layers.append(nn.Conv2d(left, right, 3, padding=1))
        if index < len(channels) - 2 or final_activation:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Encode a spatial bit pattern into a bounded RGB image."""
    def __init__(self):
        super().__init__()
        self.image_preprocess = _convolutions([3, 8, 16, 32])
        self.trigger_preprocess = _convolutions([1, 8, 16, 32])
        self.embedding = _convolutions([64, 64, 32, 16, 8, 3], final_activation=False)

    def forward(self, images, trigger):
        return torch.sigmoid(self.embedding(torch.cat(
            [self.image_preprocess(images), self.trigger_preprocess(trigger)], dim=1)))


class Decoder(nn.Module):
    """Recover the spatial bit pattern for encoder pretraining only."""
    def __init__(self):
        super().__init__()
        self.layers = _convolutions([3, 8, 16, 32, 16, 8, 1], final_activation=False)

    def forward(self, images):
        return self.layers(images)


class LiraGenerator(nn.Module):
    """Small U-Net with tanh output; epsilon scaling happens in pixel space."""
    def __init__(self):
        super().__init__()
        self.down1 = _convolutions([3, 32, 32])
        self.down2 = _convolutions([32, 64, 64])
        self.middle = _convolutions([64, 128, 128])
        self.up2 = _convolutions([128 + 64, 64, 64])
        self.up1 = _convolutions([64 + 32, 32, 32])
        self.output = nn.Conv2d(32, 3, 1)

    def forward(self, images):
        first = self.down1(images)
        second = self.down2(F.avg_pool2d(first, 2))
        middle = self.middle(F.avg_pool2d(second, 2))
        second_up = self.up2(torch.cat([
            F.interpolate(middle, size=second.shape[-2:], mode='bilinear', align_corners=False), second], 1))
        first_up = self.up1(torch.cat([
            F.interpolate(second_up, size=first.shape[-2:], mode='bilinear', align_corners=False), first], 1))
        return torch.tanh(self.output(first_up))


def shared_trigger(seed=42, bits=200):
    if not 1 <= bits <= 1024:
        raise ValueError('Encoder trigger bits must be in [1,1024]')
    rng = torch.Generator().manual_seed(int(seed))
    trigger = torch.zeros(1024)
    positions = torch.randperm(1024, generator=rng)[:bits]
    trigger[positions] = torch.randint(0, 2, (bits,), generator=rng).float()
    return trigger.reshape(1, 1, 32, 32)


def save_generator(path, kind, model, config, trigger=None, **extra):
    path = Path(path)
    if kind not in {'encoder', 'lira'}:
        raise ValueError('Unknown generator kind')
    path.parent.mkdir(parents=True, exist_ok=True)
    # Configuration must consist of weights-only-compatible primitive values.
    checkpoint = {'format_version': 1, 'kind': kind,
                  'model_state': {name: value.detach().cpu() for name, value in model.state_dict().items()},
                  'config': {key: str(value) if isinstance(value, Path) else value for key, value in config.items()},
                  **extra}
    if trigger is not None:
        checkpoint['trigger'] = trigger.detach().cpu()
    torch.save(checkpoint, path)
    return path


def load_generator(path, kind):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=True)
    if checkpoint.get('format_version') != 1 or checkpoint.get('kind') != kind:
        raise ValueError(f'Expected FiBA version-1 {kind} checkpoint')
    # fork_rng prevents checkpoint loading from consuming the training RNG.
    with torch.random.fork_rng(devices=[]):
        model = Encoder() if kind == 'encoder' else LiraGenerator()
    model.load_state_dict(checkpoint['model_state'], strict=True)
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Generator checkpoint contains non-finite weights')
    if kind == 'encoder':
        trigger = checkpoint.get('trigger')
        if not isinstance(trigger, torch.Tensor) or trigger.shape != (1, 1, 32, 32):
            raise ValueError('Encoder checkpoint must contain its shared [1,1,32,32] trigger')
        if not torch.all((trigger == 0) | (trigger == 1)):
            raise ValueError('Encoder shared trigger must contain bits')
        mask = checkpoint.get('attention_mask')
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.shape != (32, 32)
                                 or not torch.all((mask == 0) | (mask == 1))):
            raise ValueError('Encoder attention mask must be binary with shape [32,32]')
    model.eval().requires_grad_(False)
    return model, checkpoint
