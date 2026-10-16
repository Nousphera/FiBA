"""CIFAR benchmark triggers, operating exclusively on raw RGB tensors in [0,1]."""
import math
import torch
from torch.nn import functional as F
from .generators import load_generator


class Trigger:
    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.kind = cfg.get('attack', 'lsb')
        if self.kind not in {'none', 'lsb', 'badnet', 'wanet', 'encoder', 'lira'}:
            raise ValueError(f'Unknown attack: {self.kind}')
        self.seed = int(cfg.get('seed', 42))
        self.model = None
        self.attention_mask = None
        if self.kind in {'encoder', 'lira'}:
            path = cfg.get('generator_checkpoint')
            if not path:
                raise ValueError(f'{self.kind} requires generator_checkpoint; train it with python scripts/prepare_model.py --attack {self.kind}')
            self.model, self.checkpoint = load_generator(path, self.kind)
            if self.kind == 'encoder':
                self.attention_mask = self.checkpoint.get('attention_mask')
                placement = cfg.get('attention', 'none')
                if placement != 'none':
                    if self.attention_mask is None:
                        raise ValueError('Encoder attention requires a checkpoint trained with an attention mask')
                    if self.checkpoint['config'].get('attention_placement', 'low') != placement:
                        raise ValueError('Encoder checkpoint attention placement differs from the experiment')
            if self.kind == 'lira':
                trained_target = self.checkpoint.get('config', {}).get('target_class')
                if trained_target is not None and int(trained_target) != int(cfg.get('target_class', 5)):
                    raise ValueError('LIRA checkpoint target class differs from the experiment target')
        if self.kind == 'wanet':
            k = int(cfg.get('wanet_k', 6))
            if k < 2:
                raise ValueError('WaNet control grid size must be at least 2')
            rng = torch.Generator().manual_seed(self.seed)
            noise = 2 * torch.rand(1, 2, k, k, generator=rng) - 1
            noise /= noise.abs().mean().clamp_min(1e-12)
            self.noise_grid = F.interpolate(noise, size=(32, 32), mode='bicubic', align_corners=True).permute(0, 2, 3, 1)
            coordinates = torch.linspace(-1, 1, 32)
            y, x = torch.meshgrid(coordinates, coordinates, indexing='ij')
            self.identity_grid = torch.stack((x, y), -1).unsqueeze(0)
        self.attackers = math.floor(int(cfg.get('num_clients', 10)) * float(cfg.get('attacker_fraction', .1)))
        if cfg.get('distributed') and self.attackers < 1:
            raise ValueError('Distributed triggering requires at least one attacker')

    def apply(self, images, client_id=None, mask=None):
        """Apply the global trigger or one attacker's vertical image rectangle.

        A mask marks protected pixels with True. LSB skips them; encoder uses
        its saved attention-trained generator and verifies the identical mask.
        Evaluation with client_id=None always applies the complete trigger.
        """
        if (not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1:] != (3, 32, 32)
                or not images.is_floating_point()):
            raise ValueError('Triggers require float NCHW RGB CIFAR tensors [N,3,32,32]')
        if not torch.isfinite(images).all() or torch.any(images < 0) or torch.any(images > 1):
            raise ValueError('Trigger inputs must be finite raw RGB values in [0,1]')
        high = None
        if mask is not None:
            if self.kind not in {'lsb', 'encoder'}:
                raise ValueError('Attention mask is supported only for LSB and encoder')
            high = torch.as_tensor(mask, device=images.device)
            if high.shape != (32, 32) or not torch.isfinite(high).all():
                raise ValueError('Attention mask must be finite with shape [32,32]')
            high = high.bool()
        if self.kind == 'encoder':
            if high is not None and (self.attention_mask is None or
                    not torch.equal(high.cpu(), self.attention_mask.bool().cpu())):
                raise ValueError('Encoder runtime mask must equal its weighted-training checkpoint mask')
            if self.cfg.get('attention', 'none') != 'none' and high is None:
                raise ValueError('Encoder attention requires its checkpoint mask at inference')
        if len(images) == 0 or self.kind == 'none':
            return images.clone()
        if self.kind == 'lsb':
            result = self._lsb(images, high)
        elif self.kind == 'badnet':
            result = images.clone()
            # Five-pixel X in the lower-right corner (coordinates are an
            # explicit choice because the paper specifies only pixel count).
            for y, x in [(28, 28), (28, 30), (29, 29), (30, 28), (30, 30)]:
                result[:, :, y, x] = 1
        elif self.kind == 'wanet':
            strength = float(self.cfg.get('wanet_strength', .75))
            if not math.isfinite(strength) or strength < 0:
                raise ValueError('WaNet strength must be finite and non-negative')
            grid = (self.identity_grid + strength * self.noise_grid / 32).clamp(-1, 1)
            result = F.grid_sample(images, grid.to(images).expand(len(images), -1, -1, -1), align_corners=True)
        else:
            self.model.to(device=images.device, dtype=images.dtype)
            with torch.no_grad():
                if self.kind == 'encoder':
                    trigger = self.checkpoint['trigger'].to(images).expand(len(images), -1, -1, -1)
                    result = self.model(images, trigger)
                else:
                    epsilon = float(self.cfg.get('epsilon', .1))
                    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
                        raise ValueError('LIRA epsilon must be finite and in [0,1]')
                    result = (images + epsilon * self.model(images)).clamp(0, 1)
        if self.cfg.get('distributed') and client_id is not None:
            if not 0 <= int(client_id) < self.attackers:
                raise ValueError('Distributed client_id must identify an attacker')
            # Equal rectangles; final attacker receives any remainder.
            width = 32 // self.attackers
            if width < 1:
                raise ValueError('At most 32 distributed attacker rectangles are supported')
            left = int(client_id) * width
            right = 32 if int(client_id) == self.attackers - 1 else left + width
            part = images.clone()
            part[:, :, :, left:right] = result[:, :, :, left:right]
            result = part
        if not torch.isfinite(result).all():
            raise ValueError('Trigger produced non-finite output')
        return result.clamp(0, 1).to(images)

    def _lsb(self, images, high):
        characters = int(self.cfg.get('trigger_characters', 700))
        if characters <= 0:
            raise ValueError('LSB trigger_characters must be positive')
        text = ('Apple' * math.ceil(characters / 5))[:characters]
        bits = torch.tensor([int(bit) for value in text.encode('ascii') for bit in f'{value:08b}'],
                            dtype=torch.uint8, device=images.device)
        # HWC iteration order matches the released LSB implementation.
        raw = (images * 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        flat = raw.reshape(len(images), -1)
        selected = torch.ones((32, 32), dtype=torch.bool, device=images.device) if high is None else ~high
        positions = selected[:, :, None].expand(32, 32, 3).reshape(-1).nonzero().flatten()
        if len(bits) > len(positions) * 8:
            raise ValueError('LSB trigger exceeds selected pixels eight-bit capacity')
        for plane in range(8):
            start = plane * len(positions)
            if start >= len(bits):
                break
            part = bits[start:start + len(positions)]
            indices = positions[:len(part)]
            flat[:, indices] = (flat[:, indices] & (255 ^ (1 << plane))) | (part << plane)
        encoded = flat.reshape(-1, 32, 32, 3).permute(0, 3, 1, 2).to(images.dtype) / 255
        # Untouched pixels stay exact even for augmented non-quantized inputs.
        changed = (flat != (images * 255).round().to(torch.uint8).permute(0, 2, 3, 1).reshape(len(images), -1))
        changed = changed.reshape(-1, 32, 32, 3).permute(0, 3, 1, 2)
        return torch.where(changed, encoded, images)


def build_attack(cfg):
    return Trigger(cfg)
