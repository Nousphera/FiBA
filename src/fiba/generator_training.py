"""Pretrain self-contained Encoder or LIRA artifacts on public CIFAR images.

LIRA follows the alternating previous-generator/candidate-generator algorithm
of Doan et al. (ICCV 2021). Federated classifier training is its frozen-generator
second stage. Generator architecture and pretraining hyperparameters below are
declared implementation choices; FiBA specifies only epsilon=0.1 for LIRA.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch.nn import functional as F
from .generators import Encoder, Decoder, LiraGenerator, save_generator, load_generator, shared_trigger


DEFAULTS = {
    'dataset': 'cifar10', 'data_root': 'datasets', 'seed': 42,
    'generator_epochs': 100, 'batch_size': 64, 'lr': .0005,
    'generator_lr': .0001, 'classifier_lr': .01, 'num_workers': 0,
    'device': 'auto', 'encoder_bits': 200, 'encoder_beta': .7,
    'attention_penalty': 10., 'epsilon': .1, 'lira_alpha': .5,
    'lira_classifier_epochs': 1, 'lira_reset_classifier': True,
    'target_class': 5,
    'attention_placement': 'low',
}


def _device(request):
    if request == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else
                            'mps' if torch.backends.mps.is_available() else 'cpu')
    device = torch.device(request)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA was requested but is unavailable')
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS was requested but is unavailable')
    return device


def train_generator(cfg):
    """Train on actual CIFAR training data and return a new checkpoint path.

    Required keys: attack ('encoder' or 'lira'), output. An optional
    attention_mask tensor/path marks high activation with ones for the
    encoder's weighted reconstruction objective. No test images are optimized.
    """
    from .data import load_datasets, normalize, augment_raw
    from .network import build_model
    config = {**DEFAULTS, **cfg}
    kind = config.get('attack')
    if kind not in {'encoder', 'lira'}:
        raise ValueError('Generator attack must be encoder or lira')
    if config['attention_placement'] not in {'low', 'high'}:
        raise ValueError('attention_placement must be low or high')
    if config.get('initial_checkpoint') and kind != 'encoder':
        raise ValueError('initial_checkpoint is supported for encoder fine-tuning only')
    if config['dataset'] not in {'cifar10', 'cifar100'}:
        raise ValueError('Generator training supports public CIFAR-10 and CIFAR-100 only')
    if not config.get('output'):
        raise ValueError('Generator output checkpoint path is required')
    output = Path(config['output'])
    if output.exists() or output.with_suffix('.json').exists():
        raise ValueError('Output checkpoint or report already exists; choose another path')
    for key in ['generator_epochs', 'batch_size', 'lira_classifier_epochs']:
        if int(config[key]) < 1:
            raise ValueError(f'{key} must be positive')
    if int(config['num_workers']) < 0:
        raise ValueError('num_workers must be non-negative')
    for key in ['lr', 'generator_lr', 'classifier_lr']:
        if not math.isfinite(float(config[key])) or float(config[key]) <= 0:
            raise ValueError(f'{key} must be finite and positive')
    for key in ['encoder_beta', 'attention_penalty']:
        if not math.isfinite(float(config[key])) or float(config[key]) < 0:
            raise ValueError(f'{key} must be finite and non-negative')
    if not 0 < float(config['epsilon']) <= 1 or not 0 <= float(config['lira_alpha']) <= 1:
        raise ValueError('epsilon must be in (0,1], lira_alpha in [0,1]')
    seed = int(config['seed'])
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = _device(config['device'])
    train, _ = load_datasets(config['dataset'], str(config['data_root']))
    classes = len(train.classes)
    if len(train) == 0 or not 0 <= int(config['target_class']) < classes:
        raise ValueError('Training data must be nonempty and target_class valid')
    loader = torch.utils.data.DataLoader(
        train, batch_size=int(config['batch_size']), shuffle=True,
        num_workers=int(config['num_workers']),
        generator=torch.Generator().manual_seed(seed))
    mask = config.pop('attention_mask', None)
    if isinstance(mask, (str, Path)):
        mask = torch.load(mask, map_location='cpu', weights_only=True)
    if mask is not None:
        if kind != 'encoder':
            raise ValueError('Attention-aware LIRA is outside the FiBA paper')
        mask = torch.as_tensor(mask, dtype=torch.float32, device=device)
        if mask.shape != (32, 32) or not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('attention_mask must be a binary [32,32] tensor')
    config['device'] = str(device)
    config['output'] = str(output)
    config['attention_weighted_training'] = mask is not None
    config['generator_architecture'] = 'fiba-convolutional-encoder-sigmoid' if kind == 'encoder' else 'unet-32-64-128-tanh'
    config['optimizer'] = 'AdamW(weight_decay=0.01)' if kind == 'encoder' else 'Adam generator; SGD(momentum=0.9) surrogate'
    config['schedule'] = 'StepLR(step_size=5,gamma=0.5); gradient_clip=1' if kind == 'encoder' else 'constant learning rates'
    config['trigger_space'] = 'raw RGB [0,1]'
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    history = []
    if kind == 'encoder':
        initial = None
        if config.get('initial_checkpoint'):
            encoder, initial = load_generator(config['initial_checkpoint'], 'encoder')
            encoder = encoder.to(device).requires_grad_(True)
            if 'decoder_state' not in initial:
                raise ValueError('Encoder fine-tuning checkpoint must contain decoder_state')
            config['encoder_bits'] = initial['config'].get('encoder_bits', 200)
            config['trigger_seed'] = initial['config'].get('trigger_seed', initial['config'].get('seed', 42))
            if mask is None and initial.get('attention_mask') is not None:
                inherited_placement = initial['config'].get('attention_placement', 'low')
                if 'attention_placement' in cfg and config['attention_placement'] != inherited_placement:
                    raise ValueError('Cannot change inherited attention placement without providing a new attention_mask')
                mask = initial['attention_mask'].to(device=device, dtype=torch.float32)
                config['attention_weighted_training'] = True
                config['attention_placement'] = inherited_placement
        else:
            encoder = Encoder().to(device)
            config['trigger_seed'] = seed
        decoder = Decoder().to(device)
        if initial is not None:
            decoder.load_state_dict(initial['decoder_state'], strict=True)
        optimizer = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters()], lr=float(config['lr']))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=.5)
        bits = int(config['encoder_bits'])
        trigger = shared_trigger(seed, bits) if initial is None else initial['trigger']
        positions = torch.randperm(1024, generator=torch.Generator().manual_seed(int(config['trigger_seed'])))[:bits]
        trigger_rng = torch.Generator().manual_seed(seed + 1)
        for epoch in range(int(config['generator_epochs'])):
            loss_sum, reconstruction_sum, bit_sum, seen = 0., 0., 0., 0
            encoder.train(); decoder.train()
            for images, _ in loader:
                images = augment_raw(images.to(device))
                pattern = torch.zeros(len(images), 1024)
                pattern[:, positions] = torch.randint(0, 2, (len(images), bits), generator=trigger_rng).float()
                pattern = pattern.reshape(-1, 1, 32, 32).to(device)
                encoded = encoder(images, pattern)
                error = (encoded - images).square()
                if mask is not None:
                    error = error * (1 + float(config['attention_penalty']) * mask[None, None])
                reconstruction = error.mean()
                bit_loss = F.mse_loss(decoder(encoded), pattern)
                loss = reconstruction + float(config['encoder_beta']) * bit_loss
                if not torch.isfinite(loss):
                    raise ValueError('Encoder training produced non-finite loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *decoder.parameters()], 1.)
                optimizer.step()
                batch = len(images)
                seen += batch
                loss_sum += loss.item() * batch
                reconstruction_sum += reconstruction.item() * batch
                bit_sum += bit_loss.item() * batch
            scheduler.step()
            history.append({'epoch': epoch + 1, 'loss': loss_sum / seen,
                            'reconstruction_loss': reconstruction_sum / seen, 'bit_loss': bit_sum / seen})
        extras = {'decoder_state': {k: v.detach().cpu() for k, v in decoder.state_dict().items()}}
        if mask is not None:
            extras['attention_mask'] = mask.detach().cpu()
        save_generator(output, kind, encoder, config, trigger, history=history,
                       training_examples=len(train), **extras)
    else:
        candidate = LiraGenerator().to(device)
        previous = deepcopy(candidate).eval().requires_grad_(False)
        optimizer_generator = torch.optim.Adam(candidate.parameters(), lr=float(config['generator_lr']))
        surrogate = build_model(classes, seed=seed).to(device)
        optimizer_classifier = torch.optim.SGD(surrogate.parameters(), lr=float(config['classifier_lr']), momentum=.9)
        epsilon, alpha = float(config['epsilon']), float(config['lira_alpha'])
        for epoch in range(int(config['generator_epochs'])):
            if epoch and config['lira_reset_classifier']:
                surrogate = build_model(classes, seed=seed + epoch).to(device)
                optimizer_classifier = torch.optim.SGD(surrogate.parameters(), lr=float(config['classifier_lr']), momentum=.9)
            generator_sum, classifier_sum, seen = 0., 0., 0
            surrogate.train(); candidate.train()
            for _ in range(int(config['lira_classifier_epochs'])):
                for images, labels in loader:
                    images, labels = augment_raw(images.to(device)), labels.to(device).long()
                    targets = torch.full_like(labels, int(config['target_class']))
                    # Optimize candidate trigger through a fixed surrogate.
                    surrogate.requires_grad_(False)
                    optimizer_generator.zero_grad(set_to_none=True)
                    generated = (images + epsilon * candidate(images)).clamp(0, 1)
                    generator_loss = F.cross_entropy(surrogate(normalize(generated)), targets)
                    if not torch.isfinite(generator_loss):
                        raise ValueError('LIRA generator produced non-finite loss')
                    generator_loss.backward()
                    optimizer_generator.step()
                    # Train surrogate using the preceding outer epoch's frozen
                    # generator, as in the original alternating LIRA algorithm.
                    surrogate.requires_grad_(True)
                    with torch.no_grad():
                        poisoned = (images + epsilon * previous(images)).clamp(0, 1)
                    optimizer_classifier.zero_grad(set_to_none=True)
                    classifier_loss = alpha * F.cross_entropy(surrogate(normalize(images)), labels)
                    classifier_loss += (1-alpha) * F.cross_entropy(surrogate(normalize(poisoned)), targets)
                    if not torch.isfinite(classifier_loss):
                        raise ValueError('LIRA surrogate produced non-finite loss')
                    classifier_loss.backward()
                    optimizer_classifier.step()
                    seen += len(images)
                    generator_sum += generator_loss.item() * len(images)
                    classifier_sum += classifier_loss.item() * len(images)
            previous.load_state_dict(candidate.state_dict())
            history.append({'epoch': epoch + 1, 'loss': generator_sum / seen,
                            'classifier_loss': classifier_sum / seen})
        save_generator(output, kind, candidate, config, history=history, training_examples=len(train))
    output.with_suffix('.json').write_text(json.dumps({
        'kind': kind, 'config': config, 'training_examples': len(train), 'history': history,
        'implementation': 'FiBA encoder or alternating previous-generator LIRA; raw pixel-space epsilon',
        'source': ('https://doi.org/10.1007/978-981-96-4836-8_13' if kind == 'encoder' else
                   'https://openaccess.thecvf.com/content/ICCV2021/html/Doan_LIRA_Learnable_Imperceptible_and_Robust_Backdoor_Attacks_ICCV_2021_paper.html'),
    }, indent=2, allow_nan=False) + '\n')
    return output
