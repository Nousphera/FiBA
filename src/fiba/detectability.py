"""Post-training LPIPS and Neural Cleanse with auditable sampling and convergence.

Neural Cleanse follows the raw-domain mask/pattern objective and adaptive cost
of Wang et al. (2019). The exposed CIFAR optimization settings are independent
implementation choices, not a reproduction of unpublished FiBA detector settings.
Primary references: https://github.com/bolunwang/backdoor and
https://github.com/richzhang/PerceptualSimilarity .
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import load_datasets, normalize

NC_SOURCE = 'https://github.com/bolunwang/backdoor'
LPIPS_SOURCE = 'https://github.com/richzhang/PerceptualSimilarity'


def select_indices(labels, samples=256, seed=42, exclude_class=None, source_class=None):
    labels = np.asarray(labels)
    if labels.ndim != 1 or type(samples) is not int or samples < 1:
        raise ValueError('samples must be a positive integer and labels a vector')
    eligible = np.ones(len(labels), dtype=bool)
    if exclude_class is not None:
        eligible &= labels != exclude_class
    if source_class is not None and source_class >= 0:
        eligible &= labels == source_class
    indices = np.flatnonzero(eligible)
    if not len(indices):
        raise ValueError('No eligible non-target/source examples for detectability evaluation')
    return np.random.default_rng(seed).permutation(indices)[:samples]


def lpips_inputs(clean, triggered):
    if clean.shape != triggered.shape or clean.ndim != 4 or clean.shape[1:] != (3, 32, 32):
        raise ValueError('LPIPS expects equal NCHW RGB CIFAR batches')
    for batch in (clean, triggered):
        if not batch.is_floating_point() or not torch.isfinite(batch).all() or torch.any(batch < 0) or torch.any(batch > 1):
            raise ValueError('LPIPS inputs must be finite raw RGB in [0,1]')
    return 2 * clean - 1, 2 * triggered - 1


def make_lpips(device='cpu'):
    """Lazily instantiate the actual calibrated LPIPS metric with pretrained AlexNet.

    LPIPS calibration weights ship with the package. Torchvision downloads the
    pretrained AlexNet backbone into its normal torch.hub cache only if missing.
    """
    import lpips
    metric = lpips.LPIPS(net='alex', pretrained=True, pnet_rand=False, version='0.1', verbose=False)
    return metric.eval().to(device)


def evaluate_lpips(dataset, cfg, *, mask=None, samples=256, batch_size=64, device='cpu'):
    from .attacks import build_attack
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if cfg.get('attention', 'none') != 'none' and mask is None:
        raise ValueError('Attention-trigger LPIPS requires the saved protected-pixel attention_mask')
    indices = select_indices(dataset.targets, samples, cfg.get('seed', 42), cfg['target_class'], cfg.get('source_class', -1))
    trigger = build_attack(cfg)
    metric = make_lpips(device)
    values = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            clean = torch.stack([dataset[int(i)][0] for i in indices[start:start + batch_size]]).to(device)
            triggered = trigger.apply(clean, client_id=None, mask=mask)
            a, b = lpips_inputs(clean, triggered)
            distance = metric(a, b).reshape(-1)
            if distance.numel() != len(clean) or not torch.isfinite(distance).all():
                raise ValueError('LPIPS did not return one finite distance per image')
            values.extend(distance.detach().cpu().tolist())
    values = np.asarray(values, dtype=float)
    return {'metric': 'lpips', 'status': 'ok', 'backbone': 'alex', 'pretrained': True,
            'version': '0.1', 'source': LPIPS_SOURCE, 'input_range': 'raw RGB [0,1] mapped to [-1,1]',
            'samples_requested': samples, 'samples': len(indices), 'sample_indices': indices.tolist(),
            'sampling_seed': cfg.get('seed', 42), 'excluded_target_class': cfg['target_class'],
            'source_class': cfg.get('source_class', -1), 'trigger': cfg['attack'],
            'protected_attention_pixels': int(torch.as_tensor(mask).bool().sum()) if mask is not None else 0,
            'mean': float(values.mean()), 'std': float(values.std()), 'minimum': float(values.min()),
            'maximum': float(values.max()), 'per_sample': values.tolist()}


@dataclass(frozen=True)
class NCOptions:
    steps: int = 200
    lr: float = .1
    init_cost: float = .001
    success_threshold: float = .99
    patience: int = 5
    cost_multiplier: float = 2.
    batch_size: int = 64
    seed: int = 42

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (self.steps, self.patience, self.batch_size)):
            raise ValueError('steps, patience and batch_size must be positive integers')
        if (not all(math.isfinite(x) for x in (self.lr, self.init_cost, self.success_threshold, self.cost_multiplier))
                or self.lr <= 0 or self.init_cost <= 0 or not 0 < self.success_threshold <= 1 or self.cost_multiplier <= 1):
            raise ValueError('Invalid Neural Cleanse optimization settings')


def _success_rate(model, raw, target, mask, pattern, batch_size, normalizer):
    count = 0
    with torch.no_grad():
        for batch in raw.split(batch_size):
            crafted = (1 - mask) * batch + mask * pattern
            logits = model(normalizer(crafted))
            if not torch.isfinite(logits).all() or not 0 <= target < logits.shape[1]:
                raise ValueError('Invalid classifier outputs or target class')
            count += int((logits.argmax(1) == target).sum())
    return count / len(raw)


def reconstruct_trigger(model, images, target, options=None, *, normalizer=normalize):
    """Optimize one raw-domain sigmoid mask/pattern; keep the smallest success.

    One step visits every selected sample in shuffled minibatches, followed by
    exact ASR on that entire fixed selection. Failed last iterates are retained
    for inspection, but never enter the anomaly norm distribution.
    """
    options = options or NCOptions()
    if (images.ndim != 4 or images.shape[1:] != (3, 32, 32) or not len(images)
            or not images.is_floating_point() or not torch.isfinite(images).all()
            or torch.any(images < 0) or torch.any(images > 1)):
        raise ValueError('Neural Cleanse needs nonempty raw RGB CIFAR image tensors')
    if type(target) is not int or target < 0:
        raise ValueError('target must be a nonnegative integer')
    parameters = list(model.parameters())
    device = parameters[0].device if parameters else images.device
    raw = images.detach().to(device)
    rng = torch.Generator().manual_seed(options.seed + target)
    initial_mask = torch.rand((1, 32, 32), generator=rng).clamp(.01, .99)
    initial_pattern = torch.rand((3, 32, 32), generator=rng).clamp(.01, .99)
    mask_logits = nn.Parameter(torch.logit(initial_mask).to(device))
    pattern_logits = nn.Parameter(torch.logit(initial_pattern).to(device))
    optimizer = torch.optim.Adam([mask_logits, pattern_logits], lr=options.lr, betas=(.5, .9))
    training = model.training
    grad_flags = [p.requires_grad for p in parameters]
    model.eval()
    for p in parameters:
        p.requires_grad_(False)
    cost, up, down = 0., 0, 0
    best, best_norm, maximum_success = None, math.inf, 0.
    history = []
    try:
        for step in range(options.steps):
            order = torch.randperm(len(raw), generator=rng)
            ce_sum = 0.
            for indices in order.split(options.batch_size):
                batch = raw[indices.to(device)]
                mask, pattern = mask_logits.sigmoid(), pattern_logits.sigmoid()
                crafted = (1 - mask) * batch + mask * pattern
                logits = model(normalizer(crafted))
                if not 0 <= target < logits.shape[1]:
                    raise ValueError('target is outside the classifier class space')
                ce = F.cross_entropy(logits, torch.full((len(batch),), target, device=device, dtype=torch.long))
                loss = ce + cost * mask.sum()
                if not torch.isfinite(loss):
                    raise ValueError('Neural Cleanse optimization produced a nonfinite loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                ce_sum += float(ce.detach()) * len(batch)
            mask, pattern = mask_logits.sigmoid().detach(), pattern_logits.sigmoid().detach()
            norm = float(mask.sum())
            success = _success_rate(model, raw, target, mask, pattern, options.batch_size, normalizer)
            maximum_success = max(maximum_success, success)
            if success >= options.success_threshold and norm < best_norm:
                best_norm = norm
                best = (mask.cpu().clone(), pattern.cpu().clone(), success)
            history.append({'step': step + 1, 'cost': cost, 'cross_entropy': ce_sum / len(raw),
                            'mask_l1': norm, 'success_rate': success})
            if success >= options.success_threshold:
                up, down = up + 1, 0
                if up >= options.patience:
                    cost = options.init_cost if cost == 0 else min(cost * options.cost_multiplier, 1e12)
                    up = 0
            else:
                down, up = down + 1, 0
                if down >= options.patience:
                    cost = max(cost / options.cost_multiplier ** 1.5, 1e-12) if cost else 0.
                    down = 0
        if best is None:
            selected_mask, selected_pattern, selected_success = mask.cpu(), pattern.cpu(), success
        else:
            selected_mask, selected_pattern, selected_success = best
        return {'target_class': target, 'status': 'success' if best is not None else 'no_successful_reconstruction',
                'mask': selected_mask, 'pattern': selected_pattern, 'mask_l1': float(selected_mask.sum()),
                'best_successful_l1': best_norm if best is not None else None,
                'success_rate': selected_success, 'maximum_success_rate': maximum_success,
                'samples': len(raw), 'history': history}
    finally:
        for p, flag in zip(parameters, grad_flags):
            p.requires_grad_(flag)
        model.train(training)


def mad_anomaly(norms, threshold=2.):
    """Lower-tail anomaly only after every target has a successful reconstruction."""
    finite = {int(k): float(v) for k, v in norms.items() if v is not None}
    if any(not math.isfinite(v) or v < 0 for v in finite.values()):
        raise ValueError('Successful mask norms must be finite and nonnegative')
    result = {'anomaly_index': None, 'flagged_labels': [], 'threshold': threshold,
              'successful_labels': len(finite), 'total_labels': len(norms), 'median': None, 'scaled_mad': None}
    if not finite:
        return dict(result, status='no_successful_reconstructions')
    if len(finite) != len(norms):
        return dict(result, status='incomplete_reconstructions')
    if len(finite) < 3:
        return dict(result, status='insufficient_labels')
    values = np.asarray(list(finite.values()))
    median = float(np.median(values))
    scaled_mad = float(1.4826 * np.median(np.abs(values - median)))
    result.update(median=median, scaled_mad=scaled_mad)
    if scaled_mad <= np.finfo(float).eps * max(1., abs(median)):
        return dict(result, status='zero_mad')
    return dict(result, status='ok', anomaly_index=float((median - values.min()) / scaled_mad),
                flagged_labels=[k for k, v in finite.items() if v < median and (median - v) / scaled_mad > threshold])


def _save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def evaluate_neural_cleanse(model, dataset, output, *, samples=256, options=None, progress=None):
    options = options or NCOptions()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results, norms = [], {}
    for target in range(len(dataset.classes)):
        indices = select_indices(dataset.targets, samples, options.seed, exclude_class=target)
        raw = torch.stack([dataset[int(i)][0] for i in indices])
        result = reconstruct_trigger(model, raw, target, options)
        mask, pattern = result.pop('mask'), result.pop('pattern')
        artifact = output / f'target-{target:03d}.npz'
        temporary = artifact.with_suffix('.npz.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, mask=mask.numpy(), pattern=pattern.numpy())
        temporary.replace(artifact)
        result.update(sample_indices=indices.tolist(), artifact=str(artifact.resolve()))
        results.append(result)
        norms[target] = result['best_successful_l1']
        # Persist partial progress as an incomplete report if a long scan stops.
        report = {'metric': 'neural-cleanse', 'source': NC_SOURCE, 'samples_requested_per_target': samples,
                  'options': asdict(options), 'step_definition': 'one optimization pass over selected samples, then exact selected-set ASR',
                  'parameterization': 'sigmoid 2D mask shared across channels; sigmoid RGB pattern; raw-domain blend',
                  'objective': 'cross_entropy + adaptive_lambda * sum(mask)', 'optimizer': 'Adam betas=(0.5,0.9)',
                  'replication_scope': 'Independent CIFAR settings; FiBA paper does not specify detector optimization defaults',
                  'targets': results, 'complete': len(results) == len(dataset.classes),
                  'anomaly': mad_anomaly({k: norms.get(k) for k in range(len(dataset.classes))})}
        _save_json(output / 'report.json', report)
        if progress is not None:
            progress(target, result['status'], result['success_rate'], result['mask_l1'])
    return report


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _dataset_identity(manifest):
    """Content identity excludes provenance paths and JSON presentation details."""
    try:
        files = sorted((entry['name'], entry['bytes'], entry['sha256']) for entry in manifest['files'])
        return {'dataset': manifest['dataset'], 'classes': manifest['classes'],
                'splits': manifest['splits'], 'files': files}
    except (KeyError, TypeError) as error:
        raise ValueError('Invalid saved dataset provenance manifest') from error


def verify_evaluation_provenance(checkpoint, cfg, dataset, *, requires_generator):
    """Bind evaluation to training artifacts before any LPIPS weight download.

    load_datasets has already checked actual NPY bytes against this manifest.
    Compare its file hashes/counts to training evidence, allowing a relocated
    cache to carry different source paths or manifest JSON formatting.
    """
    training = checkpoint.get('provenance', {})
    manifest_path = Path(dataset.directory) / 'manifest.json'
    actual_manifest = json.loads(manifest_path.read_text())
    identity = _dataset_identity(actual_manifest)
    actual_hash = _sha256_file(manifest_path)
    saved_data = training.get('dataset', {})
    if saved_data:
        if 'manifest' in saved_data:
            if identity != _dataset_identity(saved_data['manifest']):
                raise ValueError('Evaluation dataset differs from checkpoint training dataset provenance')
        elif saved_data.get('manifest_sha256') != actual_hash:
            raise ValueError('Evaluation dataset manifest differs from saved dataset provenance; content manifest missing from checkpoint')
    evidence = {'dataset': {'manifest_sha256': actual_hash,
                            'content_sha256': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                            'manifest': actual_manifest, 'matches_training_data': True if saved_data else None},
                'generator': None}
    if requires_generator and cfg.get('attack') in ('encoder', 'lira'):
        if not cfg.get('generator_checkpoint'):
            raise ValueError('Learned trigger requires --generator-checkpoint for evaluation')
        path = Path(cfg['generator_checkpoint']).expanduser().resolve(strict=True)
        actual_generator_hash = _sha256_file(path)
        expected = training.get('generator', {}).get('sha256')
        if expected is not None and actual_generator_hash != expected:
            raise ValueError('Evaluation generator SHA-256 differs from the classifier checkpoint training generator')
        evidence['generator'] = {'path': str(path), 'sha256': actual_generator_hash,
                                 'matches_training_generator': True if expected is not None else None}
    return evidence


def evaluate_checkpoint(checkpoint_path, *, metric='all', output=None, data_root=None, generator_checkpoint=None, device='auto', samples=256, options=None):
    from .network import build_model
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if not isinstance(checkpoint, dict) or 'model' not in checkpoint or 'config' not in checkpoint:
        raise ValueError('Expected FiBA classifier checkpoint with model and config')
    cfg = dict(checkpoint['config'])
    if data_root is not None:
        cfg['data_root'] = str(Path(data_root).expanduser().resolve())
    if generator_checkpoint is not None:
        cfg['generator_checkpoint'] = str(Path(generator_checkpoint).expanduser().resolve())
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if metric not in ('all', 'lpips', 'neural-cleanse'):
        raise ValueError('metric must be lpips, neural-cleanse or all')
    torch.set_num_threads(int(cfg.get('threads', 1)))
    _, test = load_datasets(cfg['dataset'], cfg['data_root'])
    actual_provenance = verify_evaluation_provenance(checkpoint, cfg, test, requires_generator=metric in ('all', 'lpips'))
    model = build_model(len(test.classes), cfg.get('seed', 42))
    model.load_state_dict(checkpoint['model'], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Classifier checkpoint contains nonfinite tensors')
    model.eval().to(device)
    output = Path(output) if output is not None else checkpoint_path.parent / 'detectability'
    output.mkdir(parents=True, exist_ok=True)
    options = options or NCOptions(seed=cfg.get('seed', 42))
    provenance = {'checkpoint': str(checkpoint_path), 'checkpoint_sha256': _sha256_file(checkpoint_path),
                  'training_round': checkpoint.get('round'), 'dataset': cfg['dataset'], 'data_root': cfg['data_root'],
                  'device': str(device), 'config': cfg, 'evaluation_provenance': actual_provenance,
                  'training_provenance': checkpoint.get('provenance', {})}
    reports = {}
    if metric in ('all', 'lpips'):
        result = evaluate_lpips(test, cfg, mask=checkpoint.get('attention_mask'), samples=samples,
                                batch_size=options.batch_size, device=device)
        reports['lpips'] = dict(result, **provenance)
        _save_json(output / 'lpips.json', reports['lpips'])
    if metric in ('all', 'neural-cleanse'):
        result = evaluate_neural_cleanse(model, test, output / 'neural-cleanse', samples=samples, options=options,
                    progress=lambda target, status, rate, norm: print(f'Neural Cleanse target={target} status={status} ASR={rate:.4f} mask_L1={norm:.4f}', flush=True))
        reports['neural-cleanse'] = dict(result, **provenance)
        _save_json(output / 'neural-cleanse.json', reports['neural-cleanse'])
    return reports
