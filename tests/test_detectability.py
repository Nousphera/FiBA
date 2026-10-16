"""Algorithm tests use differentiable classifiers, never surrogate metric scores."""
import importlib
import json
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn


def implementation():
    return importlib.import_module('fiba.detectability')


def test_lpips_input_conversion_uses_raw_rgb_without_classifier_normalization():
    detect = implementation()
    clean = torch.zeros(2, 3, 32, 32)
    triggered = torch.ones_like(clean)
    a, b = detect.lpips_inputs(clean, triggered)
    assert torch.equal(a, -torch.ones_like(clean)) and torch.equal(b, torch.ones_like(clean))
    assert torch.equal(clean, torch.zeros_like(clean))
    with pytest.raises(ValueError):
        detect.lpips_inputs(clean - .1, triggered)


def test_sample_selection_excludes_target_and_respects_optional_source():
    detect = implementation()
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 2])
    selected = detect.select_indices(labels, 3, seed=17, exclude_class=2, source_class=0)
    assert len(selected) == 3
    assert set(selected) == {0, 3, 6}
    assert np.array_equal(selected, detect.select_indices(labels, 3, seed=17, exclude_class=2, source_class=0))
    with pytest.raises(ValueError, match='eligible'):
        detect.select_indices(labels, 5, exclude_class=2, source_class=2)


def test_mad_uses_lower_tail_and_handles_incomplete_or_zero_dispersion():
    detect = implementation()
    result = detect.mad_anomaly({0: 1., 1: 9., 2: 10., 3: 11., 4: 12.})
    assert result['anomaly_index'] == pytest.approx(9 / 1.4826)
    assert result['flagged_labels'] == [0] and result['status'] == 'ok'
    for values, status in [({0: None, 1: None, 2: None}, 'no_successful_reconstructions'),
                           ({0: 1., 1: None, 2: 3.}, 'incomplete_reconstructions'),
                           ({0: 1., 1: 1., 2: 1.}, 'zero_mad')]:
        output = detect.mad_anomaly(values)
        assert output['status'] == status and output['anomaly_index'] is None
        json.dumps(output, allow_nan=False)


class PixelClassifier(nn.Module):
    """Known one-pixel trigger: raw red(0,0) above .7 predicts target1."""
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(20.))
    def forward(self, images):
        logit = self.scale * (images[:, 0, 0, 0] - .7)
        return torch.stack((-logit, logit), 1)


def test_neural_cleanse_finds_small_successful_mask_and_preserves_classifier():
    detect = implementation()
    torch.set_num_threads(1)
    model = PixelClassifier().train()
    before = model.scale.detach().clone()
    options = detect.NCOptions(steps=100, lr=.2, init_cost=.02, patience=2, success_threshold=1., batch_size=4, seed=9)
    result = detect.reconstruct_trigger(model, torch.zeros(8, 3, 32, 32), 1, options, normalizer=lambda x: x)
    assert result['status'] == 'success' and result['success_rate'] == 1.
    assert result['mask_l1'] < 30
    assert result['mask'].shape == (1, 32, 32) and result['pattern'].shape == (3, 32, 32)
    crafted = (1 - result['mask']) * torch.zeros(8, 3, 32, 32) + result['mask'] * result['pattern']
    assert torch.all(model(crafted).argmax(1) == 1)
    assert min(item['cost'] for item in result['history']) == 0
    assert max(item['cost'] for item in result['history']) > options.init_cost
    torch.testing.assert_close(model.scale, before)
    assert model.training and model.scale.requires_grad and model.scale.grad is None
    successful = [x['mask_l1'] for x in result['history'] if x['success_rate'] >= options.success_threshold]
    assert result['mask_l1'] == pytest.approx(min(successful))


class ImpossibleClassifier(nn.Module):
    def forward(self, images):
        zero = images.flatten(1).sum(1) * 0
        return torch.stack((zero + 10, zero - 10), 1)


def test_failed_reconstruction_reports_null_best_mask_norm():
    detect = implementation()
    result = detect.reconstruct_trigger(ImpossibleClassifier(), torch.zeros(4, 3, 32, 32), 1,
                                       detect.NCOptions(steps=4, patience=1, batch_size=2), normalizer=lambda x: x)
    assert result['status'] == 'no_successful_reconstruction'
    assert result['best_successful_l1'] is None and result['success_rate'] == 0
    assert torch.isfinite(result['mask']).all()
    assert detect.mad_anomaly({0: 1., 1: result['best_successful_l1'], 2: 3.})['anomaly_index'] is None


def test_lpips_integration_with_real_pretrained_alex_only_if_weights_cached():
    # This test must never download a quarter-gigabyte backbone onto the Mac.
    weight = Path(torch.hub.get_dir()) / 'checkpoints/alexnet-owt-7be5be79.pth'
    if not weight.exists():
        pytest.skip('Real pretrained AlexNet weights are not cached; no test download')
    detect = implementation()
    metric = detect.make_lpips('cpu')
    x = torch.zeros(2, 3, 32, 32)
    a, b = detect.lpips_inputs(x, x)
    with torch.inference_mode():
        values = metric(a, b)
    torch.testing.assert_close(values, torch.zeros_like(values))


def saved_provenance_fixture(tmp_path):
    """Real small provenance files; full-corpus validation is tested in data tests."""
    import hashlib
    from types import SimpleNamespace
    directory = tmp_path / 'cifar10'
    directory.mkdir()
    manifest = {'schema': 1, 'dataset': 'cifar10', 'classes': 10, 'splits': {'train': 50000, 'test': 10000},
                'source': '/training/machine/cache',
                'files': [{'name': name, 'bytes': i + 100, 'sha256': str(i) * 64}
                          for i, name in enumerate(('train_images.npy','train_labels.npy','test_images.npy','test_labels.npy'))]}
    (directory / 'manifest.json').write_text(json.dumps(manifest))
    generator = tmp_path / 'encoder.pt'
    generator.write_bytes(b'frozen trained generator artifact')
    checkpoint = {'provenance': {'dataset': {'manifest': manifest,
                    'manifest_sha256': hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()},
                    'generator': {'sha256': hashlib.sha256(generator.read_bytes()).hexdigest()}}}
    cfg = {'dataset': 'cifar10', 'attack': 'encoder', 'generator_checkpoint': str(generator)}
    return checkpoint, cfg, SimpleNamespace(directory=str(directory))


def test_detectability_rejects_changed_generator_and_changed_dataset_provenance(tmp_path):
    detect = implementation()
    checkpoint, cfg, dataset = saved_provenance_fixture(tmp_path)
    Path(cfg['generator_checkpoint']).write_bytes(b'different generator weights')
    with pytest.raises(ValueError, match='generator.*SHA-256'):
        detect.verify_evaluation_provenance(checkpoint, cfg, dataset, requires_generator=True)
    manifest_path = Path(dataset.directory)/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][0]['sha256'] = 'f' * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='dataset.*provenance'):
        detect.verify_evaluation_provenance(checkpoint, cfg, dataset, requires_generator=False)


def test_relocated_identical_generator_and_data_content_match_despite_manifest_source(tmp_path):
    import hashlib
    detect = implementation()
    checkpoint, cfg, dataset = saved_provenance_fixture(tmp_path)
    relocated = tmp_path/'moved-encoder.pt'
    Path(cfg['generator_checkpoint']).rename(relocated)
    cfg['generator_checkpoint'] = str(relocated)
    manifest_path = Path(dataset.directory)/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['source'] = '/evaluation/machine/reimport'
    manifest['files'].reverse()  # JSON ordering is not corpus identity.
    manifest_path.write_text(json.dumps(manifest, indent=2))
    result = detect.verify_evaluation_provenance(checkpoint, cfg, dataset, requires_generator=True)
    assert result['generator']['sha256'] == hashlib.sha256(relocated.read_bytes()).hexdigest()
    assert result['generator']['matches_training_generator'] is True
    assert result['dataset']['matches_training_data'] is True
    assert result['dataset']['manifest_sha256'] != checkpoint['provenance']['dataset']['manifest_sha256']
    # Neural Cleanse needs only the classifier and matching data, even if the
    # original learned-generator artifact is no longer available.
    relocated.unlink()
    assert detect.verify_evaluation_provenance(checkpoint, cfg, dataset, requires_generator=False)['generator'] is None


def test_checkpoint_evaluation_fails_on_drift_before_loading_lpips_weights(tmp_path, monkeypatch):
    detect = implementation()
    checkpoint, cfg, dataset = saved_provenance_fixture(tmp_path)
    dataset.classes = list(range(10))
    cfg.update(data_root=str(tmp_path), threads=1, seed=42)
    checkpoint.update(config=cfg, model={})
    checkpoint_path = tmp_path/'classifier.pt'
    torch.save(checkpoint, checkpoint_path)
    # Full-corpus schema/hashing is covered by preparation tests. Substitute
    # that expensive I/O boundary, leaving the actual provenance guard real.
    monkeypatch.setattr(detect, 'load_datasets', lambda *_: (dataset, dataset))
    def no_weights(*args, **kwargs):
        raise AssertionError('A drifted run must fail before loading/downloading LPIPS')
    monkeypatch.setattr(detect, 'make_lpips', no_weights)
    Path(cfg['generator_checkpoint']).write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='generator.*SHA-256'):
        detect.evaluate_checkpoint(checkpoint_path, metric='lpips', device='cpu')
    manifest_path = Path(dataset.directory)/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['files'][0]['sha256'] = 'f'*64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='dataset.*provenance'):
        detect.evaluate_checkpoint(checkpoint_path, metric='neural-cleanse', device='cpu')
