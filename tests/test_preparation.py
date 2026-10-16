"""Preparation rejects reduced corpora and detects interrupted or corrupt input."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import numpy as np
import pytest
from fiba import data

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_data', ROOT / 'scripts/prepare_data.py')
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def test_production_preparation_rejects_reduced_npz_without_publishing(tmp_path):
    source = tmp_path / 'reduced.npz'
    np.savez(source, train_images=np.zeros((3, 32, 32, 3), dtype=np.uint8), train_labels=[0, 1, 2], test_images=np.zeros((2, 32, 32, 3), dtype=np.uint8), test_labels=[0, 1])
    with pytest.raises(ValueError, match='50000|full'):
        data.prepare_dataset('cifar10', tmp_path / 'cache', source, download=False)
    assert not (tmp_path / 'cache/cifar10/manifest.json').exists()


def test_missing_cache_does_not_silently_fabricate_data(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.prepare_dataset('cifar100', tmp_path, download=False)
    with pytest.raises(ValueError):
        data.prepare_dataset('synthetic', tmp_path)


def test_manifest_hash_change_is_rejected(tmp_path):
    root = tmp_path / 'cifar10'
    root.mkdir()
    (root / 'train_images.npy').write_bytes(b'changed')
    (root / 'manifest.json').write_text(json.dumps({'schema': 1, 'dataset': 'cifar10', 'files': [{'name': 'train_images.npy', 'bytes': 7, 'sha256': '0' * 64}]}))
    with pytest.raises(ValueError, match='checksum|manifest|integrity'):
        data.prepare_dataset('cifar10', tmp_path, download=False)


def test_resumable_download_keeps_partial_and_checks_checksum(tmp_path, monkeypatch):
    # Local HTTP response double substitutes only the network boundary.
    payload = b'official archive bytes'
    target = tmp_path / 'archive.tar.gz'
    partial = target.with_suffix(target.suffix + '.part')
    partial.write_bytes(payload[:9])
    requests = []
    class Response(io.BytesIO):
        status = 206
        headers = {'Content-Range': f'bytes 9-{len(payload)-1}/{len(payload)}'}
    def urlopen(request, timeout):
        requests.append(request)
        return Response(payload[9:])
    monkeypatch.setattr(data.urllib.request, 'urlopen', urlopen)
    data.download_archive('https://official.invalid/data', target, hashlib.md5(payload).hexdigest())
    assert target.read_bytes() == payload and not partial.exists()
    assert requests[0].get_header('Range') == 'bytes=9-'
    target.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        data.download_archive('https://official.invalid/data', target, hashlib.md5(payload).hexdigest())


def test_cli_default_is_both_corpora_and_explicit_validation_never_downloads(tmp_path, monkeypatch):
    # The expensive full-corpus action is substituted; parser behavior is real.
    seen = []
    def prepare(dataset, root, source=None, download=True):
        seen.append((dataset, Path(root), source, download))
        return {'dataset': dataset}
    monkeypatch.setattr(prep, 'prepare_dataset', prepare)
    prep.main(['--root', str(tmp_path), '--validate'])
    assert [(x[0], x[3]) for x in seen] == [('cifar10', False), ('cifar100', False)]


def full_cache(root):
    """Sparse zero-valued pixel fixture with real full split shapes and labels."""
    root.mkdir()
    entries = []
    for split, rows in [('train', 50000), ('test', 10000)]:
        mmap = np.lib.format.open_memmap(root / f'{split}_images.npy', mode='w+', dtype=np.uint8, shape=(rows, 32, 32, 3))
        mmap.flush()
        del mmap
        np.save(root / f'{split}_labels.npy', np.arange(rows, dtype=np.int64) % 10)
    for path in sorted(root.glob('*.npy')):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        entries.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    (root / 'manifest.json').write_text(json.dumps({'schema': 1, 'dataset': 'cifar10', 'classes': 10, 'splits': {'train': 50000, 'test': 10000}, 'files': entries}))


def test_full_cache_import_is_idempotent_and_reuses_files_without_copy(tmp_path):
    source = tmp_path / 'original/cifar10'
    source.parent.mkdir()
    full_cache(source)
    root = tmp_path / 'target'
    first = data.prepare_dataset('cifar10', root, source=source, download=False)
    assert first == data.prepare_dataset('cifar10', root, download=False)
    assert (source / 'train_images.npy').stat().st_ino == (root / 'cifar10/train_images.npy').stat().st_ino
    train, test = data.load_datasets('cifar10', root)
    assert (len(train), len(test), len(train.classes)) == (50000, 10000, 10)
    np.testing.assert_array_equal(train.targets[:12], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1])
    with (root / 'cifar10/train_images.npy').open('r+b') as stream:
        stream.seek(150)
        stream.write(b'corrupt')
    with pytest.raises(ValueError, match='checksum'):
        data.prepare_dataset('cifar10', root, download=False)


def test_failed_publication_leaves_no_partial_dataset(tmp_path, monkeypatch):
    source = tmp_path / 'original/cifar10'
    source.parent.mkdir()
    full_cache(source)
    original = Path.rename
    def fail(self, target):
        if Path(target) == tmp_path / 'target/cifar10':
            raise OSError('simulated publication failure')
        return original(self, target)
    monkeypatch.setattr(Path, 'rename', fail)
    with pytest.raises(OSError, match='publication'):
        data.prepare_dataset('cifar10', tmp_path / 'target', source=source, download=False)
    assert not (tmp_path / 'target/cifar10').exists()
    assert not list((tmp_path / 'target').glob('.cifar10-prepare-*'))
    assert (source / 'manifest.json').exists()


def test_completed_partial_archive_is_reused_without_network(tmp_path, monkeypatch):
    payload = b'already fully downloaded'
    target = tmp_path / 'archive.tar.gz'
    target.with_suffix('.gz.part').write_bytes(payload)
    def offline(*args, **kwargs):
        raise AssertionError('A completed partial archive must not access the network')
    monkeypatch.setattr(data.urllib.request, 'urlopen', offline)
    data.download_archive('https://official.invalid/data', target, hashlib.md5(payload).hexdigest())
    assert target.read_bytes() == payload
