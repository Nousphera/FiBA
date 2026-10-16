"""Full CIFAR corpora, bounded memory access, and reproducible client partitions.

Images remain uint8 memory maps until an individual sample is requested. Public
preparation accepts only the official 50,000/10,000 splits; unit tests can create
CIFARDataset directly without adding a reduced-corpus training mode.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import tarfile
import tempfile
import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset

DATASETS = {
    'cifar10': {'classes': 10, 'url': 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                'archive': 'cifar-10-python.tar.gz', 'md5': 'c58f30108f718f92721af3b95e74349a',
                'folder': 'cifar-10-batches-py',
                'train': [f'data_batch_{i}' for i in range(1, 6)], 'test': ['test_batch'], 'label_key': 'labels'},
    'cifar100': {'classes': 100, 'url': 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz',
                 'archive': 'cifar-100-python.tar.gz', 'md5': 'eb9058c3a382ffc7106e4002c42a8d85',
                 'folder': 'cifar-100-python', 'train': ['train'], 'test': ['test'], 'label_key': 'fine_labels'},
}
FILES = ('train_images.npy', 'train_labels.npy', 'test_images.npy', 'test_labels.npy')
MEAN = (.4914, .4822, .4465)
STD = (.2023, .1994, .2010)
_VERIFIED = {}


def _spec(dataset):
    if dataset not in DATASETS:
        raise ValueError('dataset must be cifar10 or cifar100')
    return DATASETS[dataset]


def _directory(dataset, root):
    root = Path(root).expanduser().resolve()
    return root if root.name == dataset else root / dataset


def _digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _validate_arrays(directory, classes, full=True):
    for split, count in (('train', 50000), ('test', 10000)):
        images = np.load(directory / f'{split}_images.npy', mmap_mode='r', allow_pickle=False)
        labels = np.load(directory / f'{split}_labels.npy', mmap_mode='r', allow_pickle=False)
        labels = labels.reshape(-1) if labels.ndim in (1, 2) and (labels.ndim == 1 or labels.shape[1] == 1) else labels
        if images.dtype != np.uint8 or images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ValueError('CIFAR images must be uint8 NHWC RGB 32x32 arrays')
        if full and len(images) != count:
            raise ValueError(f'Full CIFAR {split} split requires {count} examples; got {len(images)}')
        if labels.ndim != 1 or len(labels) != len(images) or not len(labels) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError('CIFAR requires one integer label per image')
        if np.any((labels < 0) | (labels >= classes)):
            raise ValueError('CIFAR label is outside its class space')
        if full and not np.all(np.bincount(labels, minlength=classes) == count // classes):
            raise ValueError('Full official CIFAR splits require balanced class counts')


def _manifest(dataset, directory, source):
    return {'schema': 1, 'dataset': dataset, 'classes': _spec(dataset)['classes'],
            'splits': {'train': 50000, 'test': 10000}, 'source': source,
            'files': [{'name': name, 'bytes': (directory / name).stat().st_size,
                       'sha256': _digest(directory / name)} for name in FILES]}


def _verify_cache(dataset, directory):
    path = directory / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'Prepared manifest missing: {path}; run scripts/prepare_data.py')
    try:
        manifest = json.loads(path.read_text())
        entries = manifest['files']
        if manifest['schema'] != 1 or manifest['dataset'] != dataset or {x['name'] for x in entries} != set(FILES) or len(entries) != len(FILES):
            raise ValueError('Invalid preparation manifest')
        signature = tuple((p.name, p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in [path, *(directory / name for name in FILES)])
        key = str(directory)
        if _VERIFIED.get(key) != signature:
            for entry in entries:
                p = directory / entry['name']
                if p.is_symlink() or p.stat().st_size != entry['bytes'] or _digest(p) != entry['sha256']:
                    raise ValueError(f'Prepared data checksum mismatch: {p}')
            _validate_arrays(directory, _spec(dataset)['classes'])
            if len(_VERIFIED) >= 32:
                _VERIFIED.clear()
            _VERIFIED[key] = signature
        return manifest
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Invalid preparation manifest or integrity failure: {path}') from error


class CIFARDataset(Dataset):
    """Read-only mmap image storage; each returned float tensor owns its memory."""
    def __init__(self, directory, split, classes):
        if split not in ('train', 'test'):
            raise ValueError('split must be train or test')
        self.directory, self.split = str(directory), split
        self.classes = list(range(classes))
        self._open()
        if self.images.dtype != np.uint8 or self.images.shape != (len(self.targets), 32, 32, 3):
            raise ValueError('Invalid CIFAR image array')
        if not np.issubdtype(self.targets.dtype, np.integer) or np.any((self.targets < 0) | (self.targets >= classes)):
            raise ValueError('Invalid CIFAR label array')

    def _open(self):
        root = Path(self.directory)
        self.images = np.load(root / f'{self.split}_images.npy', mmap_mode='r', allow_pickle=False)
        self.targets = np.load(root / f'{self.split}_labels.npy', mmap_mode='r', allow_pickle=False).reshape(-1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        # Copy only one 3-KiB image, avoiding a writable view of read-only storage.
        image = torch.from_numpy(np.array(self.images[index], copy=True)).permute(2, 0, 1).float().div_(255)
        return image, int(self.targets[index])

    def __getstate__(self):
        # Spawn workers reopen maps instead of serializing the whole dataset.
        state = self.__dict__.copy()
        state.pop('images')
        state.pop('targets')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._open()


def load_datasets(dataset: str, root: str):
    """Load an already prepared full corpus. Preparation is a separate one-call step."""
    spec = _spec(dataset)
    directory = _directory(dataset, root)
    _verify_cache(dataset, directory)
    return tuple(CIFARDataset(directory, split, spec['classes']) for split in ('train', 'test'))


def normalize(images):
    """Original CIFAR normalization, shared by CIFAR10 and CIFAR100 experiments."""
    if images.ndim not in (3, 4) or images.shape[-3:] != (3, 32, 32) or not images.is_floating_point():
        raise ValueError('Expected floating point CHW or NCHW RGB 32x32 tensors')
    mean = images.new_tensor(MEAN).view(3, 1, 1)
    std = images.new_tensor(STD).view(3, 1, 1)
    return (images - mean) / std


def augment_raw(images, generator=None):
    """Independent 50% horizontal flips on raw images, before trigger embedding.

    The original resize/center crop at 32x32 is the identity for CIFAR images.
    """
    batch = images.unsqueeze(0) if images.ndim == 3 else images
    if batch.ndim != 4:
        raise ValueError('Expected CHW or NCHW image tensors')
    random_device = generator.device if generator is not None else 'cpu'
    flip = (torch.rand(len(batch), generator=generator, device=random_device) < .5).to(batch.device)
    result = torch.where(flip[:, None, None, None], batch.flip(-1), batch)
    return result[0] if images.ndim == 3 else result


def augment(images, generator=None):
    """Original horizontal flip plus normalization for clean-only consumers."""
    return normalize(augment_raw(images, generator))


def partition_indices(labels, num_clients, split='dirichlet', alpha=.5, seed=42):
    """Seeded, complete and non-overlapping IID or class-Dirichlet partitions.

    Matches the original class-wise balancing rule. Retry at most 32 times for
    nonempty clients, then transfer one example from the largest donor to each
    empty client. This explicit cardinality repair bounds extreme-alpha cases.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or len(labels) < num_clients or num_clients < 1:
        raise ValueError('Need integer labels and at least one example per client')
    if split not in ('iid', 'dirichlet') or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('split must be iid/dirichlet and alpha must be positive')
    rng = np.random.default_rng(seed)
    if split == 'iid':
        return [x.astype(np.int64) for x in np.array_split(rng.permutation(len(labels)), num_clients)]
    for _ in range(32):
        parts = [[] for _ in range(num_clients)]
        for label in np.unique(labels):
            indices = rng.permutation(np.flatnonzero(labels == label))
            if alpha < 1e-5:
                log_gamma = np.log(rng.gamma(alpha + 1, 1, num_clients)) + np.log(np.maximum(rng.random(num_clients), np.finfo(float).tiny)) / alpha
                weights = np.exp(log_gamma - log_gamma.max())
            else:
                weights = rng.dirichlet(np.full(num_clients, alpha))
            weights *= np.array([len(p) < len(labels) / num_clients for p in parts])
            if weights.sum() == 0:
                weights = np.array([len(p) < len(labels) / num_clients for p in parts], dtype=float)
            if weights.sum() == 0:
                weights = np.ones(num_clients)
            boundaries = (np.cumsum(weights / weights.sum()) * len(indices)).astype(int)[:-1]
            for part, block in zip(parts, np.split(indices, boundaries)):
                part.extend(block.tolist())
        if all(parts):
            break
    for i in range(num_clients):
        if not parts[i]:
            donor = int(np.argmax([len(p) for p in parts]))
            parts[i].append(parts[donor].pop())
    return [rng.permutation(part).astype(np.int64) for part in parts]


def partition_metadata(labels, partitions, *, split='dirichlet', alpha=.5, seed=42):
    labels = np.asarray(labels, dtype=np.int64)
    h = hashlib.sha256()
    for indices in partitions:
        array = np.asarray(indices, dtype='<i8')
        h.update(len(array).to_bytes(8, 'little'))
        h.update(array.tobytes())
    return {'split': split, 'alpha': alpha, 'seed': seed, 'clients': len(partitions),
            'sample_counts': [len(p) for p in partitions],
            'label_sha256': hashlib.sha256(labels.astype('<i8').tobytes()).hexdigest(),
            'partition_sha256': h.hexdigest(),
            'policy': 'class-wise balance; 32 bounded retries; singleton repair if needed'}


def download_archive(url, destination, md5):
    """Resume an interrupted official archive and verify its pinned MD5 before use."""
    destination = Path(destination)
    if destination.exists():
        if _digest(destination, 'md5') != md5:
            raise ValueError(f'Official archive checksum mismatch: {destination}; remove this corrupt archive and retry')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + '.part')
    if partial.exists() and _digest(partial, 'md5') == md5:
        os.replace(partial, destination)
        return destination
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request, timeout=60) as response:
        resumed = offset and response.status == 206
        if resumed and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
            raise ValueError('Invalid HTTP resume range; partial archive retained')
        with partial.open('ab' if resumed else 'wb') as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
    if _digest(partial, 'md5') != md5:
        # A truncated response can resume; a complete but invalid response must
        # fail closed, without publishing an unverified archive.
        raise ValueError(f'Official archive checksum mismatch: {partial}; retry to resume or remove a corrupt partial file')
    os.replace(partial, destination)
    return destination


@contextmanager
def _lock(path):
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def _from_archive(dataset, archive, directory):
    spec = _spec(dataset)
    if _digest(archive, 'md5') != spec['md5']:
        raise ValueError('Official archive checksum mismatch')
    # No extraction to disk. Only exact expected regular files are read, after
    # checking the official archive hash, before decoding its trusted pickle.
    with tarfile.open(archive, 'r:gz') as source:
        for split, count in (('train', 50000), ('test', 10000)):
            images = np.lib.format.open_memmap(directory / f'{split}_images.npy', mode='w+', dtype=np.uint8, shape=(count, 32, 32, 3))
            labels = np.lib.format.open_memmap(directory / f'{split}_labels.npy', mode='w+', dtype=np.int64, shape=(count,))
            offset = 0
            for batch in spec[split]:
                member = source.getmember(f"{spec['folder']}/{batch}")
                if not member.isfile():
                    raise ValueError('Official CIFAR batch is not a regular file')
                with source.extractfile(member) as stream:
                    payload = pickle.load(stream, encoding='latin1')
                rows = np.asarray(payload['data'])
                targets = np.asarray(payload[spec['label_key']])
                n = len(rows)
                if rows.shape != (n, 3072) or offset + n > count or targets.shape != (n,):
                    raise ValueError('Invalid official CIFAR batch shape')
                images[offset:offset + n] = rows.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                labels[offset:offset + n] = targets
                offset += n
                del payload, rows, targets
            if offset != count:
                raise ValueError('Incomplete official CIFAR split')
            images.flush()
            labels.flush()
            del images, labels


def _from_source(dataset, source, directory):
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_dir():
        source = _directory(dataset, source)
        _verify_cache(dataset, source)
        for name in FILES:
            try:
                os.link(source / name, directory / name)
            except OSError:
                shutil.copyfile(source / name, directory / name)
        return f'verified prepared cache: {source}'
    if source.suffix != '.npz':
        raise ValueError('Source must be a validated prepared NPY directory or a pickle-free NPZ')
    with np.load(source, allow_pickle=False) as arrays:
        for name in FILES:
            key = name[:-4]
            value = arrays[key]
            np.save(directory / name, value, allow_pickle=False)
            del value
    return f'explicit NPZ import sha256:{_digest(source)}'


def prepare_dataset(dataset, root, source=None, download=True):
    """Idempotently validate/reuse or atomically prepare one full CIFAR dataset.

    Completed caches are never replaced. Downloads retain .part bytes on
    interruption, and can be resumed by repeating the same command.
    """
    spec = _spec(dataset)
    directory = _directory(dataset, root)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with _lock(directory.parent / f'.{dataset}.lock'):
        if directory.exists():
            return _verify_cache(dataset, directory)
        if source is None and not download:
            raise FileNotFoundError(f'Full {dataset} cache missing: {directory}')
        with tempfile.TemporaryDirectory(prefix=f'.{dataset}-prepare-', dir=directory.parent) as temporary:
            staging = Path(temporary) / dataset
            staging.mkdir()
            if source is not None:
                provenance = _from_source(dataset, source, staging)
            else:
                archive = download_archive(spec['url'], directory.parent / '.downloads' / spec['archive'], spec['md5'])
                _from_archive(dataset, archive, staging)
                provenance = {'url': spec['url'], 'archive_md5': spec['md5'], 'split_order': 'official archive order'}
            _validate_arrays(staging, spec['classes'])
            manifest = _manifest(dataset, staging, provenance)
            (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
            staging.rename(directory)
        return _verify_cache(dataset, directory)
