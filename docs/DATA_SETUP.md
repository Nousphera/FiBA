# Dataset preparation

Run commands from the repository root in the README's environment.
`python scripts/prepare_data.py --help` lists datasets and modes.
`--download` explicitly enables network acquisition; `--source` imports a safe
NPZ or already-prepared directory. Import rejects existing destination files and
symlinks in directory trees. Imports validate in a temporary staging directory before publishing; failed imports leave no dataset or manifest behind. Directory imports copy only the selected dataset subtree, and auxiliary imports copy only supported media files. `--validate` reads schema and content again and
compares the deterministic SHA-256 manifest; it never downloads. Omitting all
three modes registers an already-present dataset after validating its schema.

Prepared image files contain `train_images`, `train_labels`, `test_images`, and
`test_labels`. RGB images are uint8 NHWC with shape `(N,32,32,3)` for CIFAR.
Labels are integer IDs shaped `(N,)` or `(N,1)`. Preparation checks shapes,
class ranges, nonempty splits, and counts. Training reads these files directly
and preserves their row order.

```bash
python scripts/prepare_data.py --dataset cifar10 --root datasets --source /path/to/cifar10.npz
python scripts/prepare_data.py --dataset cifar10 --root datasets --validate
```

A manifest records dataset name, split sizes, class count and relative file
paths/content hashes. Retain the source publication, data license, and acquisition record alongside
this manifest. A changed file requires a new destination/preparation run;
manifests are not silently regenerated over changed data.

## Public vision sources

[CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html) use the official
train/test splits via the pinned torchvision loader. Files are saved
as `ROOT/cifar10.npz` and `ROOT/cifar100.npz` for training:

```bash
python scripts/prepare_data.py --dataset cifar10 --root datasets --download
python scripts/prepare_data.py --dataset cifar100 --root datasets --download
```

Use `--data-root datasets` in either `fiba` or Flower's `options`. When a prepared
NPZ is present, poisoning, partitioning, and training consume it
through the same torchvision-compatible data interface. The script validates
all examples against the fixed 10/100-class spaces; it does not silently infer
a smaller classifier from a subset. CIFAR files retain their original terms.

Encoder pretraining and external LIRA TorchScript contracts are documented in
[EXPERIMENTS.md](EXPERIMENTS.md). Train or supply those artifacts separately before selecting the corresponding
attack. Client partitions are generated from the configured seed.
