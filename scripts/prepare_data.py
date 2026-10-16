"""Download, resume, import or validate full CIFAR10 and CIFAR100 caches.

Run without --dataset to prepare both paper datasets. Repeating a command reuses
verified data; interrupted official downloads resume from their partial archive.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fiba.data import prepare_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('cifar10', 'cifar100'))
    parser.add_argument('--root', type=Path, default=Path('data'))
    parser.add_argument('--source', type=Path, help='Prepared NPY directory with manifest, or safe full-corpus NPZ')
    parser.add_argument('--validate', action='store_true', help='Verify existing data without downloading')
    args = parser.parse_args(argv)
    if args.validate and args.source:
        parser.error('--source cannot be combined with --validate')
    if args.source and not args.dataset and not args.source.is_dir():
        parser.error('A source NPZ requires --dataset; a parent NPY directory can contain both datasets')
    results = []
    for dataset in (args.dataset,) if args.dataset else ('cifar10', 'cifar100'):
        result = prepare_dataset(dataset, args.root, source=args.source, download=not args.validate)
        print(json.dumps(result, sort_keys=True))
        results.append(result)
    return results[0] if args.dataset else results


if __name__ == '__main__':
    main()
