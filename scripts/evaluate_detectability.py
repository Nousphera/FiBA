"""Evaluate a saved FiBA checkpoint with actual pretrained LPIPS or Neural Cleanse.

LPIPS lazily downloads the pretrained AlexNet backbone if it is not cached.
Neural Cleanse records all independent optimization settings and convergence;
unsuccessful scans never receive a clean verdict or a numerical anomaly score.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fiba.detectability import NCOptions, evaluate_checkpoint


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--metric', choices=('lpips', 'neural-cleanse', 'all'), default='all')
    p.add_argument('--output', type=Path)
    p.add_argument('--data-root', type=Path, help='Relocated prepared data; file hashes must match saved training provenance')
    p.add_argument('--generator-checkpoint', type=Path, help='Relocated learned generator; SHA-256 must match saved training provenance')
    p.add_argument('--device', default='auto')
    p.add_argument('--samples', type=int, default=256, help='Maximum sampled non-target images per metric/target')
    p.add_argument('--steps', type=int, default=200, help='NC passes over selected samples per target')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.1)
    p.add_argument('--init-cost', type=float, default=.001)
    p.add_argument('--success-threshold', type=float, default=.99)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--cost-multiplier', type=float, default=2.)
    p.add_argument('--seed', type=int, default=42, help='NC sampling and optimization seed, recorded in output')
    args = p.parse_args(argv)
    if args.samples < 1:
        p.error('--samples must be positive')
    options = NCOptions(steps=args.steps, batch_size=args.batch_size, lr=args.lr, init_cost=args.init_cost,
                        success_threshold=args.success_threshold, patience=args.patience,
                        cost_multiplier=args.cost_multiplier, seed=args.seed)
    reports = evaluate_checkpoint(args.checkpoint, metric=args.metric, output=args.output,
                                  data_root=args.data_root, generator_checkpoint=args.generator_checkpoint,
                                  device=args.device, samples=args.samples, options=options)
    print(json.dumps({key: {'status': value.get('status', value.get('anomaly', {}).get('status')),
                            'samples': value.get('samples'), 'anomaly_index': value.get('anomaly', {}).get('anomaly_index')}
                      for key, value in reports.items()}, sort_keys=True))
    return reports


if __name__ == '__main__':
    main()
