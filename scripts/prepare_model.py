#!/usr/bin/env python3
"""Train the shared Encoder or LIRA trigger generator on CIFAR training data."""
import argparse
from pathlib import Path
from fiba.generator_training import DEFAULTS, train_generator


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attack',choices=('encoder','lira'),required=True)
    p.add_argument('--dataset',choices=('cifar10','cifar100'),default='cifar10')
    p.add_argument('--data-root',type=Path,default=Path('data'))
    p.add_argument('--output',type=Path)
    p.add_argument('--device',default='auto')
    p.add_argument('--attention-mask',type=Path)
    p.add_argument('--attention-checkpoint',type=Path,help='Derive a protected mask from a trained FiBA model')
    p.add_argument('--attention',choices=('low','high'),default=None,
                   help='Placement for a new mask; omitted placement inherits an existing Encoder checkpoint')
    p.add_argument('--initial-checkpoint',type=Path,help='Fine-tune an existing Encoder generator')
    for key in ('seed','generator_epochs','batch_size','num_workers','encoder_bits','lira_classifier_epochs','target_class'):
        p.add_argument('--'+('epochs' if key=='generator_epochs' else key.replace('_','-')),dest=key,type=int,default=DEFAULTS[key])
    for key in ('lr','generator_lr','classifier_lr','encoder_beta','attention_penalty','epsilon','lira_alpha'):
        p.add_argument('--'+key.replace('_','-'),type=float,default=DEFAULTS[key])
    p.add_argument('--lira-reset-classifier',action=argparse.BooleanOptionalAction,default=True)
    args=p.parse_args(argv)
    cfg=vars(args)
    cfg['data_root']=str(args.data_root.expanduser().resolve())
    cfg['output']=str((args.output or Path('checkpoints')/f'{args.attack}-{args.dataset}.pt').expanduser().resolve())
    if cfg['attention_mask'] is None:
        cfg.pop('attention_mask')
    try:
        source=cfg.pop('attention_checkpoint')
        placement=cfg.pop('attention')
        if placement is not None:
            cfg['attention_placement']=placement
        if cfg['initial_checkpoint'] is None:
            cfg.pop('initial_checkpoint')
        if source is not None:
            if cfg['attack'] != 'encoder' or 'attention_mask' in cfg:
                raise ValueError('Use one attention source with --attack encoder')
            cfg.setdefault('attention_placement','low')
            import hashlib
            import torch
            from fiba.attention import compute_attention
            from fiba.config import defaults, resolve_device
            from fiba.data import load_datasets, partition_indices
            from fiba.training import load_model
            from torch.utils.data import Subset
            saved=torch.load(source,map_location='cpu',weights_only=True)
            if saved['config']['dataset'] != cfg['dataset']:
                raise ValueError('Attention checkpoint and generator dataset must match')
            attention={**defaults(),**saved['config'],'attention':cfg['attention_placement'],
                       'rankme_threshold':0.,'device':resolve_device(cfg['device'])}
            train,_=load_datasets(cfg['dataset'],cfg['data_root'])
            indices=partition_indices(train.targets,attention['num_clients'],attention['split'],attention['alpha'],attention['seed'])[0]
            cfg['attention_mask'],_=compute_attention(load_model(saved['model'],attention),Subset(train,indices),attention)
            cfg['attention_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
        print(f'Saved {train_generator(cfg)}')
    except (OSError,ValueError,RuntimeError) as exc:
        p.exit(2,f'Model preparation failed: {exc}\n')


if __name__=='__main__':
    main()
