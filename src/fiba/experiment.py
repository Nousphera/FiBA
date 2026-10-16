"""Run the FiBA benchmark with native Flower clients."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from .config import ATTACKS, AGGREGATIONS, DATASETS, defaults, resolve_device, validate


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    cfg = defaults()
    for name, choices in dict(dataset=DATASETS, attack=ATTACKS, aggregation=AGGREGATIONS,
                              split=('iid', 'dirichlet'), attention=('none','low','high'),
                              backend=('flower','sequential')).items():
        p.add_argument('--' + name, choices=choices, default=cfg[name])
    aliases = dict(num_clients='nr-clients', num_rounds='rounds', lr='learning-rate')
    for name in ('num_clients','num_rounds','local_epochs','batch_size','seed','num_workers','threads',
                 'target_class','source_class','byzantine_clients','krum_keep','trigger_characters',
                 'wanet_k','attention_start_round','attention_samples','cam_samples'):
        p.add_argument('--' + aliases.get(name,name.replace('_','-')), dest=name, type=int, default=cfg[name])
    for name in ('lr','participation','alpha','attacker_fraction','poison_fraction','trim_fraction',
                 'wanet_strength','epsilon','rankme_threshold','registration_timeout','reply_timeout'):
        p.add_argument('--' + aliases.get(name,name.replace('_','-')), dest=name, type=float, default=cfg[name])
    p.add_argument('--device', default='auto')
    p.add_argument('--data-root', type=Path, default=Path('data'))
    p.add_argument('--generator-checkpoint', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--distributed', action='store_true')
    p.add_argument('--model-replacement', action='store_true')
    p.add_argument('--replacement-rounds', default=cfg['replacement_rounds'])
    p.add_argument('--parallel-clients', type=int, default=1)
    p.add_argument('--client-cpus', type=int, default=1)
    p.add_argument('--client-gpus', type=float)
    p.add_argument('--dry-run', action='store_true')
    return p


def build_config(args):
    cfg = defaults()
    for key in cfg:
        value = getattr(args,key,None)
        if value is not None:
            cfg[key] = value
    cfg['data_root'] = str(args.data_root.expanduser().resolve())
    cfg['device'] = resolve_device(args.device)
    cfg['generator_checkpoint'] = str(args.generator_checkpoint.expanduser().resolve()) if args.generator_checkpoint else ''
    if cfg['attack'] in ('encoder','lira') and not cfg['generator_checkpoint']:
        cfg['generator_checkpoint'] = str((Path('checkpoints') / f"{cfg['attack']}-{cfg['dataset']}.pt").resolve())
    name = f"{cfg['attack']}-{cfg['dataset']}-{cfg['aggregation']}-seed{cfg['seed']}"
    cfg['output_dir'] = str((args.output or Path('results') / name).expanduser().resolve())
    if args.parallel_clients < 1 or args.client_cpus < 1:
        raise ValueError('Client CPU counts and parallelism must be positive')
    if args.client_gpus is not None and (args.client_gpus < 0 or not __import__('math').isfinite(args.client_gpus)):
        raise ValueError('client-gpus must be finite and nonnegative')
    return validate(cfg)


@contextmanager
def flower_app(cfg):
    """Materialize only the descriptor/source Flower needs for this run."""
    import tomli_w
    package = Path(__file__).resolve().parent
    descriptor = {'project': {'name':'fiba','version':'2.0.0','description':__doc__,
                              'license':'GPL-3.0-only','requires-python':'>=3.11,<3.14',
                              'dependencies':[]},
                  'tool': {'flwr': {'app': {'publisher':'nousphera','components': {
                      'serverapp':'fiba.server_app:app','clientapp':'fiba.client_app:app'}, 'config':cfg}}}}
    with TemporaryDirectory(prefix='fiba-flower-') as folder:
        root = Path(folder)
        shutil.copytree(package, root/'fiba', ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        (root/'pyproject.toml').write_text(tomli_w.dumps(descriptor))
        yield root


def flower_command(args,cfg,app,config_file):
    executable = Path(sys.executable).with_name('flwr')
    if not executable.is_file():
        raise RuntimeError('Install requirements.txt in the active environment first')
    gpus = args.client_gpus if args.client_gpus is not None else float(cfg['device'].startswith('cuda'))
    if cfg['device'].startswith('cuda') and gpus <= 0:
        raise ValueError('CUDA clients need positive --client-gpus')
    federation = (f"num-supernodes={cfg['num_clients']} client-resources-num-cpus={args.client_cpus} "
                  f"client-resources-num-gpus={gpus} init-args-num-cpus={args.client_cpus*args.parallel_clients}")
    return [str(executable),'run',str(app),'local','--run-config',str(config_file),
            '--federation-config',federation,'--stream']


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        cfg = build_config(args)
        if args.dry_run:
            print(json.dumps(cfg,indent=2))
            return
        from .data import load_datasets, partition_indices
        from .attacks import build_attack
        train, _ = load_datasets(cfg['dataset'],cfg['data_root'])
        partition_indices(train.targets,cfg['num_clients'],cfg['split'],cfg['alpha'],cfg['seed'])
        build_attack(cfg)  # Fail before starting Flower if generator weights are missing.
        output = Path(cfg['output_dir'])
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Output is not empty: {output}; choose --output for a new run')
        output.mkdir(parents=True,exist_ok=True)
        if cfg['backend'] == 'sequential':
            from .server_app import run_experiment
            run_experiment(cfg)
        else:
            import tomli_w
            env = os.environ.copy()
            env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH','')
            env['FLWR_DISABLE_RUNTIME_DEPENDENCY_INSTALLATION'] = '1'
            with flower_app(cfg) as app:
                config_file = app/'run.toml'
                config_file.write_text(tomli_w.dumps(cfg))
                command = flower_command(args,cfg,app,config_file)
                print(shlex.join(command),flush=True)
                subprocess.run(command,check=True,env=env)
        if not (output/'results.json').is_file() or not (output/'checkpoint.pt').is_file():
            raise RuntimeError('Training ended without results and checkpoint; inspect the run log')
        result = json.loads((output/'results.json').read_text())
        if len(result.get('rounds',[])) != cfg['num_rounds']:
            raise RuntimeError('Training did not complete every requested round')
        print(f'Saved {output}')
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as exc:
        p.exit(2,f'Experiment failed: {exc}\n')


if __name__ == '__main__':
    main()
