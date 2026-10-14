"""Run the recovered FiBA benchmark with stock Flower message serialization.

The sequential backend executes real client training and server aggregation on
one machine. It does not measure network latency or distributed throughput.
"""
import argparse
import json
import os
import platform
from pathlib import Path
import random
import uuid


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['cifar10','cifar100','synthetic'], default='cifar10')
    p.add_argument('--data-root', type=Path, default=Path('datasets'))
    p.add_argument('--download', action='store_true', help='Download public CIFAR data if missing')
    p.add_argument('--output', type=Path, default=Path('runs/fiba'))
    p.add_argument('--num-clients', type=int, default=10)
    p.add_argument('--num-rounds', type=int, default=100)
    p.add_argument('--local-epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--participation', type=float, default=1.)
    p.add_argument('--data-split', choices=['iid','noniid'], default='iid')
    p.add_argument('--skewness', type=float, default=.5)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--attack', choices=['steg','wanet','badnet','encoder','lira'], default='steg')
    p.add_argument('--attacker-fraction', type=float, default=.1)
    p.add_argument('--source-class', type=int, default=2)
    p.add_argument('--target-class', type=int, default=5)
    p.add_argument('--avg-strategy', choices=['avg','median','trimmed','fg','krum'], default='avg')
    p.add_argument('--krum-clients-to-keep', type=int, default=0, help='0 for Krum; positive count for Multi-Krum')
    p.add_argument('--distributed', action='store_true')
    p.add_argument('--distribution', type=int, default=1)
    p.add_argument('--boost-strat', choices=['model_replacement'])
    p.add_argument('--modified-attack', action='store_true', help='Original round-20 low-attention LSB variant')
    p.add_argument('--mask-inversion', action='store_true', help='Use high-attention pixels instead')
    p.add_argument('--trigger-characters', type=int, choices=[500,700], default=500)
    p.add_argument('--encoder-checkpoint', type=Path)
    p.add_argument('--lira-generator', type=Path, help='Externally exported TorchScript noise generator')
    return p


def build_attack(args, attackers):
    from .utils import split_image
    dimensions = split_image(32,32,args.distribution)
    common = (attackers,dimensions,args.source_class,args.target_class,args.distributed,args.boost_strat)
    if args.attack == 'steg':
        from .plain_steganography import SteganographyAttack
        attack = SteganographyAttack(*common,args.modified_attack,args.mask_inversion)
        attack.trigger_characters = args.trigger_characters
        return attack
    if args.attack == 'wanet':
        from .wanet import WaNetAttack
        return WaNetAttack(*common,attackers,dimensions)
    if args.attack == 'badnet':
        from .badnet import BadNetAttack
        return BadNetAttack(*common,False,False)
    if args.attack == 'encoder':
        from .encoder_poisoning import EncoderAttack
        return EncoderAttack(*common,False,False,checkpoint=args.encoder_checkpoint)
    from .lira_poisoning import LiraAttack
    return LiraAttack(*common,generator_path=args.lira_generator)


def run(args):
    import numpy as np
    import torch
    import flwr as fl
    from flwr.common import serde
    from .client import Client
    from .data import get_data
    from .network import Model
    from .server import Server
    from .utils import select_fraction_of_numbers

    for name in ['num_clients','num_rounds','local_epochs','batch_size','threads','distribution']:
        if getattr(args,name) <= 0:
            raise ValueError(f'{name} must be positive')
    if not 0 < args.participation <= 1 or not 0 <= args.attacker_fraction <= 1:
        raise ValueError('Participation must be in (0,1] and attacker fraction in [0,1]')
    selected_count = max(1,int(args.num_clients * args.participation))
    classes = 100 if args.dataset == 'cifar100' else (2 if args.dataset == 'synthetic' else 10)
    if not 0 <= args.target_class < classes:
        raise ValueError(f'Target class must be in [0,{classes-1}] for {args.dataset}')
    if args.modified_attack and args.attack != 'steg':
        raise ValueError('Attention placement is supported only for the paper LSB experiment')
    if args.mask_inversion and not args.modified_attack:
        raise ValueError('--mask-inversion requires --modified-attack')
    if args.distributed and args.attack not in ['steg','wanet']:
        raise ValueError('Distributed triggering is supported for LSB and WaNet only')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory is not empty; choose a new run directory')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable; use --device cpu')
    attackers = select_fraction_of_numbers(args.num_clients,args.attacker_fraction,args.seed)
    if args.distributed and (len(attackers) != args.distribution or not attackers):
        raise ValueError('Distributed trigger requires one attacker per segment')
    if args.boost_strat and len(attackers) != 1:
        raise ValueError('Recovered model-replacement formula supports exactly one attacker')
    if args.avg_strategy == 'krum' and selected_count < 2 * len(attackers) + 3:
        raise ValueError('Krum requires at least 2*f+3 participating clients')
    output = args.output.resolve()
    data_root = args.data_root.resolve()
    for attr in ['encoder_checkpoint','lira_generator']:
        if getattr(args,attr) is not None:
            setattr(args,attr,getattr(args,attr).resolve())
    # Runtime cache files in the recovered code use ../ paths. Confine every
    # invocation to its own output/work directory; never use a source checkout.
    work = output/'work'
    work.mkdir(parents=True)
    os.environ['TORCH_DATA_DIR'] = str(data_root)
    os.environ['FIBA_DOWNLOAD'] = '1' if args.download else '0'
    previous = Path.cwd()
    try:
        os.chdir(work)
        attack = build_attack(args,attackers)
        run_id = uuid.uuid4().hex
        server = Server(Model,get_data,args.num_rounds,attack,run_id,num_clients=args.num_clients,
            participation=args.participation,batch_size=args.batch_size,data_name=args.dataset,
            num_epochs=args.local_epochs,lr=args.lr,device=args.device,strategy=args.avg_strategy,
            krum_clients_to_keep=args.krum_clients_to_keep)
        parameters = server.strategy.initial_parameters
        rng = random.Random(args.seed)
        report = {'dataset':args.dataset,'backend':'sequential-stock-flower','paper_reproduction':False,
            'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'attackers':attackers,'runtime':{'python':platform.python_version(), 'platform':platform.platform(),
            'torch':torch.__version__, 'flower':fl.__version__, 'numpy':np.__version__, 'device':args.device},'rounds':[]}
        for rnd in range(1,args.num_rounds+1):
            results = []
            before = fl.common.parameters_to_ndarrays(parameters)
            for cid in sorted(rng.sample(range(args.num_clients),selected_count)):
                torch.manual_seed(args.seed + rnd * args.num_clients + cid)
                client = Client(cid,args.num_clients,attack,Model,get_data,data_split=args.data_split,
                    skewness=args.skewness,batch_size=args.batch_size,data_name=args.dataset,
                    seed=args.seed,device=args.device,run_id=run_id,m=args.num_clients)
                instruction = fl.common.FitIns(parameters,server.get_client_config_fn()(rnd))
                instruction = serde.fit_ins_from_proto(serde.fit_ins_to_proto(instruction))
                response = client.to_client().fit(instruction)
                response = serde.fit_res_from_proto(serde.fit_res_to_proto(response))
                results.append((None,response))
            parameters,_ = server.strategy.aggregate_fit(rnd,results,[])
            loss,metrics = server.strategy.evaluate(rnd,parameters)
            after = fl.common.parameters_to_ndarrays(parameters)
            entry = {'round':rnd,'trained_clients':len(results),'examples':sum(r.num_examples for _,r in results),
                'parameters_changed':any(not np.array_equal(a,b) for a,b in zip(before,after)),
                'loss':float(loss),'accuracy':float(metrics['accuracy']),'asr':float(metrics['asr'])}
            report['rounds'].append(entry)
            (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps(entry),flush=True)
        np.savez_compressed(output/'final-parameters.npz',*fl.common.parameters_to_ndarrays(parameters))
        return report
    finally:
        os.chdir(previous)


def cli():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (ValueError,FileNotFoundError) as error:
        p.error(str(error))
