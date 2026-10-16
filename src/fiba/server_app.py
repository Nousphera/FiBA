"""Federated FiBA coordination with native Flower or an equivalent local loop."""
from importlib.metadata import version
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
import torch
from flwr.app import ArrayRecord, ConfigRecord, Message, RecordDict
from flwr.serverapp import ServerApp
from .network import build_model
from .training import attacker_ids, cpu_state, evaluate, partitions, seed_everything, train_client
from .transport import identify, replies, wait_for_nodes

app = ServerApp()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024 * 1024),b''):
            digest.update(block)
    return digest.hexdigest()


def provenance(cfg, train):
    """Bind this run to the exact prepared corpus and frozen trigger artifact."""
    evidence = {}
    if hasattr(train,'directory'):
        path = Path(train.directory)/'manifest.json'
        evidence['dataset'] = {'manifest_sha256':file_sha256(path),
                               'manifest':json.loads(path.read_text())}
    if cfg['attack'] in ('encoder','lira'):
        from .generators import load_generator
        path = Path(cfg['generator_checkpoint'])
        _, checkpoint = load_generator(path,cfg['attack'])
        evidence['generator'] = {'sha256':file_sha256(path),'kind':checkpoint['kind'],
                                 'format_version':checkpoint['format_version'],
                                 'config':checkpoint.get('config',{}),
                                 'training_examples':checkpoint.get('training_examples'),
                                 'attention_policy':'saved protected mask from weighted pretraining'
                                     if cfg['attention'] != 'none' else 'none'}
    return evidence


def run_experiment(cfg, grid=None):
    """Use identical client training, aggregation, and evaluation in both backends."""
    from . import data
    from .aggregation import Aggregator
    from .config import validate
    cfg = validate(dict(cfg))
    output = Path(cfg['output_dir'])
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output directory is not empty; choose a new output directory')
    output.mkdir(parents=True,exist_ok=True)
    seed_everything(cfg['seed'],cfg['threads'])
    train, _ = data.load_datasets(cfg['dataset'],cfg['data_root'])
    ownership = partitions(train,cfg)
    counts = [len(indices) for indices in ownership]
    state = cpu_state(build_model(100 if cfg['dataset'] == 'cifar100' else 10,cfg['seed']))
    aggregator = Aggregator(cfg)
    rng = random.Random(cfg['seed'])
    mapping = None
    if grid is not None:
        nodes = wait_for_nodes(grid,cfg['num_clients'],cfg.get('registration_timeout',120))
        mapping = identify(grid,nodes,cfg['num_clients'],cfg.get('reply_timeout',3600))
    report = {'schema':1,'config':cfg,'backend':'flower-message' if grid is not None else 'sequential',
              'model':'CIFAR ResNet18 option-A','attackers':attacker_ids(cfg),
              'provenance':provenance(cfg,train),
              'attention_policy':('saved encoder pretraining mask' if cfg['attack'] == 'encoder'
                                  else 'lowest participating attacker clean-partition CAM'),
              'partition_counts':counts,'partitions':[indices.tolist() for indices in ownership],
              'runtime':{'python':platform.python_version(),'platform':platform.platform(),
                         'torch':str(torch.__version__),'flower':version('flwr'),
                         'torchvision':version('torchvision'),'numpy':version('numpy'),
                         'torchcam':version('torchcam')},'rounds':[]}
    attention_mask, attention_cid = None, None
    for rnd in range(1,cfg['num_rounds']+1):
        started = time.monotonic()
        selected = sorted(rng.sample(range(cfg['num_clients']),max(1,math.ceil(cfg['num_clients']*cfg['participation']))))
        received = {}
        if grid is None:
            for cid in selected:
                result, metrics = train_client(state,cfg,cid,rnd)
                mask = metrics.pop('_attention_mask',None)
                received[cid] = (result,metrics,mask)
        else:
            messages = []
            for cid in selected:
                content = RecordDict({'model':ArrayRecord(state),
                                      'config':ConfigRecord({'round':rnd,'phase':'train'})})
                messages.append(Message(content,dst_node_id=mapping[cid],message_type='train',group_id=str(rnd)))
            for reply in replies(grid,messages,cfg.get('reply_timeout',3600)):
                metrics = dict(reply.content['metrics'])
                cid = metrics['client_id']
                if type(cid) != int or cid not in selected or mapping[cid] != reply.metadata.src_node_id or cid in received:
                    raise ValueError('Client identity does not match selected partition')
                mask = reply.content['attention'].to_torch_state_dict()['mask'] if 'attention' in reply.content else None
                received[cid] = (reply.content['model'].to_torch_state_dict(),metrics,mask)
        states, weights, client_metrics = [], [], []
        masks = []
        for cid in selected:
            update, metrics, mask = received[cid]
            if (type(metrics.get('num_examples')) != int or metrics['num_examples'] != counts[cid]
                    or not all(type(value) in (int,float) and math.isfinite(value) for value in metrics.values())):
                raise ValueError('Invalid client metrics or partition sample count')
            states.append(update)
            weights.append(counts[cid])
            client_metrics.append(metrics)
            if mask is not None:
                if (cid not in attacker_ids(cfg) or tuple(mask.shape) != (32,32)
                        or not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all()):
                    raise ValueError('Invalid attention mask')
                masks.append((cid,mask))
        if masks:
            attention_cid,attention_mask = min(masks,key=lambda pair:pair[0])
        before = state
        state = aggregator.aggregate(states,weights,state,selected,round_number=rnd)
        evaluation = evaluate(state,cfg,attention_mask)
        row = {'round':rnd,'clients':client_metrics,'examples':sum(weights),
               'parameters_changed':any(not torch.equal(before[key],value) for key,value in state.items()),
               'attention_client_id':attention_cid,'aggregation_weights':aggregator.last_weights,
               'seconds':time.monotonic()-started,**evaluation}
        report['rounds'].append(row)
        checkpoint = {'model':state,'config':cfg,'round':rnd,'attention_mask':attention_mask,
                      'attention_client_id':attention_cid,'provenance':report['provenance']}
        checkpoint_tmp = output/'checkpoint.pt.tmp'
        torch.save(checkpoint,checkpoint_tmp)
        checkpoint_tmp.replace(output/'checkpoint.pt')
        result_tmp = output/'results.json.tmp'
        result_tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        result_tmp.replace(output/'results.json')
        print(json.dumps({'round':rnd,**evaluation}),flush=True)
    return report


@app.main()
def main(grid, context):
    run_experiment(dict(context.run_config),grid=grid)
