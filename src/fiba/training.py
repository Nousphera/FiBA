"""Shared PyTorch training and held-out evaluation for both execution backends."""
import math
import random
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from . import data
from .network import build_model


def seed_everything(seed, threads=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(cfg):
    name = cfg['device']
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if name == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS requested but unavailable')
    return torch.device(name)


def attacker_ids(cfg):
    return list(range(math.floor(cfg['num_clients'] * cfg['attacker_fraction'])))


def load_model(state, cfg):
    model = build_model(100 if cfg['dataset'] == 'cifar100' else 10, cfg['seed'])
    reference = model.state_dict()
    if set(state) != set(reference):
        raise ValueError('Model tensor names do not match ResNet18')
    for key, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != reference[key].shape
                or value.dtype != reference[key].dtype):
            raise ValueError(f'Invalid model tensor schema: {key}')
        if not torch.isfinite(value).all():
            raise ValueError(f'Model tensors must be finite: {key}')
        if key.endswith('running_var') and torch.any(value < 0):
            raise ValueError(f'BatchNorm variance must be nonnegative: {key}')
    model.load_state_dict(state, strict=True)
    return model.to(device_for(cfg))


def partitions(train, cfg):
    return data.partition_indices(train.targets, cfg['num_clients'], cfg['split'], cfg['alpha'], cfg['seed'])


def loader(dataset, cfg, *, shuffle=False, seed=None):
    generator = torch.Generator().manual_seed(cfg['seed'] if seed is None else seed)
    return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=shuffle,
                      num_workers=cfg['num_workers'], generator=generator,
                      pin_memory=cfg['device'].startswith('cuda'))


class IndexedDataset(torch.utils.data.Dataset):
    """Carry stable local membership through DataLoader shuffling/workers."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        image, label = self.dataset[index]
        return image, label, index


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_client(state, cfg, cid, round_number, mask=None):
    if type(cid) != int or not 0 <= cid < cfg['num_clients']:
        raise ValueError('Invalid client partition')
    if type(round_number) != int or not 1 <= round_number <= cfg['num_rounds']:
        raise ValueError('Invalid training round')
    local_seed = cfg['seed'] + round_number * cfg['num_clients'] + cid
    seed_everything(local_seed, cfg['threads'])
    model = load_model(state, cfg)
    train, _ = data.load_datasets(cfg['dataset'], cfg['data_root'])
    local = Subset(train, partitions(train, cfg)[cid])
    malicious = cid in attacker_ids(cfg) and cfg['attack'] != 'none'
    from .attacks import build_attack
    attack = build_attack(cfg) if malicious else None
    rank = None
    if cfg['attention'] != 'none':
        mask = None
        if malicious and round_number >= cfg['attention_start_round']:
            if cfg['attack'] == 'encoder':
                # Weighted generator training and federated poisoning must use
                # the same saved protected region, rather than a new CAM mask.
                saved = attack.checkpoint.get('attention_mask')
                if saved is None:
                    raise ValueError('Attention Encoder requires its trained protected mask')
                mask = saved.detach().cpu().bool()
                if cfg['rankme_threshold'] > 0:
                    from .attention import compute_attention
                    _, rank = compute_attention(model, local, cfg)
                    if rank < cfg['rankme_threshold']:
                        mask = None
            else:
                from .attention import compute_attention
                mask, rank = compute_attention(model, local, cfg)
    attack_active = malicious and (cfg['attention'] == 'none' or mask is not None)
    # Recreate Adam each local round, matching the released experiment.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    generator = torch.Generator().manual_seed(local_seed)
    selected = torch.zeros(len(local), dtype=torch.bool)
    if attack_active:
        labels = np.asarray(train.targets)[np.asarray(local.indices)]
        eligible = np.flatnonzero(labels != cfg['target_class'])
        if cfg['source_class'] >= 0:
            eligible = eligible[labels[eligible] == cfg['source_class']]
        amount = min(math.floor(len(local) * cfg['poison_fraction']), len(eligible))
        membership_rng = torch.Generator().manual_seed(cfg['seed'] + cid)
        chosen = torch.randperm(len(eligible), generator=membership_rng)[:amount].numpy()
        selected[torch.as_tensor(eligible[chosen])] = True
    batches = loader(IndexedDataset(local), cfg, shuffle=True, seed=local_seed)
    model.train()
    loss_sum, correct, examples = 0., 0, 0
    for _ in range(cfg['local_epochs']):
        for images, labels, indices in batches:
            images = data.augment_raw(images, generator=generator)
            poison = selected[indices]
            if poison.any():
                images[poison] = attack.apply(images[poison], client_id=cid, mask=mask)
                labels[poison] = cfg['target_class']
            images = data.normalize(images).to(model.conv1.weight.device)
            labels = labels.long().to(images.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError('Local training produced nonfinite loss')
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += int((logits.argmax(1) == labels).sum())
            examples += len(labels)
    metrics = {'client_id':cid, 'num_examples':len(local), 'loss':loss_sum/examples,
               'accuracy':correct/examples, 'poisoned_examples':int(selected.sum()),
               'attack_active':int(attack_active)}
    if rank is not None:
        metrics['rankme'] = float(rank)
    if mask is not None and malicious:
        metrics['_attention_mask'] = mask.detach().cpu()
    result = cpu_state(model)
    del optimizer, model
    return result, metrics


def evaluate(state, cfg, mask=None):
    seed_everything(cfg['seed'], cfg['threads'])
    model = load_model(state, cfg)
    _, test = data.load_datasets(cfg['dataset'], cfg['data_root'])
    from .attacks import build_attack
    enabled = cfg['attack'] != 'none' and (cfg['attention'] == 'none' or mask is not None)
    attack = build_attack(cfg) if enabled else None
    model.eval()
    total, correct, loss_sum, asr_count, successes = 0, 0, 0., 0, 0
    classes = 100 if cfg['dataset'] == 'cifar100' else 10
    class_count, class_correct = np.zeros(classes,dtype=int), np.zeros(classes,dtype=int)
    with torch.inference_mode():
        for raw, labels in loader(test,cfg):
            labels = labels.long().to(model.conv1.weight.device)
            logits = model(data.normalize(raw).to(labels.device))
            if not torch.isfinite(logits).all():
                raise ValueError('Clean evaluation produced nonfinite logits')
            predictions = logits.argmax(1)
            total += len(labels)
            correct += int((predictions == labels).sum())
            loss_sum += F.cross_entropy(logits,labels,reduction='sum').item()
            class_count += np.bincount(labels.cpu().numpy(),minlength=classes)
            class_correct += np.bincount(labels[predictions == labels].cpu().numpy(),minlength=classes)
            if enabled:
                eligible = labels.cpu() != cfg['target_class']
                if cfg['source_class'] >= 0:
                    eligible &= labels.cpu() == cfg['source_class']
                if eligible.any():
                    poisoned = attack.apply(raw[eligible],client_id=None,mask=mask)
                    attacked_logits = model(data.normalize(poisoned).to(labels.device))
                    if not torch.isfinite(attacked_logits).all():
                        raise ValueError('Triggered evaluation produced nonfinite logits')
                    predictions = attacked_logits.argmax(1)
                    asr_count += int(eligible.sum())
                    successes += int((predictions == cfg['target_class']).sum())
    return {'clean_loss':loss_sum/total,'accuracy':correct/total,
            'asr':successes/asr_count if asr_count else 0.,'clean_examples':total,
            'asr_examples':asr_count,'asr_successes':successes,
            'class_accuracy':[float(got/count) if count else None for got,count in zip(class_correct,class_count)]}
