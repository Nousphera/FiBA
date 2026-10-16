"""RankMe convergence gating and SmoothGradCAM++ pixel placement."""
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from .data import normalize


def rankme(embeddings):
    """Effective rank from the entropy of normalized singular values."""
    if embeddings.ndim != 2 or not torch.isfinite(embeddings).all():
        raise ValueError('RankMe requires a finite sample-by-feature matrix')
    values = torch.linalg.svdvals(embeddings.detach().double().cpu())
    if values.sum() == 0:
        return 0.
    probabilities = values / values.sum()
    positive = probabilities > 0
    return float(torch.exp(-(probabilities[positive] * probabilities[positive].log()).sum()))


def select_mask(heatmap, placement):
    """Return protected pixels: high half for low-attention placement, vice versa."""
    if heatmap.shape != (32,32) or not torch.isfinite(heatmap).all():
        raise ValueError('Expected a finite CIFAR attention map')
    if placement not in ('low','high'):
        raise ValueError('Attention placement must be low or high')
    indices = torch.argsort(heatmap.flatten(),stable=True)
    protected = torch.zeros(1024,dtype=torch.bool,device=heatmap.device)
    protected[indices[512:]] = True
    return (protected if placement == 'low' else ~protected).reshape(32,32)


def compute_attention(model,dataset,cfg):
    """Use only an attacker's clean training partition; return mask and RankMe.

    The round-20 schedule is applied by training.py. A positive optional RankMe
    threshold additionally gates activation; the paper does not specify tau.
    CAM sums use predicted labels and layer1, following the released source.
    """
    from torchcam.methods import SmoothGradCAMpp
    if not len(dataset):
        raise ValueError('Attention needs clean training examples')
    count = min(len(dataset),cfg['attention_samples'])
    chosen = torch.randperm(len(dataset),generator=torch.Generator().manual_seed(cfg['seed']))[:count].tolist()
    batches = DataLoader(Subset(dataset,chosen),batch_size=min(count,cfg['batch_size']),shuffle=False)
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    features = []
    handle = model.linear.register_forward_pre_hook(lambda _,args:features.append(args[0].detach().cpu()))
    try:
        with torch.no_grad():
            for images,_ in batches:
                model(normalize(images).to(device))
        rank = rankme(torch.cat(features))
    finally:
        handle.remove()
        model.train(was_training)
    if cfg['rankme_threshold'] > 0 and rank < cfg['rankme_threshold']:
        return None,rank
    total = torch.zeros(32,32,device=device)
    model.eval()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(cfg['seed'])
            with SmoothGradCAMpp(model,target_layer='layer1',num_samples=cfg['cam_samples'],input_shape=(3,32,32)) as cam:
                for images,_ in batches:
                    for image in images:
                        scores = model(normalize(image.unsqueeze(0)).to(device))
                        heat = cam(int(scores.argmax(1)),scores)[0].detach()
                        heat = F.interpolate(heat.reshape(1,1,*heat.shape[-2:]),size=(32,32),mode='bilinear',align_corners=False)
                        total += heat[0,0]
        return select_mask(total,cfg['attention']).cpu(),rank
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)
