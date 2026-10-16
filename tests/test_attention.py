import torch
import pytest
from torch.utils.data import TensorDataset
from fiba.attention import rankme, select_mask, compute_attention
from fiba.config import defaults
from fiba.network import build_model


def test_rankme_known_spectrum():
    assert rankme(torch.eye(4)) == pytest.approx(4)
    assert rankme(torch.ones(3,4)) == pytest.approx(1, abs=1e-5)
    assert rankme(torch.zeros(3,4)) == 0


def test_high_and_low_protect_opposite_halves_even_with_ties():
    heat=torch.arange(1024).reshape(32,32).float()
    low=select_mask(heat,'low'); high=select_mask(heat,'high')
    assert low.sum()==high.sum()==512
    assert torch.equal(low,~high)
    assert low[31,31] and not low[0,0]
    assert select_mask(torch.zeros(32,32),'low').sum()==512


def test_cam_uses_clean_images_restores_hooks_and_preserves_model():
    torch.set_num_threads(1)
    model=build_model(10)
    before={key:value.clone() for key,value in model.state_dict().items()}
    cfg=defaults(); cfg.update(attention='low',attention_samples=2,cam_samples=2,device='cpu')
    ds=TensorDataset(torch.rand(3,3,32,32),torch.tensor([0,1,2]))
    mask,rank=compute_attention(model,ds,cfg)
    assert mask.shape==(32,32) and mask.sum()==512 and rank>0
    assert model.training
    assert all(torch.equal(value,before[key]) for key,value in model.state_dict().items())
    assert not model.layer1._forward_hooks and not model.linear._forward_pre_hooks


def test_rank_gate_skips_cam():
    model=build_model(10)
    cfg=defaults(); cfg.update(attention='low',attention_samples=2,rankme_threshold=10.)
    ds=TensorDataset(torch.rand(2,3,32,32),torch.tensor([0,1]))
    mask,rank=compute_attention(model,ds,cfg)
    assert mask is None and rank<10
