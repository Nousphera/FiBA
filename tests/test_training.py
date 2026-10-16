"""Small fixtures exercise real ResNet training and sample-counted evaluation."""
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset


class Images(Dataset):
    classes = list(range(10))
    def __init__(self, labels):
        self.targets = np.asarray(labels)
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, index):
        return torch.full((3, 32, 32), .4 + index * .01), int(self.targets[index])


def settings(tmp_path):
    return dict(dataset='cifar10',data_root=str(tmp_path),num_clients=2,num_rounds=2,
                local_epochs=1,batch_size=3,lr=.001,participation=1.,split='iid',alpha=.5,
                seed=4,device='cpu',num_workers=0,threads=1,attack='none',attacker_fraction=0.,
                poison_fraction=.5,target_class=0,source_class=-1,attention='none',
                attention_start_round=20,rankme_threshold=0.,attention_samples=2,
                model_replacement=False,replacement_rounds='30,60,90')


def test_original_cifar_resnet_architecture_and_seed():
    from fiba.network import build_model
    first, second = build_model(100, 7), build_model(100, 7)
    assert first.conv1.kernel_size == (3, 3)
    assert [len(first.layer1),len(first.layer2),len(first.layer3),len(first.layer4)] == [2,2,2,2]
    assert first.linear.out_features == 100
    assert sum(p.numel() for p in first.parameters()) == 11046308
    assert all(torch.equal(value, second.state_dict()[key]) for key,value in first.state_dict().items())


def test_real_training_returns_partition_sample_count_and_finite_state(tmp_path, monkeypatch):
    from fiba import data, training
    from fiba.network import build_model
    monkeypatch.setattr(data, 'load_datasets', lambda *args: (Images(range(8)),Images([0,1,2,3,4])))
    cfg = settings(tmp_path)
    state = build_model(10,cfg['seed']).state_dict()
    result, metrics = training.train_client(state,cfg,0,1)
    assert metrics['num_examples'] == 4
    assert metrics['poisoned_examples'] == 0
    assert metrics['loss'] > 0
    assert all(torch.isfinite(value).all() for value in result.values())
    assert any(not torch.equal(state[key],value) for key,value in result.items())
    assert result['bn1.num_batches_tracked'].dtype == torch.int64


def test_evaluation_counts_non_target_examples_and_sample_weighted_loss(tmp_path, monkeypatch):
    from fiba import data, training
    from fiba.network import build_model
    monkeypatch.setattr(data, 'load_datasets', lambda *args: (Images(range(8)),Images([0,0,0,1,2])))
    cfg = settings(tmp_path) | {'attack':'badnet','attacker_fraction':.5}
    state = {key:torch.zeros_like(value) for key,value in build_model(10).state_dict().items()}
    result = training.evaluate(state,cfg)
    assert result['clean_examples'] == 5
    assert result['asr_examples'] == 2
    assert result['accuracy'] == pytest.approx(.6)
    assert result['asr'] == 1.
    assert result['clean_loss'] == pytest.approx(np.log(10),rel=1e-6)


def test_training_rejects_nonfinite_or_incomplete_model_state(tmp_path, monkeypatch):
    from fiba import data, training
    from fiba.network import build_model
    monkeypatch.setattr(data,'load_datasets',lambda *args:(Images(range(8)),Images([0,1])))
    cfg = settings(tmp_path)
    state = build_model(10).state_dict()
    state['conv1.weight'][0,0,0,0] = float('nan')
    with pytest.raises(ValueError,match='finite'):
        training.train_client(state,cfg,0,1)


def test_evaluation_rejects_negative_batchnorm_variance(tmp_path):
    from fiba import training
    from fiba.network import build_model
    state = build_model(10).state_dict()
    state['bn1.running_var'].fill_(-1)
    with pytest.raises(ValueError,match='variance'):
        training.load_model(state,settings(tmp_path))


def test_attention_encoder_uses_its_weighted_pretraining_mask(tmp_path,monkeypatch):
    from fiba import data, training, attention
    from fiba.generators import Encoder, save_generator, shared_trigger
    from fiba.network import build_model
    monkeypatch.setattr(data,'load_datasets',lambda *args:(Images(range(8)),Images([0,1])))
    def unexpected_cam(*args):
        pytest.fail('Ungated Encoder must reuse its trained mask without recomputing CAM')
    monkeypatch.setattr(attention,'compute_attention',unexpected_cam)
    mask = torch.zeros(32,32,dtype=torch.bool)
    mask[:16] = True
    checkpoint = tmp_path/'encoder.pt'
    save_generator(checkpoint,'encoder',Encoder(),
                   {'attention_placement':'low','attention_weighted_training':True},
                   shared_trigger(),attention_mask=mask)
    cfg = settings(tmp_path) | {'attack':'encoder','attacker_fraction':.5,'attention':'low',
                               'attention_start_round':1,'generator_checkpoint':str(checkpoint)}
    _,metrics = training.train_client(build_model(10).state_dict(),cfg,0,1)
    assert metrics['attack_active'] == 1
    assert torch.equal(metrics['_attention_mask'],mask)
