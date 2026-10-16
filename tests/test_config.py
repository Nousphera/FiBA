import pytest
from fiba.config import defaults, validate


def test_paper_defaults():
    cfg = validate(defaults())
    assert (cfg['num_clients'], cfg['num_rounds'], cfg['local_epochs'], cfg['batch_size']) == (10,100,5,64)
    assert cfg['split'] == 'dirichlet' and cfg['alpha'] == .5
    assert cfg['trigger_characters'] == 700


@pytest.mark.parametrize('updates', [
    {'participation': 0}, {'poison_fraction': 1.1}, {'lr': float('nan')},
    {'target_class':10}, {'attack':'lira','attention':'low'},
    {'num_clients':3, 'aggregation':'krum'}, {'split':'dirichlet','alpha':0},
    {'num_clients':2,'attacker_fraction':.1}, {'device':'fake'},
])
def test_invalid_configuration_fails_early(updates):
    cfg=defaults(); cfg.update(updates)
    with pytest.raises(ValueError): validate(cfg)


def test_clean_control_does_not_require_attackers():
    cfg=defaults(); cfg.update(attack='none',num_clients=2,attacker_fraction=0)
    assert validate(cfg)['attack'] == 'none'


def test_flower_cuda_indices_are_not_worker_local_device_indices():
    cfg=defaults(); cfg.update(device='cuda:1')
    with pytest.raises(ValueError,match='CUDA_VISIBLE_DEVICES'):
        validate(cfg)
    cfg['backend']='sequential'
    assert validate(cfg)['device']=='cuda:1'
