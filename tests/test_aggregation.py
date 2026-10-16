import pytest
import torch

from fiba.aggregation import Aggregator


def state(value, counter=0):
    return {'fc.weight': torch.tensor([[value, 0.]], dtype=torch.float32),
            'num_batches_tracked': torch.tensor(counter)}


def test_fedavg_weighted_and_integral_max():
    out = Aggregator({}).aggregate([state(1, 1), state(5, 3)], [1, 3], state(0), [0, 1])
    assert out['fc.weight'][0, 0] == 4
    assert out['num_batches_tracked'] == 3
    assert out['num_batches_tracked'].dtype == torch.int64


@pytest.mark.parametrize('method', ['median', 'trimmed-mean'])
def test_coordinate_defenses_reject_outlier(method):
    states = [state(x) for x in [0., 1., 2., 3., 100.]]
    result = Aggregator({'aggregation': method, 'trim_fraction': .2}).aggregate(
        states, [1] * 5, state(0), list(range(5)))
    assert result['fc.weight'][0, 0] == 2


def test_krum_and_multikrum_choose_low_scores_and_validate_population():
    states = [state(x) for x in [0., 1., 2., 3., 100.]]
    for method, expected in [('krum', 1.), ('multi-krum', 1.5)]:
        result = Aggregator({'aggregation': method, 'byzantine_clients': 1, 'krum_keep': 2}).aggregate(
            states, [1] * 5, state(0), list(range(5)))
        assert result['fc.weight'][0, 0] == expected
    with pytest.raises(ValueError, match='2.f.3'):
        Aggregator({'aggregation': 'krum', 'byzantine_clients': 1}).aggregate(
            states[:4], [1] * 4, state(0), list(range(4)))


def test_foolsgold_downweights_identical_sybil_updates_and_tracks_identity():
    defense = Aggregator({'aggregation': 'foolsgold'})
    honest = {'fc.weight': torch.tensor([[0., 1.]]), 'num_batches_tracked': torch.tensor(1)}
    out = defense.aggregate([state(10), state(10), honest], [1, 1, 1], state(0), [8, 3, 9])
    assert out['fc.weight'][0, 0] == 0
    assert out['fc.weight'][0, 1] == 1
    defense.aggregate([honest], [1], state(0), [9])
    assert set(defense.history) == {8, 3, 9}
    assert torch.equal(defense.history[9], torch.tensor([0., 2.], dtype=torch.float64))


def test_schema_nonfinite_and_empty_rejected():
    defense = Aggregator({})
    with pytest.raises(ValueError):
        defense.aggregate([], [], state(0), [])
    with pytest.raises(ValueError):
        defense.aggregate([state(float('nan'))], [1], state(0), [0])


def test_model_replacement_scales_delta_on_selected_rounds_only():
    config = {'model_replacement': True, 'replacement_rounds': '3,6',
              'num_clients': 2, 'attacker_fraction': .5}
    reference = state(2)
    clients = [state(3), state(2)]
    defense = Aggregator(config)
    assert defense.aggregate(clients, [1, 3], reference, [0, 1], round_number=2)['fc.weight'][0, 0] == 2.25
    assert defense.aggregate(clients, [1, 3], reference, [0, 1], round_number=3)['fc.weight'][0, 0] == 3


def test_foolsgold_supports_original_resnet_linear_classifier():
    reference = {'linear.weight': torch.zeros(2, 2)}
    state = {'linear.weight': torch.eye(2)}
    output = Aggregator({'aggregation': 'foolsgold'}).aggregate([state], [1], reference, [0])
    assert torch.equal(output['linear.weight'], state['linear.weight'])


def test_replacement_preserves_batchnorm_running_variance():
    reference = {'linear.weight': torch.zeros(2, 2), 'bn.running_var': torch.ones(2)}
    malicious = {'linear.weight': torch.ones(2, 2), 'bn.running_var': torch.full((2,), .1)}
    result = Aggregator({'model_replacement': True, 'replacement_rounds': '1',
                         'num_clients': 2, 'attacker_fraction': .5}).aggregate(
        [malicious, reference], [1, 9], reference, [0, 1], round_number=1)
    assert torch.all(result['bn.running_var'] > 0)
    torch.testing.assert_close(result['bn.running_var'], torch.full((2,), .91))


def test_aggregation_rejects_invalid_batchnorm_variance():
    reference = {'linear.weight': torch.zeros(2, 2), 'bn.running_var': torch.ones(2)}
    bad = {'linear.weight': torch.zeros(2, 2), 'bn.running_var': torch.full((2,), -1.)}
    with pytest.raises(ValueError, match='variance'):
        Aggregator({}).aggregate([bad], [1], reference, [0])
