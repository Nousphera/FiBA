"""Small independent fixtures for transport, aggregation and trigger semantics."""
import numpy as np
import pytest
import flwr as fl
from fiba.aggregation_strategies import weighted_averaging, median, trimmed_mean
from fiba.fools_gold import AggregationStrategyFoolsGold
from fiba.plain_steganography import hide_trigger_in_image, hide_trigger_in_image_modified


def response(cid, value, gradient):
    return fl.common.FitRes(fl.common.Status(fl.common.Code.OK, ''),
        fl.common.ndarrays_to_parameters([np.array([value], dtype=np.float32)]), 4,
        {'client_id': cid, 'gradient': fl.common.ndarray_to_bytes(np.array(gradient, dtype=np.float32))})


def test_foolsgold_accepts_standard_metrics_and_keeps_partial_client_identity():
    strategy = AggregationStrategyFoolsGold()
    first, _ = strategy.aggregate_fit(1, [(None, response(5, 4, [0, 1])), (None, response(1, 2, [1, 0]))], [])
    np.testing.assert_allclose(fl.common.parameters_to_ndarrays(first)[0], [3])
    second, _ = strategy.aggregate_fit(2, [(None, response(5, 8, [0, 1]))], [])
    np.testing.assert_allclose(fl.common.parameters_to_ndarrays(second)[0], [8])
    np.testing.assert_allclose(strategy.memory[1], [1, 0])
    np.testing.assert_allclose(strategy.memory[5], [0, 2])


def test_weighted_and_robust_aggregation_have_distinct_semantics():
    updates = [([np.array([v], float)], n) for v, n in [(0, 1), (2, 3), (4, 1), (6, 1), (100, 1)]]
    np.testing.assert_allclose(weighted_averaging(updates)[0], [116/7])
    np.testing.assert_allclose(median(updates)[0], [4])
    np.testing.assert_allclose(trimmed_mean(updates, .2)[0], [4])


def test_lsb_embeds_ascii_bits_and_preserves_other_pixels():
    image = np.full((2, 2, 3), 128, dtype=np.uint8)
    actual = hide_trigger_in_image(image.copy(), 'A').reshape(-1)
    np.testing.assert_array_equal(actual, [128,129,128,128,128,128,128,129,128,128,128,128])


def test_lsb_rejects_trigger_larger_than_capacity():
    with pytest.raises(ValueError, match='capacity'):
        hide_trigger_in_image(np.zeros((1, 1, 3), np.uint8), 'toolong')


def test_attention_mask_only_changes_selected_pixels():
    image = np.full((2, 2, 3), 128, dtype=np.uint8)
    mask = np.array([[0,1],[1,1]])
    actual = hide_trigger_in_image_modified(image.copy(), 'A', mask, 0)
    np.testing.assert_array_equal(actual[mask == 1], image[mask == 1])
    assert np.any(actual[mask == 0] != image[mask == 0])


def test_krum_selects_central_update_and_multikrum_averages_selected_updates():
    from fiba.krum import aggregate_krum
    updates = [([np.array([value],float)],1) for value in [0,1,2,3,100]]
    np.testing.assert_allclose(aggregate_krum(updates,1,0)[0],[1])
    np.testing.assert_allclose(aggregate_krum(updates,1,2)[0],[1.5])


def test_attention_mask_selects_half_of_image(tmp_path, monkeypatch):
    import torch
    from fiba.network import Model
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = Model(2)
    images = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.rand(1,3,32,32),torch.tensor([0])),batch_size=1)
    mask = model.get_activation_map(images)
    assert tuple(mask.shape) == (32,32)
    assert set(mask.unique().tolist()) == {0.,1.}
    assert int(mask.sum()) == 512
