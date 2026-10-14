"""Catch the original patched-Flower return shape with a real local update."""
from types import SimpleNamespace
import numpy as np
import torch
import flwr as fl
from flwr.common import serde
from fiba.client import Client
from fiba.network import Model


def test_real_client_fit_roundtrips_stock_flower(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(7)
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.chdir(work)
    dataset = torch.utils.data.TensorDataset(torch.rand(4, 3, 32, 32), torch.tensor([0, 1, 0, 1]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    attack = SimpleNamespace(attackers=[], target_class=1, targeted=False, boost_strat=None)
    client = Client(0, 1, attack, Model, lambda **kw: (loader, 2, 4), device='cpu', run_id='test')
    initial = [value.numpy() for value in Model(2).state_dict().values()]
    instruction = fl.common.FitIns(fl.common.ndarrays_to_parameters(initial), {'round': 1, 'epochs': 1, 'lr': 0.001})
    response = client.to_client().fit(instruction)
    restored = serde.fit_res_from_proto(serde.fit_res_to_proto(response))
    assert restored.num_examples == 4
    assert restored.metrics['client_id'] == 0
    assert isinstance(restored.metrics['gradient'], bytes)
    assert np.isfinite(restored.metrics['loss'])
    after = fl.common.parameters_to_ndarrays(restored.parameters)
    assert any(not np.array_equal(before, current) for before, current in zip(initial, after))
