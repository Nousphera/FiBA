import json
import subprocess
import sys
from types import SimpleNamespace
import torch
from fiba import data


def test_partition_sizes_are_examples_not_batches(tmp_path, monkeypatch):
    work = tmp_path / 'work'
    work.mkdir()
    monkeypatch.chdir(work)
    dataset = torch.utils.data.TensorDataset(torch.zeros(12,3,32,32), torch.tensor([0,1]*6))
    dataset.targets = [0,1]*6
    monkeypatch.setitem(data.DATALOADER, 'fixture', lambda *a: (dataset,dataset,2,'fixture'))
    attack = SimpleNamespace(improved=False, distributed=False, attackers=[], source_class=0, target_class=1,
        embed_trigger_in_image=lambda images,labels,*a,**kw:list(zip(images,labels)))
    loader, _, count = data.get_data(attack, 'count', name='fixture', data_split='iid', num_clients=2, batch_size=2)
    assert len(loader.dataset) == 6
    assert count == 6
    for _ in range(2):
        evaluation, _, eval_count = data.get_data(attack,'count',name='fixture',num_clients=2,batch_size=2,return_eval_ds=True)
        assert len(evaluation.dataset) == eval_count == 12


def test_cli_runs_a_real_synthetic_federated_round(tmp_path):
    result = subprocess.run([sys.executable, '-m', 'fiba', '--dataset', 'synthetic', '--num-clients', '2',
        '--num-rounds', '1', '--local-epochs', '1', '--batch-size', '4', '--attacker-fraction', '.5',
        '--target-class', '1', '--output', str(tmp_path/'run'), '--device', 'cpu'], capture_output=True,text=True,timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path/'run'/'results.json').read_text())
    assert report['dataset'] == 'synthetic'
    assert len(report['rounds']) == 1
    assert report['rounds'][0]['trained_clients'] == 2
    assert report['rounds'][0]['examples'] == 16
    assert report['rounds'][0]['parameters_changed'] is True
    assert 0 <= report['rounds'][0]['asr'] <= 1


def test_noniid_partitions_are_disjoint_complete_and_seeded():
    dataset = torch.utils.data.TensorDataset(torch.zeros(120,3,32,32),torch.tensor([0,1,2]*40))
    dataset.targets = [0,1,2]*40
    first = data.noniid_split(dataset,3,[0,1,2],3,skewness=.5,seed=42)
    second = data.noniid_split(dataset,3,[0,1,2],3,skewness=.5,seed=42)
    indices = [idx for subset in first.values() for idx in subset.indices]
    assert sorted(indices) == list(range(120))
    assert [list(v.indices) for v in first.values()] == [list(v.indices) for v in second.values()]
