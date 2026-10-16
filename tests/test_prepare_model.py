"""Model preparation must distinguish omitted placement from an explicit choice."""
import importlib.util
from pathlib import Path
import sys
import pytest


@pytest.mark.parametrize('placement', [None, 'low', 'high'])
def test_prepare_cli_preserves_omitted_attention_placement(tmp_path, monkeypatch, placement):
    path = Path(__file__).parents[1] / 'scripts' / 'prepare_model.py'
    spec = importlib.util.spec_from_file_location('prepare_model_cli', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    observed = []
    monkeypatch.setattr(module, 'train_generator', lambda cfg: observed.append(dict(cfg)) or tmp_path / 'result.pt')
    arguments = ['--attack', 'encoder', '--initial-checkpoint', str(tmp_path / 'high.pt'),
                 '--output', str(tmp_path / 'result.pt')]
    if placement is not None:
        arguments += ['--attention', placement]
    monkeypatch.setattr(sys, 'argv', [str(path), *arguments])
    module.main()
    assert len(observed) == 1
    if placement is None:
        assert 'attention_placement' not in observed[0]
    else:
        assert observed[0]['attention_placement'] == placement
