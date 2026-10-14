"""Importing algorithms must not allocate/download optional LPIPS weights."""
import importlib
import pytest

@pytest.mark.parametrize('module', ['plain_steganography', 'wanet', 'badnet'])
def test_attack_import_does_not_construct_lpips(module, monkeypatch):
    import lpips
    def forbidden(*args, **kwargs):
        raise AssertionError('LPIPS must only be constructed by an explicit metric command')
    monkeypatch.setattr(lpips, 'LPIPS', forbidden)
    importlib.import_module("fiba." + module)
