"""Preparation contracts tested with bounded files, never a corpus download."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_data", ROOT / "scripts/prepare_data.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def fixture(path, shape=(32,32,3), classes=10):
    rng = np.random.default_rng(7)
    np.savez(path, train_images=rng.integers(0,256,(40,*shape),dtype=np.uint8),
             train_labels=np.arange(40) % classes,
             test_images=rng.integers(0,256,(10,*shape),dtype=np.uint8), test_labels=np.arange(10)%classes)


def test_prepare_import_validate_no_overwrite_and_tamper(tmp_path):
    source = tmp_path / "input.npz"
    fixture(source)
    target = tmp_path / "prepared"
    args = ["--dataset", "cifar10", "--root", str(target)]
    first = prep.main([*args, "--source", str(source)])
    assert first["metadata"]["train_rows"] == 40
    assert prep.main([*args, "--validate"]) == first
    with pytest.raises(FileExistsError):
        prep.main([*args, "--source", str(source)])
    with (target / "cifar10.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="manifest"):
        prep.main([*args, "--validate"])


@pytest.mark.parametrize("shape,classes", [((28,28,3),10),((32,32,3),11)])
def test_reject_wrong_shape_or_out_of_range_labels(tmp_path, shape, classes):
    source = tmp_path / "invalid.npz"
    fixture(source, shape, classes)
    with pytest.raises(ValueError):
        prep.main(["--dataset","cifar10","--source",str(source),"--root",str(tmp_path/"target")])
    assert not (tmp_path/"target/cifar10.npz").exists()


def test_prepared_arrays_reach_training_loader(tmp_path, monkeypatch):
    source = tmp_path / "input.npz"
    fixture(source)
    root = tmp_path / "data"
    prep.main(["--dataset","cifar10","--source",str(source),"--root",str(root)])
    monkeypatch.setenv("TORCH_DATA_DIR", str(root))
    from fiba.data_sources.dataloader import get_cifar10
    from torchvision.transforms import ToTensor
    train, test, classes, name = get_cifar10(ToTensor(), ToTensor())
    assert classes == 10 and len(train) == 40 and len(test) == 10
    assert train[0][0].shape == (3,32,32)


def test_public_archive_extraction_blocks_traversal(tmp_path):
    import io
    import tarfile
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as out:
        info = tarfile.TarInfo("../escaped")
        info.size = 1
        out.addfile(info, io.BytesIO(b"x"))
    with tarfile.open(archive) as source:
        with pytest.raises(tarfile.FilterError):
            source.extractall(tmp_path / "safe")
    assert not (tmp_path / "escaped").exists()


def test_existing_manifest_cannot_leave_a_half_published_array(tmp_path):
    source = tmp_path / "input.npz"
    fixture(source)
    target = tmp_path / "prepared"
    target.mkdir()
    saved = target / "cifar10.manifest.json"
    saved.write_text('{"existing":true}')
    with pytest.raises((ValueError, FileExistsError)):
        prep.main(["--dataset", "cifar10", "--source", str(source), "--root", str(target)])
    assert not (target / "cifar10.npz").exists()
    assert saved.read_text() == '{"existing":true}'


def test_array_publication_failure_rolls_back_both_files(tmp_path, monkeypatch):
    source = tmp_path / "input.npz"
    fixture(source)
    target = tmp_path / "prepared"
    real_link = prep.os.link
    calls = []
    def fail_manifest(source, destination):
        calls.append(destination)
        if str(destination).endswith(".manifest.json"):
            raise OSError("publication failed")
        return real_link(source, destination)
    monkeypatch.setattr(prep.os, "link", fail_manifest)
    with pytest.raises(OSError, match="publication failed"):
        prep.main(["--dataset", "cifar10", "--source", str(source), "--root", str(target)])
    assert len(calls) == 2
    assert not (target / "cifar10.npz").exists()
    assert not (target / "cifar10.manifest.json").exists()
    assert not list(target.glob(".prepare-*"))
