"""Real tensor, partition and memory-map contracts on bounded unit fixtures."""
import numpy as np
import pytest
import torch
from fiba import data


def test_raw_dataset_is_memory_mapped_and_leaves_shared_storage_unchanged(tmp_path):
    images = np.zeros((4, 32, 32, 3), dtype=np.uint8)
    images[0, 0, 0] = [255, 128, 0]
    np.save(tmp_path / 'train_images.npy', images)
    np.save(tmp_path / 'train_labels.npy', np.array([0, 1, 2, 3]))
    dataset = data.CIFARDataset(tmp_path, 'train', 10)
    x, y = dataset[0]
    assert isinstance(dataset.images, np.memmap)
    assert x.shape == (3, 32, 32) and x.dtype == torch.float32 and y == 0
    torch.testing.assert_close(x[:, 0, 0], torch.tensor([1., 128 / 255, 0.]))
    x.zero_()
    assert dataset.images[0, 0, 0, 0] == 255
    assert len(dataset.classes) == 10


@pytest.mark.parametrize('split', ['iid', 'dirichlet'])
def test_partitions_cover_every_sample_once_without_global_rng_changes(split):
    labels = np.arange(503) % 10
    np.random.seed(6)
    state = np.random.get_state()
    first = data.partition_indices(labels, 10, split, alpha=.5, seed=42)
    second = data.partition_indices(labels, 10, split, alpha=.5, seed=42)
    np.testing.assert_array_equal(np.random.get_state()[1], state[1])
    assert all(len(x) for x in first)
    np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(503))
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    other = data.partition_indices(labels, 10, split, alpha=.5, seed=43)
    assert any(not np.array_equal(a, b) for a, b in zip(first, other))


def test_extreme_dirichlet_and_one_example_per_client_still_terminate():
    result = data.partition_indices(np.array([0, 0, 1, 1]), 4, 'dirichlet', alpha=1e-9)
    assert sorted(map(len, result)) == [1, 1, 1, 1]
    np.testing.assert_array_equal(np.sort(np.concatenate(result)), np.arange(4))


@pytest.mark.parametrize('labels,clients,split,alpha', [([], 2, 'iid', .5), ([0], 2, 'iid', .5), ([0, 1], 0, 'iid', .5), ([0, 1], 2, 'bad', .5), ([0, 1], 2, 'dirichlet', 0)])
def test_bad_partitions_fail_early(labels, clients, split, alpha):
    with pytest.raises(ValueError):
        data.partition_indices(labels, clients, split, alpha)


def test_batch_preprocessing_matches_original_normalization_and_seeded_flip():
    x = torch.zeros(2, 3, 32, 32)
    x[:, :, :, 0] = 1
    before = x.clone()
    expected = (x - torch.tensor([.4914, .4822, .4465])[None, :, None, None]) / torch.tensor([.2023, .1994, .2010])[None, :, None, None]
    torch.testing.assert_close(data.normalize(x), expected)
    a = data.augment(x, torch.Generator().manual_seed(8))
    b = data.augment(x, torch.Generator().manual_seed(8))
    torch.testing.assert_close(a, b)
    assert all(torch.equal(image, expected[i]) or torch.equal(image, expected[i].flip(-1)) for i, image in enumerate(a))
    torch.testing.assert_close(x, before)


def test_raw_augmentation_can_precede_trigger_without_normalization():
    x = torch.rand(3, 3, 32, 32)
    a = data.augment_raw(x, torch.Generator().manual_seed(10))
    assert torch.all((0 <= a) & (a <= 1))
    assert all(torch.equal(v, x[i]) or torch.equal(v, x[i].flip(-1)) for i, v in enumerate(a))
    torch.testing.assert_close(data.normalize(a), data.augment(x, torch.Generator().manual_seed(10)))
