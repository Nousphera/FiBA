import pytest
import torch

from fiba.generators import Encoder, Decoder, LiraGenerator, save_generator, load_generator
from fiba.attacks import build_attack


def test_encoder_checkpoint_preserves_shared_trigger_and_inference(tmp_path):
    torch.manual_seed(3)
    encoder = Encoder()
    trigger = torch.randint(0, 2, (1, 1, 32, 32)).float()
    checkpoint = tmp_path / 'encoder.pt'
    save_generator(checkpoint, 'encoder', encoder, {'seed': 3}, trigger)
    loaded, metadata = load_generator(checkpoint, 'encoder')
    x = torch.rand(2, 3, 32, 32)
    assert torch.equal(encoder(x, trigger.expand(2, -1, -1, -1)),
                       loaded(x, metadata['trigger'].expand(2, -1, -1, -1)))
    output = build_attack({'attack': 'encoder', 'generator_checkpoint': str(checkpoint)}).apply(x)
    assert output.shape == x.shape
    assert 0 <= output.min() <= output.max() <= 1


def test_generator_gradients_and_lira_noise_bound():
    x = torch.rand(2, 3, 32, 32)
    noise = torch.rand(2, 1, 32, 32)
    encoder, decoder = Encoder(), Decoder()
    encoded = encoder(x, noise)
    loss = (encoded - x).square().mean() + (decoder(encoded) - noise).square().mean()
    loss.backward()
    assert encoder.embedding[-1].weight.grad.abs().sum() > 0
    generator = LiraGenerator()
    assert generator(x).abs().max() <= 1
    generator(x).square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in generator.parameters())


def test_training_runs_both_real_optimizers_and_saves_complete_artifacts(tmp_path, monkeypatch):
    from fiba import data, network
    from fiba import generator_training
    from fiba.generator_training import train_generator
    # Small test fixture only: the production loader accepts CIFAR exclusively.
    dataset = torch.utils.data.TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1]))
    dataset.classes = list(range(10))
    monkeypatch.setattr(data, 'load_datasets', lambda name, root: (dataset, dataset))
    class Surrogate(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Conv2d(3, 10, 1)
        def forward(self, images):
            return self.layer(images).mean((2, 3))
    monkeypatch.setattr(network, 'build_model', lambda num_classes, seed=42: Surrogate())
    observed = []
    def inspect_raw(module, inputs):
        images = inputs[0]
        assert images.min() >= 0 and images.max() <= 1
        observed.append(len(images))
    for name in ['Encoder', 'LiraGenerator']:
        original = getattr(generator_training, name)
        def monitored_factory(original=original):
            model = original()
            model.register_forward_pre_hook(inspect_raw)
            return model
        monkeypatch.setattr(generator_training, name, monitored_factory)
    for kind in ['encoder', 'lira']:
        output = tmp_path / (kind + '.pt')
        result = train_generator({'attack': kind, 'dataset': 'cifar10', 'generator_epochs': 1,
                                  'batch_size': 2, 'device': 'cpu', 'output': str(output)})
        assert result == output
        loaded, metadata = load_generator(result, kind)
        assert metadata['history'][0]['loss'] >= 0
        assert metadata['training_examples'] == 2
        assert metadata['config']['generator_epochs'] == 1
        if kind == 'lira':
            assert metadata['history'][0]['classifier_loss'] > 0
    assert len(observed) >= 3


def test_encoder_attention_requires_matching_weighted_checkpoint(tmp_path):
    trigger = torch.zeros(1, 1, 32, 32)
    plain = tmp_path / 'plain.pt'
    weighted = tmp_path / 'weighted.pt'
    mask = torch.zeros(32, 32, dtype=torch.bool)
    mask[:16] = True
    model = Encoder()
    save_generator(plain, 'encoder', model, {}, trigger)
    with pytest.raises(ValueError, match='attention'):
        build_attack({'attack': 'encoder', 'attention': 'low', 'generator_checkpoint': plain})
    save_generator(weighted, 'encoder', model, {'attention_placement': 'low'}, trigger, attention_mask=mask)
    with pytest.raises(ValueError, match='placement'):
        build_attack({'attack': 'encoder', 'attention': 'high', 'generator_checkpoint': weighted})
    attack = build_attack({'attack': 'encoder', 'attention': 'low', 'generator_checkpoint': weighted})
    images = torch.rand(1, 3, 32, 32)
    with pytest.raises(ValueError, match='mask'):
        attack.apply(images, mask=~mask)
    # Weighted reconstruction is used directly, without an extra hard-clipping
    # trigger definition at inference time.
    loaded, _ = load_generator(weighted, 'encoder')
    assert torch.equal(attack.apply(images, mask=mask), loaded(images, trigger))


def test_encoder_finetuning_preserves_shared_trigger_and_decoder(tmp_path, monkeypatch):
    from fiba import data
    from fiba.generator_training import train_generator
    fixture = torch.utils.data.TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1]))
    fixture.classes = list(range(10))
    monkeypatch.setattr(data, 'load_datasets', lambda name, root: (fixture, fixture))
    initial = train_generator({'attack': 'encoder', 'generator_epochs': 1, 'device': 'cpu',
                               'encoder_bits': 73, 'seed': 7, 'output': tmp_path / 'initial.pt'})
    mask = torch.zeros(32, 32, dtype=torch.bool)
    mask[:, 16:] = True
    tuned = train_generator({'attack': 'encoder', 'generator_epochs': 1, 'device': 'cpu',
                             'initial_checkpoint': initial, 'attention_mask': mask,
                             'attention_placement': 'low', 'output': tmp_path / 'tuned.pt'})
    _, old = load_generator(initial, 'encoder')
    _, new = load_generator(tuned, 'encoder')
    assert torch.equal(old['trigger'], new['trigger'])
    assert torch.equal(new['attention_mask'], mask)
    assert new['config']['encoder_bits'] == 73
    assert any(not torch.equal(old['decoder_state'][key], new['decoder_state'][key]) for key in old['decoder_state'])


def test_encoder_finetuning_inherits_high_attention_and_cannot_relabel_mask(tmp_path, monkeypatch):
    from fiba import data
    from fiba.generator_training import train_generator
    fixture = torch.utils.data.TensorDataset(torch.rand(2, 3, 32, 32), torch.tensor([0, 1]))
    fixture.classes = list(range(10))
    monkeypatch.setattr(data, 'load_datasets', lambda name, root: (fixture, fixture))
    mask = torch.zeros(32, 32, dtype=torch.bool)
    mask[:, :16] = True
    config = {'attack': 'encoder', 'generator_epochs': 1, 'device': 'cpu'}
    initial = train_generator({**config, 'attention_mask': mask, 'attention_placement': 'high',
                               'output': tmp_path / 'high.pt'})
    inherited = train_generator({**config, 'initial_checkpoint': initial,
                                 'output': tmp_path / 'inherited.pt'})
    _, metadata = load_generator(inherited, 'encoder')
    assert metadata['config']['attention_placement'] == 'high'
    assert torch.equal(metadata['attention_mask'], mask)
    with pytest.raises(ValueError, match='placement'):
        train_generator({**config, 'initial_checkpoint': initial, 'attention_placement': 'low',
                         'output': tmp_path / 'invalid.pt'})
    assert not (tmp_path / 'invalid.pt').exists()
    replaced = train_generator({**config, 'initial_checkpoint': initial, 'attention_placement': 'low',
                                 'attention_mask': ~mask, 'output': tmp_path / 'new-mask.pt'})
    _, metadata = load_generator(replaced, 'encoder')
    assert metadata['config']['attention_placement'] == 'low'
    assert torch.equal(metadata['attention_mask'], ~mask)
