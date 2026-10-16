import pytest
import torch

from fiba.attacks import build_attack


def test_lsb_uses_two_bitplanes_for_700_characters_and_obeys_mask():
    x = torch.full((1, 3, 32, 32), 128 / 255)
    trigger = build_attack({'attack': 'lsb', 'trigger_characters': 700})
    result = trigger.apply(x)
    delta = (result * 255).round() - (x * 255).round()
    assert delta.abs().max() <= 3
    assert torch.any(delta != 0)
    mask = torch.zeros(32, 32, dtype=torch.bool)
    mask[:16] = True
    masked = trigger.apply(x, mask=mask)
    assert torch.equal(masked[:, :, :16], x[:, :, :16])
    assert torch.any(masked[:, :, 16:] != x[:, :, 16:])


def test_badnet_five_white_pixels_and_distributed_composition():
    x = torch.zeros(1, 3, 32, 32)
    trigger = build_attack({'attack': 'badnet', 'distributed': True,
                            'num_clients': 10, 'attacker_fraction': .4})
    full = trigger.apply(x)
    assert (full[0, 0] == 1).sum() == 5
    parts = [trigger.apply(x, client_id=i) for i in range(4)]
    assert torch.equal(sum(parts), full)
    with pytest.raises(ValueError, match='mask'):
        trigger.apply(x, mask=torch.zeros(32, 32))


def test_wanet_is_seeded_shared_and_validates_input():
    x = torch.rand(2, 3, 32, 32)
    a = build_attack({'attack': 'wanet', 'seed': 7})
    b = build_attack({'attack': 'wanet', 'seed': 7})
    assert torch.equal(a.apply(x), b.apply(x))
    assert not torch.equal(a.apply(x), x)
    with pytest.raises(ValueError):
        a.apply(x + 2)
