import torch
from fiba.lira_poisoning import LiraAttack

class ConstantNoise(torch.nn.Module):
    def forward(self, x):
        return torch.ones_like(x) * .5


def test_lira_uses_external_artifact_without_bundled_generator(tmp_path):
    path = tmp_path / 'generator.pt'
    torch.jit.trace(ConstantNoise(), torch.zeros(1,3,32,32)).save(str(path))
    attack = LiraAttack([0],[(0,0,32,32)],0,1,False,None,generator_path=path)
    result = attack.embed_trigger_in_image([torch.zeros(3,32,32)],[0],attack)
    assert result[0][1] == 1
    torch.testing.assert_close(result[0][0], torch.full((3,32,32), .05))
    assert attack.targeted is False
