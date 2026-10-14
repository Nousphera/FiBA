"""FiBA's LIRA perturbation wrapper around an external TorchScript generator.

No BackdoorBox generator implementation or trained weights are distributed.
The external artifact accepts [N,3,32,32] normalized images and returns noise of
identical shape. FiBA's recovered perturbation scale is 0.1.
"""
from pathlib import Path
import torch


class LiraAttack:
    def __init__(self, attacker_ids, ranges, source_class, target_class,
                 distributed, boost_strat, generator_path=None):
        if generator_path is None:
            raise ValueError('LIRA requires --lira-generator; see docs/EXPERIMENTS.md')
        self.name = 'LIRA Attack'
        self.attackers = attacker_ids
        self.ranges = ranges
        self.source_class = source_class
        self.target_class = target_class
        self.distributed = distributed
        self.boost_strat = boost_strat
        self.improved = False
        self.targeted = False
        self.atkmodel = torch.jit.load(str(Path(generator_path)), map_location='cpu').eval()

    def embed_trigger_in_image(self, images, labels, attack_setting,
                               input_range=None, mixup=False, mask=None):
        if input_range is not None or self.distributed:
            raise ValueError('Distributed LIRA is not part of the recovered paper experiment')
        poisoned = []
        with torch.inference_mode():
            for image in images:
                batch = image.detach().cpu().unsqueeze(0)
                noise = self.atkmodel(batch)
                if noise.shape != batch.shape or not torch.isfinite(noise).all():
                    raise ValueError('LIRA generator must return finite noise matching its input shape')
                poisoned.append(((batch + .1 * noise).squeeze(0), self.target_class))
        return poisoned
