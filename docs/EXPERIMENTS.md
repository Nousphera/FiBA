# Experiment guide and method correspondence

The native `flwr run .` path is documented in [FLOWER_APP.md](FLOWER_APP.md).
The sequential commands below remain useful for CPU debugging; both paths reuse
the same maintained algorithm core. [DATA_SETUP.md](DATA_SETUP.md) provides the
current preparation/import/validation entry points.

## What is implemented

| Paper component | Code | Scope |
| --- | --- | --- |
| CIFAR ResNet-18, Adam | `src/fiba/network.py` | Recovered architecture and local training |
| Client partitions and 50% poisoning | `src/fiba/data.py` | IID or class Dirichlet; non-target examples relabeled to target |
| LSB500 / LSB700 | `plain_steganography.py` | ASCII `Apple` repeated to the requested character count |
| WaNet | `wanet.py` | Shared random grid, k=6, s=0.75 |
| BadNet | `badnet.py` | Recovered four-pixel pattern; paper says five pixels |
| Encoder | `encoder_poisoning.py`, `encoder_training.py` | Recovered encoder/decoder training; 200-bit noise |
| LIRA | `lira_poisoning.py` | External TorchScript generator, perturbation scale 0.1 |
| Server defenses | `aggregation_strategies.py`, `fools_gold.py`, `krum.py` | FedAvg/Median/20%-trimmed mean/FoolsGold/Krum/Multi-Krum |
| Attention LSB | `network.py`, `client.py` | SmoothGradCAM++ at layer1; original fixed extraction round 20 |
| Model replacement | `client.py` | Recovered single-attacker formula, rounds 30/60/90 |

The implementation comes from a recovered research checkout. These mappings
establish source correspondence, not reproduction of every published table.

## Data, seeds and metrics

CIFAR-10 and CIFAR-100 use torchvision's public train/test splits. The recovered
preprocessing resizes to 32, uses random horizontal flips for training and
center crops, and normalizes RGB with mean `(0.4914,0.4822,0.4465)` and standard
deviation `(0.2023,0.1994,0.2010)` for both datasets. Client partitions do not
share indices. `--data-split noniid --skewness 0.5` selects the paper's class
Dirichlet setting. The partition is regenerated with the same seed for each
client, selecting the client's distinct subset. Poison caches retain the
recovered source's precomputed, augmented samples.

An attacker replaces half of its local examples, sampled among non-target
examples, with triggered inputs labeled as the target. A partition with fewer
eligible non-target examples than the requested replacements cannot satisfy
that definition; the run raises an error rather than silently changing the
poisoning rate. Use and report a valid partition/seed.

Reported accuracy and ASR are fractions in `[0,1]`. Clean accuracy uses the
unmodified test set. ASR uses poisoned test examples excluding the target class.
`results.json` records each round and configuration; it is written after every
completed round. `final-parameters.npz` stores NumPy arrays in model state-dict
order, including batch normalization state. Do not treat the synthetic dataset
as evidence of attack strength.

The paper fixes partition randomness but does not state the seed value,
number of repeated runs, or meaning of its ± statistic. The release defaults to
seed 42 as an explicit new choice. Record separate seeds and aggregate your own
runs; do not label the paper's ± values as standard deviations without evidence.

## Attacks and defenses

Start from the CIFAR command in the README and change these options:

- LSB: `--attack steg --trigger-characters 700` (paper baseline) or 500.
- WaNet: `--attack wanet`.
- BadNet: `--attack badnet`.
- Encoder: `--attack encoder --encoder-checkpoint /absolute/path/encoder.pt`.
- LIRA: `--attack lira --lira-generator /absolute/path/lira-generator.pt`.
- Defenses: `--avg-strategy avg`, `median`, `trimmed`, `fg`, or `krum`.
- Multi-Krum: `--avg-strategy krum --krum-clients-to-keep 3`. The chosen count
  must be reported; the paper does not specify it unambiguously. Krum requires
  at least `2*f+3` participating clients for the configured attacker bound.

CIFAR-100 uses `--dataset cifar100`; the paper's defense table uses 20% attackers
there, versus 10% on CIFAR-10. Those settings should not be pooled.

For the attention study add `--modified-attack` to LSB; add `--mask-inversion`
for high-attention placement. The recovered mask marks the highest-activation
half with ones, and the low placement uses its zeros. The mask is extracted
at round 20 and used when clients load data in following rounds. Although the
paper discusses RankMe convergence, the recovered RankMe calculation is
commented out. This release keeps the fixed-round behavior and does not claim
an implemented convergence detector. This path has an activation-map unit check,
not a full 100-round attention benchmark rerun.

Distributed LSB/WaNet uses `--distributed --distribution 4` with exactly four
attackers. The publication's figure caption mixes four parts with a one-attacker
setting; this release requires a consistent explicit configuration.
`--boost-strat model_replacement` retains the original formula and requires
exactly one attacker. Full model-replacement trajectories have not been rerun.

## Train the Encoder artifact

```bash
python -m fiba.encoder_training --download --data-root datasets \
  --epochs 100 --noise-dim 200 --batch-size 128 --seed 42 \
  --device cuda:0 --output checkpoints/encoder.pt
```

This retains the recovered AdamW training loop, MSE reconstruction objective,
noise weight 0.7, learning rate 0.0005, gradient clipping and StepLR schedules.
It writes the encoder state dictionary and a JSON training record. A four-image
training/checkpoint compatibility check is available with `--synthetic
--epochs 1 --batch-size 2 --device cpu`. No trained encoder is bundled, and this
release has not rerun the 100-epoch pretraining.

## Supply a LIRA artifact

LIRA training and generator source are external. In a **separate environment**,
use the original [BackdoorBox](https://github.com/THUYimingLi/BackdoorBox) training
workflow and its applicable GPL-2.0 license. Load the trained CIFAR generator
there, put it in evaluation mode on CPU, and export it with TorchScript:

```python
# Run in that external project's environment after training/loading generator.
generator = generator.cpu().eval()
example = torch.zeros(1, 3, 32, 32)
exported = torch.jit.trace(generator, example)
exported.save("lira-generator.pt")
```

The artifact must accept normalized float32 `[N,3,32,32]` inputs and return a
finite noise tensor of the same shape. This adapter adds `0.1 * noise` to each
input, matching the recovered FiBA perturbation scale. The package neither
imports BackdoorBox nor includes its copied UNet source. A constant synthetic
generator is used only in the adapter compatibility test. No trained LIRA
artifact or successful attack reproduction is claimed. Use artifacts from
trusted sources, and retain the artifact's own applicable distribution terms.

## Unreproduced evaluation and known differences

- LPIPS and Neural Cleanse numbers are not reproduced by the training command.
  LPIPS remains an optional upstream dependency; Neural Cleanse is external at
  [bolunwang/backdoor](https://github.com/bolunwang/backdoor). No Neural Cleanse
  script or model outputs were recovered. The training package does not
  instantiate or download LPIPS weights on import.
- The recovered BadNet implementation changes **four** pixels, whereas the paper
  describes five. It is retained and disclosed; no fifth pixel was invented.
- Weighted aggregation now uses actual example counts. The research source
  returned the number of batches, which can alter unequal-client weighting.
- FoolsGold accumulates the last minibatch's classifier-weight gradient as in
  the original source. Client identity is now stable under partial participation.
  If all trust weights vanish, it fails explicitly rather than producing NaNs.
- The script runner is sequential; the native Flower App uses local Ray workers.
  Neither is a reproduction of network traffic or timing measurements.
- PCA and Lockdown were ancillary source experiments, absent from the paper's
  evaluated defense grid, and are excluded because no upstream license grant
  was found. LIRA is absent from the paper's defense-grid table as well.

No reference checkpoints or raw paper result logs were recovered for inclusion.
