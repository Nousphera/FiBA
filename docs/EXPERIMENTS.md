# Experiment guide

The [Flower App](FLOWER_APP.md) and sequential runner share the same training
core. Prepare datasets with [DATA_SETUP.md](DATA_SETUP.md).

## Components

| Component | Code | Setting |
| --- | --- | --- |
| CIFAR ResNet-18, Adam | `src/fiba/network.py` | ResNet-18 with local Adam optimization |
| Client partitions and 50% poisoning | `src/fiba/data.py` | IID or class Dirichlet; non-target examples relabeled to target |
| LSB500 / LSB700 | `plain_steganography.py` | ASCII `Apple` repeated to the requested character count |
| WaNet | `wanet.py` | Shared random grid, k=6, s=0.75 |
| BadNet | `badnet.py` | Four-pixel pattern |
| Encoder | `encoder_poisoning.py`, `encoder_training.py` | Encoder/decoder training; 200-bit noise |
| LIRA | `lira_poisoning.py` | External TorchScript generator, perturbation scale 0.1 |
| Server defenses | `aggregation_strategies.py`, `fools_gold.py`, `krum.py` | FedAvg/Median/20%-trimmed mean/FoolsGold/Krum/Multi-Krum |
| Attention LSB | `network.py`, `client.py` | SmoothGradCAM++ at layer1; Fixed extraction round 20 |
| Model replacement | `client.py` | Single-attacker formula, rounds 30/60/90 |

## Data, seeds and metrics

CIFAR-10 and CIFAR-100 use torchvision's public train/test splits. The
preprocessing resizes to 32, uses random horizontal flips for training and
center crops, and normalizes RGB with mean `(0.4914,0.4822,0.4465)` and standard
deviation `(0.2023,0.1994,0.2010)` for both datasets. Client partitions do not
share indices. `--data-split noniid --skewness 0.5` selects the paper's class
Dirichlet setting. The partition is regenerated with the same seed for each
client, selecting the client's distinct subset. Poison caches contain precomputed, augmented samples.

An attacker replaces half of its local examples, sampled among non-target
examples, with triggered inputs labeled as the target. A partition with fewer
eligible non-target examples than the requested replacements cannot satisfy
that definition; the run raises an error rather than silently changing the
poisoning rate. Use and report a valid partition/seed.

Reported accuracy and ASR are fractions in `[0,1]`. Clean accuracy uses the
unmodified test set. ASR uses poisoned test examples excluding the target class.
`results.json` records each round and configuration; it is written after every
completed round. `final-parameters.npz` stores NumPy arrays in model state-dict
order, including batch normalization state. The generated dataset supports short installation and training checks.

The default seed is 42. Save separate outputs for each seed and specify the
aggregation statistic when combining runs. The paper does not define its ±
statistic; choose and name your own uncertainty measure explicitly.

## Attacks and defenses

Start from the CIFAR command in the README and change these options:

- LSB: `--attack steg --trigger-characters 700` (paper baseline) or 500.
- WaNet: `--attack wanet`.
- BadNet: `--attack badnet`.
- Encoder: `--attack encoder --encoder-checkpoint /absolute/path/encoder.pt`.
- LIRA: `--attack lira --lira-generator /absolute/path/lira-generator.pt`.
- Defenses: `--avg-strategy avg`, `median`, `trimmed`, `fg`, or `krum`.
- Multi-Krum: `--avg-strategy krum --krum-clients-to-keep 3`. Record the chosen count with the results. Krum requires
  at least `2*f+3` participating clients for the configured attacker bound.

CIFAR-100 uses `--dataset cifar100`; the paper's defense table uses 20% attackers
there, versus 10% on CIFAR-10. Those settings should not be pooled.

For attention placement, add `--modified-attack` to LSB; add `--mask-inversion`
for high-attention placement. The mask marks the highest-activation half with
ones, and low placement uses zeros. SmoothGradCAM++ extracts the mask at round
20 for use in following rounds. Extraction follows this fixed schedule rather
than a RankMe convergence trigger.

Distributed LSB/WaNet uses `--distributed --distribution 4` with exactly four
attackers. `--boost-strat model_replacement` requires exactly one attacker and
scales its update on rounds 30, 60, and 90.

## Train the Encoder artifact

```bash
python -m fiba.encoder_training --download --data-root datasets \
  --epochs 100 --noise-dim 200 --batch-size 128 --seed 42 \
  --device cuda:0 --output checkpoints/encoder.pt
```

Training uses AdamW, MSE reconstruction objective,
noise weight 0.7, learning rate 0.0005, gradient clipping and StepLR schedules.
It writes the encoder state dictionary and a JSON training record. A four-image
training/checkpoint compatibility check is available with `--synthetic
--epochs 1 --batch-size 2 --device cpu`. Supply the resulting checkpoint with `--encoder-checkpoint`.

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
input. The generator is loaded as an external artifact; keep its applicable
distribution terms and use a trusted checkpoint.

## Metrics and implementation settings

- Weighted aggregation uses actual example counts.
- BadNet changes four pixels in this implementation; the paper describes a
  five-pixel pattern. Specify the implemented pattern when comparing attacks.
- FoolsGold accumulates the last minibatch's classifier-weight gradient by
  stable client identity. If all trust weights vanish, aggregation raises an
  error instead of producing non-finite weights.
- The trainer reports clean accuracy and ASR. LPIPS can be installed through
  the optional `metrics` extra; [Neural Cleanse](https://github.com/bolunwang/backdoor)
  is a separate evaluation workflow.
- The script runner is sequential; the Flower App uses local Ray workers with
  a shared filesystem. Network and device timing require separate measurements.

Component terms are included in [licenses/THIRD_PARTY.txt](../licenses/THIRD_PARTY.txt).
