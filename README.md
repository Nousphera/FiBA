# FiBA

**On the Effectiveness of Invisible Backdoor Attacks in Federated Learning**
Mattjis Marinus · Vasileios Tsouvalas · Nirvana Meratnia
UbiSec 2024 · Proceedings published 2025

[Paper](https://doi.org/10.1007/978-981-96-4836-8_13) · [Project page](https://nousphera.github.io/FiBA/) · [Research overview](https://nousphera.github.io/projects/fiba/) · [Poster](https://nousphera.github.io/FiBA/poster.pdf)

Recovered research code for the FiBA benchmark: CIFAR ResNet-18 training, LSB,
WaNet, BadNet, Encoder, an external-artifact LIRA adapter, and FedAvg, Median,
Trimmed Mean, FoolsGold and Krum/Multi-Krum aggregation. The package runs with
**unmodified Flower**. Its sequential runner executes real local training and
serializes Flower messages in process; it does not simulate network latency.

This release has installation, algorithm and synthetic training checks. **The
paper's CIFAR tables have not been reproduced with this release.** Recovered
source discrepancies, unavailable pretrained generators and external evaluation
steps are listed in [the experiment guide](docs/EXPERIMENTS.md).

## Install and check

Use Python 3.10–3.12; validation used Python 3.11 on macOS ARM64, CPU. A CUDA
machine is recommended for the paper-scale CIFAR experiments.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

The package pins a tested Flower/PyTorch combination. Do not apply the manual
Flower edits described in the historical source repository.

## Run a small training check

```bash
fiba --dataset synthetic --num-clients 2 --num-rounds 2 \
  --local-epochs 1 --batch-size 4 --attacker-fraction 0.5 \
  --target-class 1 --device cpu --output runs/smoke
```

This trains the recovered ResNet-18 on 16 generated images and evaluates eight
separate generated images, with one attacker. It checks the training path; its
accuracy and ASR are not scientific benchmark results. Each run requires a new
output directory and records its options, seed, participating sample counts,
clean accuracy and ASR in `results.json`, plus `final-parameters.npz`. Runtime
caches remain inside that output directory.

## Run CIFAR-10

```bash
fiba --dataset cifar10 --download --data-root datasets \
  --num-clients 10 --num-rounds 100 --local-epochs 5 --batch-size 64 \
  --data-split noniid --skewness 0.5 --participation 1.0 \
  --lr 0.001 --attacker-fraction 0.1 --target-class 5 \
  --attack steg --trigger-characters 700 --avg-strategy avg \
  --seed 42 --device cuda:0 --output runs/cifar10-lsb700-seed42
```

Downloads happen only when explicitly requested. See `fiba --help` and
[EXPERIMENTS.md](docs/EXPERIMENTS.md) for other attacks, defenses, Encoder training,
LIRA artifact export, attention placement and interpretation limits.

## Provenance and license

The release preserves the recovered implementation's algorithms where possible
and records compatibility repairs in [PROVENANCE.md](PROVENANCE.md). No private
datasets, experiment results, checkpoints or private Git history are included.

The combined source is **GPL-3.0-only**, with original Apache-2.0 and MIT notices
preserved in [NOTICE](NOTICE) and `licenses/`. The historical WaNet license and
excluded third-party code are explained there. Paper/poster assets retain their
own applicable rights; the software license does not relicense the paper.

Please cite the paper using [CITATION.cff](CITATION.cff) or
[citation.bib](citation.bib).
