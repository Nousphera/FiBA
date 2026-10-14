# FiBA

**On the Effectiveness of Invisible Backdoor Attacks in Federated Learning**
[Paper](https://doi.org/10.1007/978-981-96-4836-8_13) · [Project page](https://nousphera.github.io/FiBA/) · [Code](https://github.com/Nousphera/FiBA) · [Source provenance](PROVENANCE.md)

## Abstract

Federated Learning (FL) enables collaborative training of machine learning models across multiple devices, while preserving data privacy. However, it also introduces vulnerabilities to backdoor attacks, where malicious updates can corrupt the global model. This work focuses on the largely unexplored domain of Federated invisible Backdoor Attacks (FiBAs), which use visually indistinguishable triggers to manipulate model behavior without being detected. We investigate the feasibility and effectiveness of these attacks, considering the unique challenges posed by FL, such as limited local training time and the need for model update aggregation. Our study presents a comprehensive evaluation of the attack success rate (ASR) of invisible BAs under various settings and defenses. Based on our observations, we propose a backdoor trigger hiding technique based on low model attention regions to improve attack resilience in federated settings. Our findings provide critical insights into the optimization of invisible BAs in FL and highlights the need for robust defense mechanisms to safeguard FL systems.

*The abstract describes the published work. The maintained release has not reproduced its benchmark tables.*

## Setup

Use Python 3.11. The commands below create a dedicated virtual environment.
Flower 1.13.1 and Ray 2.10.0 are pinned; TensorFlow projects require the
compatible protobuf dependency range. Keep the checkout and virtual environment
on paths without spaces for Ray subprocesses. Do not upgrade Flower independently.

```bash
mkdir -p "$HOME/research-code" "$HOME/.venvs"
cd "$HOME/research-code"
git clone https://github.com/Nousphera/FiBA.git
cd FiBA
python3.11 -m venv "$HOME/.venvs/fiba"
source "$HOME/.venvs/fiba/bin/activate"
python -m pip install -e '.[test]' -r requirements-simulation.txt
python -m pytest -q
```

## Datasets

```bash
python scripts/prepare_data.py --dataset cifar10 --root datasets --download
python scripts/prepare_data.py --dataset cifar10 --root datasets --validate
```

Use `cifar100` for CIFAR-100. [Dataset setup](docs/DATA_SETUP.md) explains validated
NPZ imports, hashes and how the files feed training. Encoder and LIRA additionally
require trained artifacts; their training/export contracts are in the
[experiment guide](docs/EXPERIMENTS.md). No trained artifact is bundled.

## Run

```bash
flwr run .
```

The default runs two rounds on 16 generated training images and eight separate
evaluation images, with two clients and one attacker. It invokes the actual
recovered client/server code through Flower App messages. For CIFAR-10:

```bash
flwr run . paper --run-config 'options="--dataset cifar10 --data-root datasets --num-clients 10 --num-rounds 100 --local-epochs 5 --batch-size 64 --data-split noniid --skewness .5 --participation 1 --lr .001 --attacker-fraction .1 --target-class 5 --attack steg --trigger-characters 700 --avg-strategy avg --seed 42 --device cpu --output runs/cifar10-lsb700-seed42"'
```

For CUDA, choose `--device cuda:0` and configure Flower's GPU resources in
`pyproject.toml`; the checked profile is CPU. Other attacks/defenses and external
artifact requirements are in [EXPERIMENTS.md](docs/EXPERIMENTS.md). The sequential
runner accepts the same arguments:

```bash
fiba --dataset synthetic --num-clients 2 --num-rounds 2 --local-epochs 1 \
  --batch-size 4 --attacker-fraction .5 --target-class 1 --output runs/sequential
```

## Outputs and reproducibility

Every run requires a new output directory. `results.json` records configuration,
versions, sampled logical clients, example counts and round metrics;
`final-parameters.npz` saves the evaluated global weights. Native simulations
preserve run-owned caches under the output directory across worker lifetimes.
They are local simulations using a shared filesystem, not multi-host deployments.
They do not measure network latency. Final weights are not complete resumable
optimizer/client-state checkpoints. Synthetic metrics are execution checks only.

The `options` run-config value is the complete runner argument string. Set the selected federation's
`options.num-supernodes` to exactly the configured client count; mismatches fail
explicitly. [Flower setup details](docs/FLOWER_APP.md) document identity, state,
sampling and the tested runtime boundary.

The combined release retains its GPL-3.0-only provenance and disclosed source
limits: four-pixel BadNet, fixed round-20 attention placement, external LIRA,
and missing original checkpoints/evaluation logs. FoolsGold history is keyed by
logical client identity. See [EXPERIMENTS.md](docs/EXPERIMENTS.md),
[PROVENANCE.md](PROVENANCE.md) and [validation](docs/VALIDATION.md).

## Citation

Use [CITATION.cff](CITATION.cff) or [citation.bib](citation.bib).

## License

The combined source is [GPL-3.0-only](LICENSE). Original Apache-2.0 and MIT notices remain in [NOTICE](NOTICE) and `licenses/`. The historical WaNet license and excluded third-party code are explained there. Software licensing does not relicense paper/poster assets.
