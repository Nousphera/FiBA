# Testing

Activate the environment from the [README](../README.md#installation), then run:

```bash
python -m pip check
python -m pytest -q
```

The suite covers client training, attack capacity and attention placement, Encoder save/load, LIRA input contracts, aggregation rules, FoolsGold identity, dataset preparation, and Flower round outputs.
Generated inputs keep regression tests small and deterministic. Dataset tests
exercise import, validation, content hashes, and consumption by the training
loader without downloading complete corpora.

## Native Flower smoke check

```bash
flwr run .
python scripts/check_run.py runs/flower-smoke --rounds 2
```

Use a new output directory for repeated runs. The checker inspects saved results
and requires completed rounds and finite final parameters. It supplements the
CLI exit status with checks on training outputs.

[GitHub Actions](https://github.com/Nousphera/FiBA/actions) runs the CPU checks
on Linux with Python 3.11. See each run for its commit, dependency installation,
and test output.
