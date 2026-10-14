# Release validation — 2026-10-15

Environment: macOS ARM64, Python 3.11.15, CPU, PyTorch 2.6.0,
torchvision 0.21.0, Flower 1.13.1 and NumPy 1.26.4, in a dedicated environment.
No paper dataset, pretrained attack generator or private result file is needed
for these checks.

- Editable installation with `python -m pip install -e '.[test]'` succeeded.
- `python -m pytest -q`: 16 tests passed. They cover real client training with
  stock Flower serialization; scalar/gradient transport; example counts;
  disjoint seeded partitions; sample-weighted, median, trimmed and Krum rules;
  FoolsGold client identity under partial participation; ASCII LSB bits and
  capacity; attention placement; an external synthetic LIRA artifact;
  Encoder training/save/load; import side effects; and the end-to-end CLI.
- One-round synthetic ResNet training also completed with WaNet+Median,
  BadNet+Trimmed Mean, LSB+FoolsGold, and LSB+Krum. The first three used two
  clients; Krum used five clients with one attacker. These test path execution,
  not attack efficacy. Multi-Krum has an independent numerical unit fixture.
- Source distribution and wheel builds passed. The wheel contains GPL-3.0,
  original Apache-2.0 and FoolsGold MIT license texts and NOTICE.
- The built wheel was installed without editable source and completed a
  WaNet synthetic training round from an unrelated working directory.
- `pip check` found no broken requirements.

The package's default `--device cpu` and the generated-image fixture make the
checks bounded. CIFAR-10/100 paper-scale runs, 100-epoch Encoder pretraining,
trained LIRA weights, full attention/model-replacement trajectories, LPIPS and
Neural Cleanse were not reproduced. No result from these checks is substituted
for a published accuracy, ASR or detectability value.

GitHub Actions is configured for Python 3.11/Linux, but its remote run has not
been observed as part of this local validation.
