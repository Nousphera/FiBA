# Source provenance and release changes

## Recovered source

The benchmark was recovered from the authors' `fl_battacks` research repository
at commit `77ee2a9208b909e9e83e24033f07a1338dc45756`. Its source contributors are
Matthijs Marinus and Vasileios Tsouvalas. The original repository's Apache-2.0
license is preserved in `licenses/FiBA-Apache-2.0.txt`. Private Git history,
checkpoints, datasets and experiment outputs are not copied into this release.

`fedavg/src/*.py` became `src/fiba/*.py`; dataset loaders became
`src/fiba/data_sources/dataloader.py`; the used Encoder/Decoder training path
from `encoder/src/main.py` became `src/fiba/encoder_training.py`. Existing public
project-page, poster and citation resources are retained.

## Changes made 2026-10-15

1. Added a package, CLI, explicit options, scoped output directories, pinned
   tested dependencies, CI and tests. The sequential runner executes the real
   recovered client/server methods and round-trips standard Flower protobuf
   messages. No Flower internals are patched.
2. Replaced the five-field NumPyClient return and custom FitRes attributes with
   three standard return values. Client ID and classifier gradient use scalar
   and ndarray-bytes metrics; gradients are decoded without pickle on transport.
   Training metrics are scalars, and parameter retrieval follows Flower's API.
3. Removed unused import-time LPIPS model construction/downloads. Dataset
   downloads are explicit. The training CLI forwards the device to server and
   clients and supports CPU. Encoder loading uses CPU state dictionaries.
4. Corrected sample counts from batches to examples for weighted aggregation.
   FoolsGold memory is keyed by client ID across partial-participation rounds.
   Gradient finite checks and explicit zero-trust failures prevent silent
   invalid aggregates. LSB overflow now raises a useful capacity error.
5. Preserved the recovered sign/trigger behavior, Adam/AdamW loops, ResNet-18,
   attack poisoning and defense formulas except for those documented repairs.
   The paper's low-attention convergence criterion is not claimed implemented;
   the source's fixed round 20 remains. BadNet's four-pixel/paper-five-pixel
   discrepancy is explicit in the experiment guide.
6. Removed the copied BackdoorBox GPL-2.0 generator implementation. LIRA now
   consumes a separately exported TorchScript artifact and retains perturbation
   scale 0.1. No BackdoorBox import or pretrained generator is bundled.
7. Excluded PCA and Lockdown ancillary experiments because their cited upstream
   repositories provided no license grant. They are not part of the published
   defense comparison. The historical upstream README is reference material,
   not current installation instructions.

## Third-party licensing evidence

WaNet source in the recovered repository first appears at
`76798fb436e832242fa4495f472995157231b365`, dated 2024-02-26. The latest upstream
code at that time was [94453080f241053ac7c8cc4717da20806ee17e5c](https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release/tree/94453080f241053ac7c8cc4717da20806ee17e5c),
whose `LICENSE` is GPL-3.0. Upstream changed its license in July 2024. The
combined release therefore uses GPL-3.0 and preserves the original Apache
notice. It does not claim an Apache license over the WaNet adaptation.

FoolsGold attribution is retained in the module and `NOTICE`, with upstream MIT
text in `licenses/FoolsGold-MIT.txt`. The release does not distribute GPL-2.0
BackdoorBox source or the unlicensed PCA/Lockdown implementations.

## Validation boundary

See `docs/VALIDATION.md` for the actual release checks. These establish a runnable
package and bounded method/transport behavior. They do not establish reproduction
of the paper's CIFAR accuracy, ASR, LPIPS, Neural Cleanse, attention or model
replacement results. The software license does not alter rights in paper/poster
assets.


## Reproducible setup follow-up

Added dataset import/download/validation commands and deterministic file manifests.
Prepared NPZ files feed the maintained trainer directly. Image labels and shapes
are validated against fixed dataset class spaces; explicit dataset imports do not
certify historical order, partitions or benchmark equivalence. Existing source,
teacher identity and cache qualifications are retained in the linked guides.

Added real Flower ClientApp/ServerApp entry points reusing the maintained core.
Logical partition IDs determine sampling and persistent local state; evaluated
global weights and round results are saved. Flower 1.13/Ray 2.10 remain pinned for
compatibility. See docs/FLOWER_APP.md for single-host simulation limits.
