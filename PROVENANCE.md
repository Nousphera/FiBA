# Attribution and licenses

FiBA's research implementation is by Matthijs Marinus and Vasileios Tsouvalas,
based on their `fl_battacks` source at commit
`77ee2a9208b909e9e83e24033f07a1338dc45756`. Its Apache-2.0 license is preserved in
[licenses/FiBA-Apache-2.0.txt](licenses/FiBA-Apache-2.0.txt).

The combined source release is [GPL-3.0-only](LICENSE) because the WaNet
adaptation includes GPL-3.0 code. The applicable upstream revision is
[`94453080f241053ac7c8cc4717da20806ee17e5c`](https://github.com/VinAIResearch/Warping-based_Backdoor_Attack-release/tree/94453080f241053ac7c8cc4717da20806ee17e5c).
FiBA's WaNet adaptation was committed on 26 February 2024, before upstream's
July 2024 license change. The GPL-3.0 terms for that version are retained;
individual original contributions keep their applicable Apache permissions.

FoolsGold adapts code by Clement Fung, Chris J. M. Yoon, and Ivan Beschastnikh,
[DistributedML/FoolsGold](https://github.com/DistributedML/FoolsGold), under MIT.
The notice and license text are retained in [NOTICE](NOTICE) and
[licenses/FoolsGold-MIT.txt](licenses/FoolsGold-MIT.txt).

LIRA accepts a separately supplied TorchScript artifact. BackdoorBox source is
not included in this package; artifacts retain their applicable terms. PCA and
Lockdown implementations are excluded because their upstream repositories do
not provide a license grant. Supported attacks and defenses are listed in
[EXPERIMENTS.md](docs/EXPERIMENTS.md).

Software licensing does not change rights in the paper, poster, datasets, or
externally supplied model artifacts.
