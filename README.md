# FiBA

**On the Effectiveness of Invisible Backdoor Attacks in Federated Learning**

Mattjis Marinus · Vasileios Tsouvalas · Nirvana Meratnia  
UbiSec 2024 · Proceedings published 2025

A visually subtle trigger does not behave the same way in centralized and federated training. This study examines how limited local training, competing client updates, and server-side defenses affect invisible backdoor attacks, then investigates whether placing triggers in regions receiving little model attention changes attack success and detectability.

## Project resources

[Project page](https://nousphera.github.io/FiBA/) · [Research overview](https://nousphera.github.io/projects/fiba/) · [A0 poster](https://nousphera.github.io/FiBA/poster.pdf) · [Paper](https://doi.org/10.1007/978-981-96-4836-8_13)

This repository hosts the paper’s project page, poster and citation. **The implementation is not included in this release.** Code will be added here after the final paper implementation has been identified and prepared for release.

## Method at a glance

- **Evaluate invisible triggers.** The study compares several invisible backdoor families with a visible-trigger baseline under federated training. Evaluation measures clean accuracy, attack success, and perceptual similarity.
- **Probe training and defenses.** Experiments vary attacker participation and test server-side defenses, including robust aggregation and behavior detection. Distributed triggers and model replacement are also examined.
- **Study attention-based placement.** Activation maps identify regions receiving less model attention. The paper evaluates placing LSB triggers in these regions and checks both attack success and detection by Neural Cleanse.

## Citation

```bibtex
@inproceedings{marinus2025invisible,
  title = {On the Effectiveness of Invisible Backdoor Attacks in Federated Learning},
  author = {Marinus, Mattjis and Tsouvalas, Vasileios and Meratnia, Nirvana},
  booktitle = {Ubiquitous Security: 4th International Conference, UbiSec 2024, Revised Selected Papers},
  series = {Communications in Computer and Information Science},
  volume = {2469},
  pages = {174--190},
  year = {2025},
  publisher = {Springer},
  doi = {10.1007/978-981-96-4836-8_13},
  url = {https://doi.org/10.1007/978-981-96-4836-8_13}
}
```

## Materials and licensing

Publication and linked implementation licenses remain with their respective authors and publishers. No license for unreleased code is implied by this resource repository.
