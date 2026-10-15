> **Archived upstream reference.** Original instructions below are preserved for provenance and are superseded by the [maintained README](../README.md) and [experiment guide](EXPERIMENTS.md). Historical commands and directory names describe the archived layout. The dataset and framework links point to maintained equivalents; do not apply the Flower patches below.

# Benchmark of Backdoor Attacks in FL Systems

## Overview

Federated Learning (FL) offers a privacy-preserving approach in distributed machine learning, enabling model training without direct data sharing, proving especially valuable in fields like medicine. However, its decentralized nature introduces vulnerabilities, notably backdoor attacks, where attackers embed subtle "triggers" to manipulate model behavior. Despite mitigation efforts, these threats evolve. This thesis delves into a lesser-explored aspect: embedding visually undetectable triggers in FL models. We investigate the challenges posed by FL's inherent model update averaging and explore techniques to enhance the efficacy and stealthiness of backdoor attacks, considering their detectability by defenses.

## Installation

Project was build with Python 3.11.1. It is highly suggest to install [pyenv](https://github.com/pyenv/pyenv#installation) and create a virtual enviroment for this project. In a Linux-based system, you can do that by running the following commands:

```console
curl https://pyenv.run | bash
```

Next, setup all required dependces to install Python via pyenv by executing:

```console
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
sudo apt update; sudo apt install build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
```

Restart your terminal and run:

```console
pyenv install 3.11.1 && pyenv virtualenv 3.11.1 fl_battacks
```

Now, you will have created a virtualenv called `fl_battacks`. You can activate it and install all required python packages by running:

```console
pyenv activate fl_battacks && pip3 install -r requirements.txt
```

### Datasets preparation

You need to set the enviromental variable 'TORCH_DATA_DIR' to direct to [dataset setup](DATA_SETUP.md) dir. In a Linux-based system, from the root of this repo, run:

```console
echo "export TORCH_DATA_DIR=$(realpath .)/datasets" >> ~/.bashrc && source ~/.bashrc
```

## Execution Instructions

### Standard FedAvg training

A minimal, highly configurable FL framework for fast prototyping of federated applications is provided under [maintained federated-learning implementation](../src/fiba/). The framework is based on [PyTorch](https://pytorch.org/) and [Flower](https://flower.dev/), where clients are simulated via distinct processes using [Ray](https://docs.ray.io/en/latest/index.html). From the root directory of this repo, run:

```console
cd ./fedavg/src
python main.py --dataset cifar10 --num_clients 4  --attacker_fraction --num_rounds 10 --local_epochs 1 --participation 1.0 --vram_per_client 0.25 --cpu_per_client 1
```

A full list of all parameters with a small explanation of what it changes can be found in main.py.

### Attacks
#### LSB Attack
The LSB attack can be tested using the '--attack steg' parameter. Be sure to set the desired trigger size in the plain_steganography.py file. This attack can be modified by specifying the --distributed and --distribution x commands which splits the trigger over x attackers. Make sure the number of attackers equals x. Additionally a modified version of this attack can be used which uses model attention to hide the trigger in low/high activation areas. Use --modified_attack to use low activation areas and --modified_attack --mask_inversion to use high activation areas.


#### WaNet Attack
The WaNet attack can be tested using the '--attack wanet' command. Be sure to set the correct warping parameters in the wanet.py file which control the warping strength. This attack can be modified by using the '--distributed' and --distribution x commands which splits the trigger over x attacks.

#### Encoder Attack
The Encoder attack can be tested using the '--attack encoder' command. To be able to use this attack you first have to create an encoder model. The encoder model can be found under /encoder/main.py. Simply run this file (be sure to set the correct noise_dim for the trigger size). This produces an encoder.th file which can be placed in the encoder_model folder. The attack will then use this model to poison samples.

#### LIRA Attack
The LIRA attack can be tested using the --attack lira command. To be able to use this attack you first have to create a LIRA poisoning model. This can be done using the following repo: https://github.com/THUYimingLi/BackdoorBox. Once you have the model you can put the model in a lira_model folder. The attack will then use this model to poison the samples.

#### BadNet Attack
The BadNet attack can be tested using the --attack badnet command.

The model replacement strategy can be applied to all attacks using --boost_strat model_replacement. This will scale up the model update of attackers in rounds 30, 60, 90 (currently only works for 1 attacker in the system because of how the formula is derived).

### Defenses
There are a number of defenses available to test on. They can be selected using --avg_strategy.

* Weighted Averaging (FedAvg): "avg"
* Median: "median"
* Trimmed Mean: "trimmed"
* (Multi)-KRUM (set num_clients_to_keep in server.py to 0 for KRUM, a higher number selects Multi-KRUM): "krum"
* FoolsGold: "fg"
* PCA: "pca"
* Neural Cleanse: https://github.com/bolunwang/backdoor

### Framework Modifications
This code uses the flwr framework to simulate Federated Learning. To facilitate some of the attacks and defenses we had to modify the framework slightly to return some extra information to the server.

#### App.py
In the flwr client/app.py file replace the _fit function with the following:
```py
def _fit(self: Client, ins: FitIns) -> FitRes:
    """Refine the provided parameters using the locally held dataset."""
    # Deconstruct FitIns
    parameters: NDArrays = parameters_to_ndarrays(ins.parameters)

    # Train
    results = self.numpy_client.fit(parameters, ins.config)  # type: ignore
    if not (
        len(results) == 5
        and isinstance(results[0], list)
        and isinstance(results[1], int)
        and isinstance(results[2], dict)
        and isinstance(results[3], list)
        and isinstance(results[4], int)
    ):
        raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT)

    # Return FitRes
    parameters_prime, num_examples, metrics, gradients, client_id = results
    parameters_prime_proto = ndarrays_to_parameters(parameters_prime)
    return FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=parameters_prime_proto,
        num_examples=num_examples,
        metrics=metrics,
        gradients=gradients,
        client_id=client_id
    )
```

In the client/client.py file replace maybe_call_fit with the following:
```py
def maybe_call_fit(client: Client, fit_ins: FitIns) -> FitRes:
    """Call `fit` if the client overrides it."""
    # Check if client overrides `fit`
    if not has_fit(client=client):
        # If client does not override `fit`, don't call it
        status = Status(
            code=Code.FIT_NOT_IMPLEMENTED,
            message="Client does not implement `fit`"
        )
        return FitRes(
            status=status,
            parameters=Parameters(tensor_type="", tensors=[]),
            num_examples=0,
            metrics={},
            gradients=[],
            client_id=0
        )

    # If the client implements `fit`, call it
    return client.fit(fit_ins)
```

Finally in common/typing.py file replace FitRes class with the follwing:
```py
@dataclass
class FitRes:
    """Fit response from a client."""

    status: Status
    parameters: Parameters
    num_examples: int
    metrics: Dict[str, Scalar]
    gradients: []
    client_id: int
```
