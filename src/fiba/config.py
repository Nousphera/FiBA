"""Experiment settings for the CIFAR federated backdoor benchmark."""

import math

DATASETS = ("cifar10", "cifar100")
ATTACKS = ("none", "badnet", "lsb", "wanet", "encoder", "lira")
AGGREGATIONS = ("fedavg", "median", "trimmed-mean", "foolsgold", "krum", "multi-krum")


def defaults():
    return dict(dataset="cifar10", data_root="data", num_clients=10, num_rounds=100,
                local_epochs=5, batch_size=64, lr=.001, participation=1., split="dirichlet",
                alpha=.5, seed=42, device="auto", num_workers=0, threads=1,
                attack="lsb", attacker_fraction=.1, poison_fraction=.5, target_class=5,
                source_class=-1, aggregation="fedavg", byzantine_clients=1,
                trim_fraction=.1, krum_keep=1, trigger_characters=700,
                wanet_k=6, wanet_strength=.75, epsilon=.1, generator_checkpoint="",
                distributed=False, attention="none", attention_start_round=20,
                rankme_threshold=0., attention_samples=64, cam_samples=4,
                model_replacement=False, replacement_rounds="30,60,90",
                output_dir="results/lsb-cifar10-fedavg-seed42", backend="flower",
                registration_timeout=180., reply_timeout=3600.)


def validate(cfg):
    cfg = dict(cfg)
    choices = dict(dataset=DATASETS, attack=ATTACKS, aggregation=AGGREGATIONS,
                   split=("iid", "dirichlet"), attention=("none", "low", "high"),
                   backend=("flower", "sequential"))
    for name, options in choices.items():
        if cfg[name] not in options:
            raise ValueError(f"{name} must be one of {', '.join(options)}")
    for name in ("num_clients", "num_rounds", "local_epochs", "batch_size", "threads",
                 "trigger_characters", "wanet_k", "attention_start_round", "attention_samples", "cam_samples", "krum_keep"):
        if not isinstance(cfg[name], int) or isinstance(cfg[name], bool) or cfg[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("num_workers", "byzantine_clients", "seed"):
        if not isinstance(cfg[name], int) or cfg[name] < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for name in ("lr", "alpha", "registration_timeout", "reply_timeout"):
        if not math.isfinite(cfg[name]) or cfg[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("participation", "attacker_fraction", "poison_fraction", "trim_fraction", "epsilon"):
        if not math.isfinite(cfg[name]) or not 0 <= cfg[name] <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    if cfg["participation"] == 0 or cfg["trim_fraction"] >= .5:
        raise ValueError("Participation must be positive and trim_fraction below 0.5")
    for name in ("wanet_strength", "rankme_threshold"):
        if not math.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    classes = 10 if cfg["dataset"] == "cifar10" else 100
    if not 0 <= cfg["target_class"] < classes or not -1 <= cfg["source_class"] < classes:
        raise ValueError("Source/target class is outside the dataset's label range")
    if cfg["source_class"] == cfg["target_class"]:
        raise ValueError("Source and target classes must differ")
    attackers = int(cfg["num_clients"] * cfg["attacker_fraction"])
    if cfg["attack"] != "none" and (attackers < 1 or cfg["poison_fraction"] == 0):
        raise ValueError("An attack requires at least one malicious client and a positive poison fraction")
    if cfg["attention"] != "none" and cfg["attack"] not in ("lsb", "encoder"):
        raise ValueError("Attention placement is supported for LSB and Encoder")
    selected = max(1, math.ceil(cfg["num_clients"] * cfg["participation"]))
    if cfg["aggregation"] in ("krum", "multi-krum"):
        f = cfg["byzantine_clients"]
        if selected < 2 * f + 3:
            raise ValueError("Krum requires at least 2*f+3 participating clients")
        if cfg["aggregation"] == "multi-krum" and cfg["krum_keep"] > selected - f - 2:
            raise ValueError("Multi-Krum keeps at most n-f-2 participating updates")
    try:
        rounds = [int(value) for value in cfg["replacement_rounds"].split(",")]
        if not rounds or min(rounds) < 1:
            raise ValueError
    except (AttributeError, ValueError) as exc:
        raise ValueError("replacement_rounds must be positive round numbers separated by commas") from exc
    if cfg["device"] not in ("auto", "cpu", "mps", "cuda") and not (
        cfg["device"].startswith("cuda:") and cfg["device"][5:].isdigit()
    ):
        raise ValueError("device must be auto, cpu, mps or cuda[:index]")
    if cfg["backend"] == "flower" and cfg["device"].startswith("cuda:"):
        raise ValueError("Flower workers use remapped GPU indices; use --device cuda and CUDA_VISIBLE_DEVICES")
    return cfg


def resolve_device(name):
    import torch
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    return name
