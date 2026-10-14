"""Run-owned cache paths, independent of Flower loader working directories."""
import os
from pathlib import Path


def cache_path(relative):
    return str(Path(os.environ.get("FIBA_RUN_ROOT", "..")) / relative)
