from pathlib import Path

import torch
import yaml
from types import SimpleNamespace


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(
        **{
            k: SimpleNamespace(**v) if isinstance(v, dict) else v
            for k, v in cfg.items()
        }
    )


def load_spk_id_map(input) -> dict:
    if isinstance(input, dict):
        return input
    elif isinstance(input, (str, Path)):  # noqa: F821
        path = Path(input)
        if not path.exists():
            raise FileNotFoundError(f"spk_id_path not found: {path}")
        return torch.load(path)  # noqa: F821
    else:
        raise ValueError(
            "spk_id_path must be a dict or a path to a file containing a dict"
        )
