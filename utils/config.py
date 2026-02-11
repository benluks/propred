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
