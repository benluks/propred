from functools import partial
from pathlib import Path
import sys
from hyperpyyaml import load_hyperpyyaml

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.lightning_module import DurationRegressor


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()

    with open(args.config, "r") as f:
        hparams = load_hyperpyyaml(f)

    lit = DurationRegressor(**hparams["model"])

    trainer_kwargs = hparams["trainer"]
    version = trainer_kwargs.pop("version")

    trainer = L.Trainer(
        logger=TensorBoardLogger(
            save_dir="lightning_logs", name="duration", version=version
        ),
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                monitor="train/loss",
                mode="min",
                save_top_k=1,
                save_last=True,
            )
        ],
        **trainer_kwargs,
    )

    trainer.fit(
        lit,
        **hparams["dataloaders"],
    )


if __name__ == "__main__":
    main()
