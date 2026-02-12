from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import List, Tuple

import lightning as L
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import sys
from pathlib import Path

from model.predictor import DurationPredictor

# add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


from data.ljspeech import DurationsDataset
from model.conv_decoder import ConvDecoder
from utils.config import load_config


@dataclass
class Batch:
    values: torch.Tensor
    durations: torch.Tensor
    lengths: torch.Tensor
    mask: torch.Tensor


def collate_values_durations(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_value: int = 0,
) -> Batch:

    values_list, durs_list = zip(*batch)

    values = pad_sequence(values_list, batch_first=True, padding_value=pad_value)
    durations = pad_sequence(durs_list, batch_first=True, padding_value=pad_value)

    mask = (values != pad_value).float()
    lengths = mask.sum(-1).to(torch.long)

    return Batch(
        values=values.to(torch.long),
        durations=durations.to(torch.long),
        lengths=lengths,
        mask=mask,
    )


# -------------------------
# Lightning module (Option A: regression to durations)
# -------------------------
class DurationRegressor(L.LightningModule):
    def __init__(
        self,
        embeddings_path: str | Path,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        filter_channels: int = 256,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        loss_type: str = "l1",  # "l1" or "mse" or "huber"
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DurationPredictor(
            embeddings_path=embeddings_path,
            output_dim=1,
            filter_channels=filter_channels,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_type = loss_type

    def forward(self, values, mask):
        return self.model(values, mask)

    def training_step(self, batch: Batch, batch_idx: int):
        values = batch.values.to(self.device)  # (B, R)
        durations = batch.durations.to(self.device)  # (B, R)
        mask = batch.mask.to(self.device)  # (B, R)

        pred = self.model(values, mask)  # (B, R) float
        target = durations.float()

        if self.loss_type == "mse":
            loss_per = (pred - target) ** 2
        elif self.loss_type == "huber":
            loss_per = F.smooth_l1_loss(pred, target, reduction="none")
        else:  # l1
            loss_per = (pred - target).abs()

        loss = (loss_per * mask).sum() / (mask.sum().clamp_min(1.0))

        # some useful logging
        with torch.no_grad():
            mae = ((pred - target).abs() * mask).sum() / mask.sum().clamp_min(1.0)
            self.log("train/loss", loss, prog_bar=True)
            self.log("train/mae", mae, prog_bar=True)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return opt


# -------------------------
# Main: train 20 epochs, no val
# -------------------------
def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()

    cfg = load_config(args.config)
    ds = DurationsDataset(
        cfg.data.root,
        pattern=cfg.data.pattern,
    )

    dl = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=partial(collate_values_durations, pad_value=cfg.data.pad_value),
        persistent_workers=(cfg.data.num_workers > 0),
    )

    lit = DurationRegressor(
        embeddings_path=cfg.model.embeddings_path,
        lr=cfg.optim.lr,
        weight_decay=cfg.optim.weight_decay,
        filter_channels=cfg.model.filter_channels,
        kernel_size=cfg.model.kernel_size,
        p_dropout=cfg.model.dropout,
        loss_type=cfg.optim.loss,
    )

    trainer = L.Trainer(
        max_epochs=cfg.optim.epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=getattr(cfg.trainer, "log_every_n_steps", 50),
    )

    trainer.fit(lit, train_dataloaders=dl)


if __name__ == "__main__":
    main()
