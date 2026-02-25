from dataclasses import dataclass
from functools import partial
from pathlib import Path
import sys
from typing import List, Optional, Tuple
from hyperpyyaml import load_hyperpyyaml

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.predictor import DurationPredictor
from dataset import DurationsDataset


@dataclass
class Batch:
    values: torch.Tensor
    durations: torch.Tensor
    lengths: torch.Tensor
    mask: torch.Tensor
    spk_ids: Optional[torch.Tensor] = None


def collate_values_durations(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    pad_value: int = 0,
) -> Batch:

    if len(batch[0]) == 2:
        values_list, durs_list = zip(*batch)
        spk_ids_list = None
    else:
        values_list, durs_list, spk_ids_list = zip(*batch)

    values = pad_sequence(values_list, batch_first=True, padding_value=pad_value)
    durations = pad_sequence(durs_list, batch_first=True, padding_value=pad_value)
    spk_ids = (
        torch.tensor(spk_ids_list).to(torch.long) if spk_ids_list is not None else None
    )

    mask = (values != pad_value).float()
    lengths = mask.sum(-1).to(torch.long)

    return Batch(
        values=values.to(torch.long),
        durations=durations.to(torch.long),
        spk_ids=spk_ids,
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
        do_log=False,
        padding_idx=0,
        n_layers=1,
        use_spk_id=False,
        spk_id_path: Optional[str | Path] = None,
        spk_embedding_dim: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.do_log = do_log
        self.use_spk_id = use_spk_id
        if self.use_spk_id:
            if spk_embedding_dim is None:
                raise ValueError(
                    "spk_embedding_dim must be provided if use_spk_id is True"
                )
            spk_id_map = torch.load(spk_id_path)
            num_speakers = len(spk_id_map)

        self.model = DurationPredictor(
            embeddings_path=embeddings_path,
            output_dim=1,
            filter_channels=filter_channels,
            kernel_size=kernel_size,
            p_dropout=p_dropout,
            padding_idx=padding_idx,
            n_layers=n_layers,
            do_log=self.do_log,
            use_spk_id=use_spk_id,
            num_speakers=num_speakers if self.use_spk_id else None,
            spk_embedding_dim=spk_embedding_dim if self.use_spk_id else None,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.loss_type = loss_type

    def forward(self, values, mask):
        return self.model(values, mask)

    def _compute_loss_and_mae(self, batch: Batch):
        pred = self.model(batch.values, batch.mask, batch.spk_ids)  # (B, R)
        target = batch.durations.float()

        if self.do_log:
            target = target.clone()
            target[batch.mask.bool()] = torch.log1p(target[batch.mask.bool()])

        if self.loss_type == "mse":
            loss_per = (pred - target) ** 2
        elif self.loss_type == "huber":
            loss_per = F.smooth_l1_loss(pred, target, reduction="none")
        else:
            loss_per = (pred - target).abs()

        denom = batch.mask.sum().clamp_min(1.0)
        loss = (loss_per * batch.mask).sum() / denom
        mae = ((pred - target).abs() * batch.mask).sum() / denom

        mae_lin = None
        if self.do_log:
            pred_lin = torch.expm1(pred).clamp_min(0.0)
            mae_lin = (
                (pred_lin - batch.durations.float()).abs() * batch.mask
            ).sum() / denom

        return loss, mae, mae_lin

    def training_step(self, batch: Batch, batch_idx: int):
        loss, mae, mae_lin = self._compute_loss_and_mae(batch)
        self.log_dict({"train/loss": loss, "train/mae": mae}, prog_bar=True)
        if mae_lin is not None:
            self.log("train/mae_linear", mae_lin, prog_bar=False)
        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        loss, mae, mae_lin = self._compute_loss_and_mae(batch)
        self.log_dict({"val/loss": loss, "val/mae": mae}, prog_bar=True)
        if mae_lin is not None:
            self.log("val/mae_linear", mae_lin, prog_bar=False)

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

    with open(args.config, "r") as f:
        cfg = load_hyperpyyaml(f)

    train_ds_index_path = cfg["data"].pop("train_index", None)
    val_ds_index_path = cfg["data"].pop("val_index", None)

    train_ds = DurationsDataset(**cfg["data"], index_path=train_ds_index_path)

    # dl_kwargs = cfg.dataloader.__dict__
    # pad_value = dl_kwargs.pop("pad_value")
    dl = DataLoader(
        train_ds,
        shuffle=True,
        pin_memory=True,
        batch_size=cfg["dataloader"]["batch_size"],
        num_workers=cfg["dataloader"]["num_workers"],
        persistent_workers=(cfg["dataloader"]["num_workers"] > 0),
        collate_fn=partial(
            collate_values_durations, pad_value=cfg["dataloader"]["pad_value"]
        ),
    )

    val_dl = None
    if val_ds_index_path is not None:
        val_ds = DurationsDataset(**cfg["data"], index_path=val_ds_index_path)
        val_dl = DataLoader(
            val_ds,
            shuffle=False,
            pin_memory=True,
            batch_size=cfg["dataloader"]["batch_size"],
            num_workers=cfg["dataloader"]["num_workers"],
            persistent_workers=(cfg["dataloader"]["num_workers"] > 0),
            collate_fn=partial(
                collate_values_durations, pad_value=cfg["dataloader"]["pad_value"]
            ),
        )

    lit = DurationRegressor(**cfg["model"])

    trainer_kwargs = cfg["trainer"]
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
        train_dataloaders=dl,
        val_dataloaders=val_dl,
    )


if __name__ == "__main__":
    main()
