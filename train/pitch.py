from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.ljspeech import PitchDataset
from model.predictor import ProsodyPredictor


@dataclass
class Batch:
    x: torch.Tensor  # [B, Tmax] long
    mask: torch.Tensor  # [B, Tmax] float
    f0: torch.Tensor  # [B, Tmax] float
    vuv: torch.Tensor  # [B, Tmax] float (0/1)
    energy: Optional[torch.Tensor] = None


def collate_pitch(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]], pad_value: int = 0
) -> Batch:
    xs, feats_list = zip(*batch)

    xs = [x.to(torch.long).flatten() for x in xs]
    x_pad = pad_sequence(xs, batch_first=True, padding_value=pad_value)  # [B, Tmax]
    mask = (x_pad != pad_value).to(torch.float32)  # [B, Tmax]

    f0s = [torch.as_tensor(f["f0"]).to(torch.float32).flatten() for f in feats_list]
    vuvs = [torch.as_tensor(f["vuv"]).to(torch.float32).flatten() for f in feats_list]

    f0_pad = pad_sequence(f0s, batch_first=True, padding_value=pad_value)
    vuv_pad = pad_sequence(vuvs, batch_first=True, padding_value=pad_value)

    return Batch(x=x_pad, mask=mask, f0=f0_pad, vuv=vuv_pad)


# -------------------------
# LightningModule
# -------------------------
class ProsodyLit(L.LightningModule):
    def __init__(
        self,
        embeddings_path: str | Path,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        filter_channels: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.1,
        rep_proj_dim: Optional[int] = None,
        f0_loss: str = "l1",  # "l1" | "huber" | "mse"
        vuv_pos_weight: Optional[float] = None,  # helpful if voiced/unvoiced imbalance
        f0_log_eps: float = 1e-6,
        loss_w_f0: float = 1.0,
        loss_w_vuv: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = ProsodyPredictor(
            embeddings_path=embeddings_path,
            filter_channels=filter_channels,
            kernel_size=kernel_size,
            p_dropout=dropout,
            rep_proj_dim=rep_proj_dim,
            heads=["f0", "vuv"],
        )

        self.lr = lr
        self.weight_decay = weight_decay
        self.f0_loss = f0_loss
        self.f0_log_eps = f0_log_eps
        self.loss_w_f0 = loss_w_f0
        self.loss_w_vuv = loss_w_vuv

        if vuv_pos_weight is not None:
            self.register_buffer(
                "vuv_pos_weight", torch.tensor([vuv_pos_weight], dtype=torch.float32)
            )
        else:
            self.vuv_pos_weight = None

    def training_step(self, batch: Batch, batch_idx: int):
        x = batch.x.to(self.device)  # [B, T]
        mask = batch.mask.to(self.device)  # [B, T]
        f0 = batch.f0.to(self.device)  # [B, T]
        vuv = batch.vuv.to(self.device)  # [B, T] (0/1)

        # model outputs:
        pred_f0, pred_vuv_logits = self.model(x, mask)  # both [B, T]

        # --- VUV loss (all valid frames) ---
        if self.vuv_pos_weight is not None:
            vuv_loss_per = F.binary_cross_entropy_with_logits(
                pred_vuv_logits, vuv, reduction="none", pos_weight=self.vuv_pos_weight
            )
        else:
            vuv_loss_per = F.binary_cross_entropy_with_logits(
                pred_vuv_logits, vuv, reduction="none"
            )
        vuv_loss = (vuv_loss_per * mask).sum() / mask.sum().clamp_min(1.0)

        # --- log-f0 loss (voiced frames only) ---
        # voiced_mask: valid & voiced
        voiced_mask = mask * (vuv > 0.5).to(mask.dtype)

        # target log-f0 (avoid log(0))
        log_f0_tgt = torch.log(f0.clamp_min(self.f0_log_eps))

        if self.f0_loss == "mse":
            f0_loss_per = (pred_f0 - log_f0_tgt) ** 2
        elif self.f0_loss == "huber":
            f0_loss_per = F.smooth_l1_loss(pred_f0, log_f0_tgt, reduction="none")
        else:
            f0_loss_per = (pred_f0 - log_f0_tgt).abs()

        f0_loss = (f0_loss_per * voiced_mask).sum() / voiced_mask.sum().clamp_min(1.0)

        loss = self.loss_w_vuv * vuv_loss + self.loss_w_f0 * f0_loss

        # metrics
        with torch.no_grad():
            pred_vuv = (torch.sigmoid(pred_vuv_logits) > 0.5).to(vuv.dtype)
            vuv_acc = ((pred_vuv == vuv) * mask).sum() / mask.sum().clamp_min(1.0)

            self.log("train/loss", loss, prog_bar=True)
            self.log("train/vuv_loss", vuv_loss, prog_bar=False)
            self.log("train/f0_loss", f0_loss, prog_bar=False)
            self.log("train/vuv_acc", vuv_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )


# -------------------------
# Main
# -------------------------
def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--bn_root",
        type=str,
        default="/Users/ben/dev/propred/data/LJSpeech-1.1/emb_ids",
    )
    p.add_argument(
        "--feats_root",
        type=str,
        default="/Users/ben/dev/propred/data/LJSpeech-1.1/feats",
    )
    p.add_argument("--pattern", type=str, default="*.pt")
    p.add_argument(
        "--embeddings_path",
        type=str,
        default="/Users/ben/dev/propred/data/LJSpeech-1.1/embeddings.pt",
    )

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--num_workers", type=int, default=0
    )  # macOS/MPS: keep 0 to avoid shm issues
    p.add_argument("--pad_value", type=int, default=0)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0)

    p.add_argument("--filter_channels", type=int, default=256)
    p.add_argument("--kernel_size", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--rep_proj_dim", type=int, default=None)

    p.add_argument("--f0_loss", type=str, default="l1", choices=["l1", "huber", "mse"])
    p.add_argument("--vuv_pos_weight", type=float, default=None)
    p.add_argument("--loss_w_f0", type=float, default=1.0)
    p.add_argument("--loss_w_vuv", type=float, default=1.0)

    p.add_argument("--accelerator", type=str, default="mps")
    p.add_argument("--experiment_name", type=str, default="pitch")
    p.add_argument("--devices", type=str, default="auto")
    p.add_argument("--precision", type=str, default="32")

    args = p.parse_args()

    ds = PitchDataset(args.bn_root, args.feats_root, pattern=args.pattern)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,  # MPS: ignore pinning
        collate_fn=lambda b: collate_pitch(b, pad_value=args.pad_value),
        persistent_workers=(args.num_workers > 0),
    )

    lit = ProsodyLit(
        embeddings_path=args.embeddings_path,
        lr=args.lr,
        weight_decay=args.weight_decay,
        filter_channels=args.filter_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        rep_proj_dim=args.rep_proj_dim,
        f0_loss=args.f0_loss,
        vuv_pos_weight=args.vuv_pos_weight,
        loss_w_f0=args.loss_w_f0,
        loss_w_vuv=args.loss_w_vuv,
    )

    # ckpt_cb = L.pytorch.callbacks.ModelCheckpoint(
    #     dirpath=args.save_dir,
    #     filename="prosody-{epoch:02d}-{train_loss:.4f}",
    #     monitor="train/loss",
    #     mode="min",
    #     save_top_k=1,
    #     save_last=True,
    # )

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=TensorBoardLogger(
            save_dir="lightning_logs",
            name=args.experiment_name,
        ),
        precision=args.precision,
        callbacks=[
            # ckpt_cb
        ],
        log_every_n_steps=50,
    )

    trainer.fit(lit, train_dataloaders=dl)


if __name__ == "__main__":
    main()
