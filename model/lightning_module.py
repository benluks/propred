import torch
import torch.nn.functional as F
import lightning as L

from model.duration_predictor import DurationPredictor
from utils.collate import Batch


class DurationRegressor(L.LightningModule):
    def __init__(
        self,
        duration_predictor: DurationPredictor,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        loss_type: str = "l1",  # "l1" or "mse" or "huber"
        do_log=False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.do_log = do_log

        self.model = duration_predictor
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
