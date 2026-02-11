"""
Train ProsodyPredictor with Hugging Face Trainer on streaming LibriTTS.

Assumes your batch dict from collate() contains:
  wav:            [B, Tw]
  frame_pad_mask: [B, Tf] (bool) True=pad
  speaker_id:     [B] (long)
  f0:             [B, Tf]
  vuv:            [B, Tf] (float 0..1)
  energy:         [B, Tf]

And your model forward:
  pred_f0, pred_vuv_logits, pred_energy = model(wavs, speaker_id, x_mask)

Run:
  python train_prosody_trainer.py \
    --speaker_dict_json f0_stats/speaker_dict.json \
    --max_steps 50000 \
    --output_dir exp/prosody

Notes:
- Iterable/streaming datasets require max_steps (epochs are not well-defined).
- We set remove_unused_columns=False so Trainer won’t drop your custom keys.
"""

import argparse
import json
from dataclasses import dataclass
import logging
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments, set_seed

from data.data import collate, preprocess
from model.predictor import ProsodyPredictor

# ---- import your code ----
# from your_model_file import ProsodyPredictor
# from your_data_file import preprocess, collate
# If they're in this file, just ensure they're defined above.

# --------------------------
# Loss helpers
# --------------------------


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mask = mask.to(dtype=x.dtype)
    return (x * mask).sum() / mask.sum().clamp_min(eps)


def masked_mse(
    pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return masked_mean((pred - tgt) ** 2, mask)


def masked_bce_with_logits(
    logits: torch.Tensor, tgt01: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    per = F.binary_cross_entropy_with_logits(logits, tgt01, reduction="none")
    return masked_mean(per, mask)


# --------------------------
# Custom Trainer
# --------------------------


class ProsodyTrainer(Trainer):
    """
    Custom compute_loss so Trainer can optimize your multi-head outputs.
    """

    def __init__(
        self,
        *args,
        w_f0: float = 1.0,
        w_vuv: float = 1.0,
        w_energy: float = 1.0,
        supervise_f0_only_voiced: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.w_f0 = w_f0
        self.w_vuv = w_vuv
        self.w_energy = w_energy
        self.supervise_f0_only_voiced = supervise_f0_only_voiced

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        inputs: dict from collate()
        """
        # Move tensors to correct device (Trainer usually does this, but keep it explicit-friendly)
        wav = inputs["wav"]
        spk_id = inputs["speaker_id"]
        f0_tgt = inputs["f0"]
        vuv_tgt = inputs["vuv"]
        e_tgt = inputs["energy"]
        frame_pad_mask = inputs["frame_pad_mask"]  # True=pad

        # x_mask is 1 for valid frames
        x_mask = (~frame_pad_mask).unsqueeze(1).to(dtype=wav.dtype, device=wav.device)

        # Forward
        pred_f0, pred_vuv, pred_e = model(wav, spk_id, x_mask)

        # Build masks in [B, T]
        mask = x_mask.squeeze(1).to(dtype=pred_f0.dtype)

        # again, trim to shortest length; f0, vuv, and e are reliably the same shape
        min_T = min(pred_f0.shape[-1], pred_vuv.shape[-1], pred_e.shape[-1])

        pred_f0 = pred_f0[..., :min_T]
        f0_tgt = f0_tgt[..., :min_T]
        pred_vuv = pred_vuv[..., :min_T]
        vuv_tgt = vuv_tgt[..., :min_T]
        pred_e = pred_e[..., :min_T]
        e_tgt = e_tgt[..., :min_T]

        mask = mask[..., :min_T]

        # F0 supervision: usually best only on voiced frames
        if self.supervise_f0_only_voiced:
            f0_mask = mask * vuv_tgt.to(mask.dtype)  # vuv_tgt in [0,1]
        else:
            f0_mask = mask

        loss_f0 = masked_mse(pred_f0, f0_tgt, f0_mask)
        loss_vuv = masked_bce_with_logits(pred_vuv, vuv_tgt, mask)
        loss_e = masked_mse(pred_e, e_tgt, mask)

        loss = self.w_f0 * loss_f0 + self.w_vuv * loss_vuv + self.w_energy * loss_e

        # Log scalars
        self.log(
            {
                "loss/total": loss.detach().cpu().item(),
                "loss/f0": loss_f0.detach().cpu().item(),
                "loss/vuv": loss_vuv.detach().cpu().item(),
                "loss/energy": loss_e.detach().cpu().item(),
                "mask/valid_frames": mask.sum().detach().cpu().item(),
            }
        )

        return (loss, (pred_f0, pred_vuv, pred_e)) if return_outputs else loss


# --------------------------
# Data collator adapter
# --------------------------


@dataclass
class CollatorWrapper:
    """
    Wrap your existing collate(batch)->dict so it can be used as Trainer's data_collator.
    """

    collate_fn: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.collate_fn(features)


# --------------------------
# Main
# --------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--speaker_dict_json", type=str, default="f0_stats/speaker_dict.json"
    )
    ap.add_argument("--dataset_name", type=str, default="mythicinfinity/libritts")
    ap.add_argument("--dataset_config", type=str, default="clean")
    ap.add_argument("--train_split", type=str, default="train.clean.100")
    ap.add_argument("--eval_split", type=str, default="dev.clean")
    ap.add_argument("--do_eval", action="store_true")

    # Training
    ap.add_argument("--output_dir", type=str, required=False)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--max_steps", type=int, default=50_000)
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--eval_steps", type=int, default=1000)

    ap.add_argument("--per_device_train_batch_size", type=int, default=16)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=16)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--dataloader_num_workers", type=int, default=10)

    # Loss weights
    ap.add_argument("--w_f0", type=float, default=1.0)
    ap.add_argument("--w_vuv", type=float, default=1.0)
    ap.add_argument("--w_energy", type=float, default=1.0)
    ap.add_argument("--supervise_f0_only_voiced", action="store_true")

    # Model hyperparams (example defaults; set to yours)
    ap.add_argument("--rep_dim", type=int, default=256)  # update to your BN dim
    ap.add_argument("--spk_dim", type=int, default=128)
    ap.add_argument("--filter_channels", type=int, default=256)
    ap.add_argument("--kernel_size", type=int, default=3)
    ap.add_argument("--p_dropout", type=float, default=0.1)
    ap.add_argument("--rep_proj_dim", type=int, default=128)  # 0 => None

    # Precision
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    speaker_dict = json.load(open(args.speaker_dict_json))
    n_speakers = len(speaker_dict.keys())

    logging.info(
        f"Loading streaming datasets for {args.train_split} and {args.eval_split}"
    )
    ds = load_dataset(args.dataset_name, args.dataset_config, streaming=True)

    train_ds = ds[args.train_split].map(
        preprocess, fn_kwargs={"speaker_dict": speaker_dict}
    )

    if args.do_eval:
        eval_ds = ds[args.eval_split].map(
            preprocess, fn_kwargs={"speaker_dict": speaker_dict}
        )
    else:
        eval_ds = None

    # ---- Build model ----
    rep_proj_dim = None if args.rep_proj_dim == 0 else args.rep_proj_dim

    model = ProsodyPredictor(
        rep_dim=args.rep_dim,
        n_speakers=n_speakers,
        spk_dim=args.spk_dim,
        filter_channels=args.filter_channels,
        kernel_size=args.kernel_size,
        p_dropout=args.p_dropout,
        vuv_output="logits",
        rep_proj_dim=rep_proj_dim,
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logging.info("Using {device} as training device")

    model.to(device)

    # ---- Training args ----
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps" if args.do_eval else "no",
        eval_steps=args.eval_steps if args.do_eval else None,
        report_to=["tensorboard"],  # or ["wandb"] if you use it
        save_total_limit=2,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = ProsodyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=CollatorWrapper(collate),
        w_f0=args.w_f0,
        w_vuv=args.w_vuv,
        w_energy=args.w_energy,
        supervise_f0_only_voiced=args.supervise_f0_only_voiced,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
