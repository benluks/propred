from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import lightning as L
from tqdm import tqdm

# If this script is in scripts/ or train/, add repo root for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from train.pitch import ProsodyLit


# -------------------------
# Dataset (same pairing logic as training)
# -------------------------
class PitchDataset(Dataset):
    """
    Expects:
      bn_root/*.pt: 1D long tensor of indices (T,)
      feats_root/<same filename>.pt: dict with keys: 'f0', 'vuv' (optional for eval)
    """

    def __init__(
        self,
        bn_root: str | Path,
        feats_root: Optional[str | Path] = None,
        pattern: str = "*.pt",
    ):
        self.bn_root = Path(bn_root)
        self.files = sorted(self.bn_root.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No BN files matched {pattern} in {self.bn_root}")

        self.feats_root = Path(feats_root) if feats_root is not None else None
        if self.feats_root is not None and not self.feats_root.exists():
            raise FileNotFoundError(f"feats_root not found: {self.feats_root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        bn_path = self.files[i]
        indices = torch.load(bn_path, map_location="cpu").to(torch.long).flatten()

        feats = None
        if self.feats_root is not None:
            feats_path = self.feats_root / bn_path.name
            feats = torch.load(feats_path, map_location="cpu")

        return {
            "utt_id": bn_path.stem,
            "bn_path": str(bn_path),
            "indices": indices,
            "feats": feats,  # can be None
        }


# -------------------------
# Collate: pad indices -> x [B,T], mask [B,T]
# Keep utt_ids so we can write per-file outputs.
# -------------------------
@dataclass
class Batch:
    utt_id: List[str]
    x: torch.Tensor  # [B, T]
    mask: torch.Tensor  # [B, T]
    feats: List[Optional[dict]]


def collate_pitch_infer(batch: List[dict], pad_value: int = 0) -> Batch:
    utt_id = [b["utt_id"] for b in batch]
    xs = [b["indices"].to(torch.long).flatten() for b in batch]

    x = pad_sequence(xs, batch_first=True, padding_value=pad_value)  # [B, T]
    mask = (x != pad_value).to(torch.float32)  # [B, T]

    feats = [b.get("feats", None) for b in batch]
    return Batch(utt_id=utt_id, x=x, mask=mask, feats=feats)


@torch.inference_mode()
def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt", type=str, required=True, help="Path to Lightning checkpoint (.ckpt)"
    )
    p.add_argument(
        "--bn_root",
        type=str,
        default="/Users/ben/dev/propred/data/LJSpeech-1.1/emb_ids",
    )
    p.add_argument("--pattern", type=str, default="*.pt")
    p.add_argument(
        "--feats_root",
        type=str,
        default="/Users/ben/dev/propred/data/LJSpeech-1.1/feats",
        help="Optional: load cached feats for comparison",
    )

    p.add_argument(
        "--out_dir",
        type=str,
        default="/Users/ben/dev/propred/data/LJSpeech-1.1/predictions/version_2",
        help="Where to write predictions",
    )
    p.add_argument(
        "--save_mode", type=str, default="per_utt", choices=["per_utt", "single_file"]
    )

    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)  # macOS/MPS safest
    p.add_argument("--pad_value", type=int, default=0)

    p.add_argument("--device", type=str, default="mps", help="auto|cpu|mps|cuda")
    p.add_argument(
        "--return_probs", action="store_true", help="Also save vuv probabilities"
    )
    p.add_argument(
        "--f0_log_eps", type=float, default=1e-6, help="exp(log_f0) clamp floor"
    )

    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    lit = ProsodyLit.load_from_checkpoint(args.ckpt, map_location="cpu")
    lit.eval()

    # Pick device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    lit.to(device)

    # Data
    ds = PitchDataset(args.bn_root, feats_root=args.feats_root, pattern=args.pattern)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        collate_fn=lambda b: collate_pitch_infer(b, pad_value=args.pad_value),
        persistent_workers=(args.num_workers > 0),
    )

    all_outputs = []  # used only in single_file mode

    for batch in tqdm(dl):
        x = batch.x.to(device)  # [B,T]
        mask = batch.mask.to(device)  # [B,T]

        # forward: returns (pred_log_f0, pred_vuv_logits) with shape [B,T]
        pred_log_f0, pred_vuv_logits = lit.model(x, mask)

        # Convert to useful forms
        pred_vuv_prob = torch.sigmoid(pred_vuv_logits)
        pred_vuv = (pred_vuv_prob > 0.5).to(torch.float32)

        # Convert log-f0 -> f0 Hz (optionally)
        pred_f0 = torch.exp(pred_log_f0).clamp_min(args.f0_log_eps)

        # Trim padding per utterance using mask sum
        lengths = mask.sum(dim=-1).to(torch.long).tolist()

        for i, utt in enumerate(batch.utt_id):
            T = lengths[i]
            rec = {
                "utt_id": utt,
                "pred_log_f0": pred_log_f0[i, :T].detach().cpu(),
                "pred_f0": pred_f0[i, :T].detach().cpu(),
                "pred_vuv": pred_vuv[i, :T].detach().cpu(),
                "pred_vuv_logits": pred_vuv_logits[i, :T].detach().cpu(),
            }
            if args.return_probs:
                rec["pred_vuv_prob"] = pred_vuv_prob[i, :T].detach().cpu()

            # If feats were provided, optionally save for later scoring/plots
            feats = batch.feats[i]
            if feats is not None:
                # be conservative: just carry them through
                rec["ref_feats"] = feats

            if args.save_mode == "per_utt":
                torch.save(rec, out_dir / f"{utt}.pt")
            else:
                all_outputs.append(rec)

    if args.save_mode == "single_file":
        torch.save(all_outputs, out_dir / "predictions.pt")

    print(f"Done. Wrote predictions to: {out_dir}")


if __name__ == "__main__":
    main()
