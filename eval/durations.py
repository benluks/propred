import argparse
from functools import partial
import os
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from clustering.generate_bns import load_bn_extractor
from data.ljspeech import DurationsDataset
from data.utils import warp_f0_by_durations
from train import DurationRegressor
from utils.run_lengths import (
    expand_batch,
    expand_by_duration,
    perturb_durations_logjitter,
)

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


@torch.inference_mode
class Converter(nn.Module):
    def __init__(
        self,
        dp_ckpt: str,
        target_speaker: str,
        device: str,
        embedding_path=None,
        stochastic=True,
        stoch_kwargs={},
    ):
        super().__init__()

        self.device = device
        self.model = load_bn_extractor().to(device)
        self.model.eval()
        self.stochastic = stochastic

        if self.stochastic:
            self.duration_predictor = partial(
                perturb_durations_logjitter, kwargs=stoch_kwargs
            )
        else:
            self.duration_predictor = DurationRegressor.load_from_checkpoint(
                dp_ckpt
            ).model.to(device)
            if embedding_path:
                embedding = torch.load(embedding_path)
                self.duration_predictor.embedding = nn.Embedding.from_pretrained(
                    embedding
                ).to(device)
            self.duration_predictor.eval()
        self.target_speaker = target_speaker

    @torch.no_grad
    def forward(
        self,
        wav,
        orig_durations,
        C: float = 1.0,
        values=None,
        values_mask=None,
        target_speaker=None,
        f0=None,
        dp_kwargs={},
    ):
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        if values is None:
            values = self.model.get_bn(values)
            values_mask = torch.ones_like(values)
        values, values_mask, wav, orig_durations = (
            values.to(self.device),
            values_mask.to(self.device),
            wav.to(self.device),
            orig_durations.to(self.device),
        )
        if values.ndim == 1:
            values = values.unsqueeze(0)

        pred_durations = self.duration_predictor(values, mask=values_mask, **dp_kwargs)
        if self.duration_predictor.do_log:
            pred_durations = torch.expm1(pred_durations).round().clamp_min(1).long()
        durations = torch.round(C * pred_durations + (1 - C) * orig_durations)

        bn_ids = expand_batch(values, durations)
        bn = self.duration_predictor.embedding(bn_ids).transpose(1, 2)

        if not f0:
            f0 = self.model.get_f0(wav)

        if f0.numel() != orig_durations.sum():
            f0 = F.interpolate(f0.unsqueeze(0), orig_durations.sum())

        warped_f0 = warp_f0_by_durations(
            f0, orig_durations, durations, in_log_domain=False
        ).reshape(1, 1, -1)

        spk_id = self.model.get_spk_id(
            wav, target_speaker if target_speaker else self.target_speaker
        )
        return self.model._forward(warped_f0, bn, spk_id).squeeze(0)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Convert a dataset split and write outputs under <dataset_root>/converted/"
            "<split>/t=<target_speaker>_C=<C>/..."
        )
    )

    # Dataset + iteration
    p.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Dataset root (e.g., /path/to/LibriTTS).",
    )
    p.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split name (e.g., test-clean, dev-clean).",
    )
    p.add_argument(
        "--wavs-folder",
        type=str,
        default=None,
        help=(
            "Folder under dataset-root that contains wavs for this split. "
            "Defaults to --split."
        ),
    )
    p.add_argument(
        "--wavs-pattern",
        type=str,
        default="*/*/*.wav",
        help="Glob pattern (relative to <dataset-root>/<wavs-folder>) for wav discovery.",
    )
    p.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap for number of items to process (debug).",
    )

    # Conversion params
    p.add_argument(
        "--target-speaker",
        type=str,
        required=True,
        help="Target speaker ID used by the BN extractor model (string).",
    )
    p.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Interpolation factor between predicted durations (1.0) and original (0.0).",
    )
    p.add_argument(
        "--dp-ckpt",
        type=str,
        required=True,
        help="Path to duration predictor Lightning checkpoint (.ckpt).",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="The max size of run to smooth out (passed as `kill_singletons` to durations dataset)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on.",
    )

    # Output layout control
    p.add_argument(
        "--out-rootname",
        type=str,
        default="converted",
        help='Name of output folder created in dataset root (default: "converted").',
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Output sample rate for saved wavs.",
    )
    return p


def main():
    args = build_argparser().parse_args()

    dataset_root: Path = args.dataset_root
    split: str = args.split
    wavs_folder: str = args.wavs_folder or split

    # <dataset_root>/converted/<split>/<target_speaker>/C=<C>/

    out_name = f"t={str(args.target_speaker)}_C={args.C:g}"
    if args.smooth:
        out_name += f"smooth={str(args.smooth)}"

    out_dir = dataset_root / args.out_rootname / split / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = DurationsDataset(
        str(dataset_root),
        split=split,
        wavs_folder=wavs_folder,
        wavs_pattern=args.wavs_pattern,
        kill_singletons=args.smooth,
    )
    embedding_path = dataset_root / f"embeddings{f'/{split if split else str()}'}.pt"
    converter = Converter(
        dp_ckpt=args.dp_ckpt,
        target_speaker=args.target_speaker,
        device=args.device,
        embedding_path=embedding_path,
    )

    n = len(ds) if args.max_items is None else min(len(ds), args.max_items)

    for i in tqdm(range(n), desc=f"Converting audio into {str(out_dir)}"):
        values, orig_durations, wav, utt_id = ds[i]
        values = values.unsqueeze(0)
        mask = torch.ones_like(values)

        y = converter(values, mask, wav, orig_durations, C=args.C).to("cpu")
        out_path = out_dir / f"{utt_id}.wav"

        torchaudio.save(str(out_path), y, args.sample_rate)


if __name__ == "__main__":
    main()
