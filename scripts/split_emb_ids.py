#!/usr/bin/env python3
"""
Speaker-stratified train/val split for emb_id .pt files.

Expected layout:
  {root}/emb_ids/
  {root}/emb_ids/{split}/   (optional)

Files are named like: speaker-chapter-utterance.pt (e.g., 103-1240-0009.pt)

Creates:
  - train.txt, val.txt (lists of file paths)
Optionally:
  - train/ and val/ dirs with symlinks or copies of the files
"""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


def parse_speaker_id(stem: str) -> str:
    """
    stem: filename without suffix, e.g. "103-1240-0009"
    returns speaker id, e.g. "103"
    """
    # safest: split on first '-'
    if "-" not in stem:
        raise ValueError(
            f"Filename stem '{stem}' doesn't contain '-' to parse speaker id."
        )
    return stem.split("-", 1)[0]


def round_half_up(x: float) -> int:
    """
    Python's round() is bankers rounding; this rounds halves up instead.
    """
    return int(math.floor(x + 0.5))


@dataclass
class SplitResult:
    train: List[Path]
    val: List[Path]
    per_speaker_counts: Dict[str, Tuple[int, int]]  # speaker -> (n_total, n_val)


def make_split(
    files: List[Path],
    val_pct: float,
    seed: int,
    min_val_per_speaker: int,
    allow_empty_val_for_small_speakers: bool,
) -> SplitResult:
    by_spk: Dict[str, List[Path]] = defaultdict(list)
    for p in files:
        spk = parse_speaker_id(p.stem)
        by_spk[spk].append(p)

    rng = random.Random(seed)

    train: List[Path] = []
    val: List[Path] = []
    per_counts: Dict[str, Tuple[int, int]] = {}

    for spk, spk_files in sorted(by_spk.items(), key=lambda kv: kv[0]):
        spk_files = list(spk_files)
        rng.shuffle(spk_files)

        n = len(spk_files)
        n_val = round_half_up(val_pct * n)

        if not allow_empty_val_for_small_speakers:
            if n > 0:
                n_val = max(n_val, min_val_per_speaker)

        # never exceed n-1 unless you truly want all in val (usually undesirable)
        if n >= 2:
            n_val = min(n_val, n - 1)
        else:
            # n == 1: either val gets 0 or 1 depending on allow_empty... + min_val...
            if allow_empty_val_for_small_speakers:
                n_val = min(n_val, 1)
            else:
                n_val = 0  # keep the single example in train to avoid empty train for that speaker

        val_spk = spk_files[:n_val]
        train_spk = spk_files[n_val:]

        val.extend(val_spk)
        train.extend(train_spk)
        per_counts[spk] = (n, n_val)

    # Final shuffle across speakers (optional). Keep deterministic.
    rng.shuffle(train)
    rng.shuffle(val)

    return SplitResult(train=train, val=val, per_speaker_counts=per_counts)


def write_list(
    paths: List[Path], out_path: Path, base_dir: Path, relative: bool
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in paths:
            s = str(p.relative_to(base_dir)) if relative else str(p)
            f.write(s + "\n")


def materialize(
    paths: List[Path],
    dest_dir: Path,
    method: str,
    overwrite: bool,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    for src in paths:
        dst = dest_dir / src.name
        if dst.exists() or dst.is_symlink():
            if overwrite:
                if dst.is_dir():
                    shutil.rmtree(dst)
                else:
                    dst.unlink()
            else:
                raise FileExistsError(f"{dst} already exists (use --overwrite).")

        if method == "symlink":
            # use absolute to avoid surprises if you move the output folder
            os.symlink(src.resolve(), dst)
        elif method == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unknown method: {method}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create a speaker-stratified validation split (10% per speaker, rounded)."
    )
    ap.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root containing emb_ids/ (e.g., data/LibriSpeech).",
    )
    ap.add_argument(
        "--split",
        type=str,
        default=None,
        help="Optional split subdir under emb_ids/ (e.g., train-clean-100). If omitted, uses emb_ids/ directly.",
    )
    ap.add_argument(
        "--val_pct",
        type=float,
        default=0.10,
        help="Validation percentage per speaker (default 0.10).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="RNG seed for deterministic splitting.",
    )
    ap.add_argument(
        "--glob",
        type=str,
        default="*.pt",
        help="Glob pattern to match files (default *.pt).",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Where to write outputs. Default: {root}/emb_ids/_splits/{split or 'ALL'}",
    )
    ap.add_argument(
        "--absolute_paths",
        action="store_true",
        help="Write absolute paths into train.txt/val.txt (default: relative to emb_ids[/split]).",
    )
    ap.add_argument(
        "--min_val_per_speaker",
        type=int,
        default=1,
        help="Minimum #val utterances per speaker when not allowing empty val (default 1).",
    )
    ap.add_argument(
        "--allow_empty_val_for_small_speakers",
        action="store_true",
        help="Allow speakers to contribute 0 val utterances if rounding yields 0 (default: force at least --min_val_per_speaker when possible).",
    )
    ap.add_argument(
        "--materialize",
        choices=["none", "symlink", "copy"],
        default="none",
        help="Optionally create train/ and val/ dirs with symlinks or copies.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing train/val dirs when using --materialize symlink/copy.",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="Print per-speaker counts and totals.",
    )

    args = ap.parse_args()

    emb_dir = args.root / "emb_ids"
    base_dir = emb_dir / args.split if args.split else emb_dir
    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    files = sorted(base_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No files found in {base_dir} matching glob '{args.glob}'")

    split_tag = args.split if args.split else "ALL"
    out_dir = args.out_dir if args.out_dir else (emb_dir / "_splits" / split_tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    res = make_split(
        files=files,
        val_pct=args.val_pct,
        seed=args.seed,
        min_val_per_speaker=args.min_val_per_speaker,
        allow_empty_val_for_small_speakers=args.allow_empty_val_for_small_speakers,
    )

    write_list(
        res.train,
        out_dir / "train.txt",
        base_dir=base_dir,
        relative=not args.absolute_paths,
    )
    write_list(
        res.val,
        out_dir / "val.txt",
        base_dir=base_dir,
        relative=not args.absolute_paths,
    )

    if args.materialize != "none":
        train_dir = out_dir / "train"
        val_dir = out_dir / "val"
        # materialize expects absolute paths to source files
        materialize(
            res.train, train_dir, method=args.materialize, overwrite=args.overwrite
        )
        materialize(res.val, val_dir, method=args.materialize, overwrite=args.overwrite)

    if args.report:
        total = len(files)
        n_train = len(res.train)
        n_val = len(res.val)
        n_spk = len(res.per_speaker_counts)
        print(f"Base: {base_dir}")
        print(f"Speakers: {n_spk}")
        print(f"Total files: {total}")
        print(f"Train: {n_train}  Val: {n_val}  (val_pct overall ~ {n_val/total:.3f})")
        print("\nPer-speaker (speaker: total -> val):")
        for spk, (n, nv) in sorted(
            res.per_speaker_counts.items(),
            key=lambda kv: int(kv[0]) if kv[0].isdigit() else kv[0],
        ):
            print(f"  {spk}: {n} -> {nv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
