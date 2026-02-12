import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path

from librosa import pyin
from librosa.feature import rms

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

FINAL_RATE = 50
F_MIN = 60
F_MAX = 8000

TARGET_SR = 16000


def get_f0_energy(
    y: np.ndarray,
    sample_rate: int,
    do_log=True,
    speaker_normalized: bool = False,
    speaker_dict: dict = None,
    speaker_id: str = None,
):

    if torch.is_tensor(y):
        y = y.numpy()

    hop_length = sample_rate // FINAL_RATE
    f0, vuv, _ = pyin(
        y=y,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=F_MIN,
        fmax=F_MAX,
    )

    f0 = torch.from_numpy(f0).float()
    vuv = torch.from_numpy(vuv).bool()
    f0[~vuv] = 0
    f0[~f0.isfinite()] = 0

    energy = torch.from_numpy(rms(y=y, hop_length=hop_length)[0]).float()

    if do_log:
        f0[vuv] = torch.log(f0[vuv])

    if speaker_normalized:
        assert (
            speaker_dict is not None
        ), "`speaker_normalized` is set to `True`, but `speaker_dict` is `None`"
        # log speaker mean normalized f0
        speaker_mu = speaker_dict["mu_logf0"][int(speaker_id)]
        speaker_sigma = max(speaker_dict["sigma_logf0"][int(speaker_id)], 1e-6)

        f0[vuv] = (f0[vuv] - speaker_mu) / speaker_sigma

    return f0, vuv.to(torch.float32), energy


def load_audio(path, target_sr):
    x, sr = torchaudio.load(path)
    x = x.mean(dim=0)
    return T.Resample(sr, target_sr)(x)


def process_one(wav_path_str: str, out_dir_str: str):
    # Avoid each worker spinning up many CPU threads
    torch.set_num_threads(1)

    wav_path = Path(wav_path_str)
    out_dir = Path(out_dir_str)
    out_file = out_dir / f"{wav_path.stem}.pt"

    # Skip if already computed (useful if you re-run)
    if out_file.exists():
        return str(out_file)

    wav = load_audio(wav_path, TARGET_SR)
    f0, vuv, energy = get_f0_energy(wav, TARGET_SR)
    torch.save(dict(f0=f0, vuv=vuv, energy=energy), out_file)
    return str(out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavs_path", default="./data/LJSpeech-1.1/wavs")
    parser.add_argument("--out_path", default="./data/LJSpeech-1.1")
    parser.add_argument(
        "--num_workers", type=int, default=max(1, (os.cpu_count() or 1) - 1)
    )
    args = parser.parse_args()

    out_dir = Path(args.out_path) / "feats"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(Path(args.wavs_path).glob("*.wav"))
    if not wav_files:
        raise RuntimeError(f"No wav files found in {args.wavs_path}")

    # Submit jobs
    with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(process_one, str(w), str(out_dir)) for w in wav_files]

        # Progress bar as tasks finish
        for fut in tqdm(
            as_completed(futures), total=len(futures), desc="Extracting feats"
        ):
            # This will re-raise exceptions with traceback if a worker fails
            fut.result()


import torch
import torch.nn.functional as F


@torch.no_grad()
def warp_f0_by_durations(
    f0_old: torch.Tensor,
    d_old: torch.Tensor,
    d_new: torch.Tensor,
    *,
    interp: str = "linear",  # "linear" is what you want
    in_log_domain: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Piecewise warp f0 according to per-run duration changes.

    Args:
      f0_old: (T,) float tensor (Hz or log-Hz)
      d_old:  (R,) long tensor, original run durations; sum(d_old) == T
      d_new:  (R,) long tensor, target run durations (predicted)
      interp: interpolation mode for 1D resampling ("linear" recommended)
      in_log_domain: if True, treat f0_old as Hz but interpolate log(f0)
      eps: floor for log

    Returns:
      f0_new: (sum(d_new),) float tensor
    """
    f0_old = f0_old.float().flatten()
    d_old = d_old.to(torch.long).flatten()
    d_new = d_new.to(torch.long).flatten()

    if d_old.numel() != d_new.numel():
        raise ValueError(
            f"d_old and d_new must have same length. Got {d_old.numel()} vs {d_new.numel()}"
        )

    T = int(d_old.sum().item())
    if f0_old.numel() != T:
        raise ValueError(
            f"sum(d_old) must match len(f0_old). sum(d_old)={T}, len(f0_old)={f0_old.numel()}"
        )

    # Optional: interpolate in log domain (common for pitch)
    if in_log_domain:
        x = torch.log(f0_old.clamp_min(eps))
    else:
        x = f0_old

    out_chunks = []
    start = 0
    for L0, L1 in zip(d_old.tolist(), d_new.tolist()):
        seg = x[start : start + L0]  # (L0,)
        start += L0

        if L1 <= 0:
            # drop this run entirely (rare, but can happen if predictor outputs 0)
            continue

        if L0 == 0:
            # shouldn't happen if d_old are run-lengths, but guard anyway
            out_chunks.append(torch.zeros(L1, dtype=x.dtype, device=x.device))
            continue

        if L0 == L1:
            out_chunks.append(seg)
            continue

        # Resample seg from length L0 -> L1 using 1D interpolate.
        # F.interpolate expects (N,C,L)
        seg3 = seg.view(1, 1, L0)
        seg_rs = F.interpolate(seg3, size=L1, mode=interp, align_corners=True).view(L1)
        out_chunks.append(seg_rs)

    y = (
        torch.cat(out_chunks, dim=0)
        if out_chunks
        else torch.empty(0, dtype=x.dtype, device=x.device)
    )

    # Convert back from log if needed
    if in_log_domain:
        y = torch.exp(y)

    return y


from collections import OrderedDict
from pathlib import Path
import torch


def rewrite_duration_ckpt_for_modular_model(in_path: str | Path, out_path: str | Path):
    ckpt = torch.load(in_path, map_location="cpu")

    old_sd = ckpt["state_dict"]
    new_sd = OrderedDict()

    for k, v in old_sd.items():
        # old: model.conv_1.* / model.norm_1.* / model.conv_2.* / model.norm_2.*
        # new: model.conv.conv_1.* / model.conv.norm_1.* / ...
        if k.startswith("model.") and k[len("model.") :].startswith(
            ("conv_1.", "conv_2.", "norm_1.", "norm_2.")
        ):
            k = "model.conv." + k[len("model.") :]
        # embedding + proj stay the same: model.embedding.* , model.proj.*
        new_sd[k] = v

    ckpt["state_dict"] = new_sd
    torch.save(ckpt, out_path)



# model = DurationRegressor.load_from_checkpoint("rewritten.ckpt")  # now works
