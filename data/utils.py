import argparse
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


if __name__ == "__main__":
    # compute f0 and vuv over dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--wavs_path", default="./data/LJSpeech-1.1/wavs")
    parser.add_argument("--out_path", default="./data/LJSpeech-1.1")

    args = parser.parse_args()
    out_dir = Path(args.out_path) / "feats"
    out_dir.mkdir(parents=True, exist_ok=True)

    for wav_file in tqdm(Path(args.wavs_path).glob("*.wav")):
        wav = load_audio(wav_file, 16_000)

        f0, vuv, energy = get_f0_energy(wav, 16_000)
        feat_dict = dict(f0=f0, vuv=vuv, energy=energy)
        torch.save(feat_dict, out_dir / f"{wav_file.stem}.pt")
