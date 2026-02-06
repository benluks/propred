from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from clustering.clusters import load_bns_and_clusters

import torch
import torchaudio
import torchaudio.transforms as T


def mel_spectrogram_50hz(
    wav: torch.Tensor,  # [T] or [1,T]
    sr: int = 16000,
    n_mels: int = 80,
    n_fft: int = 1024,
    f_min: float = 40.0,
    f_max: float | None = None,
):
    """
    Returns:
      S_db: [n_mels, n_frames] float32 (log-mel in dB-ish)
      hop: hop length in samples (≈ sr/50)
    """
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    hop = int(round(sr / 50))  # <-- key: 50 Hz frames

    mel = T.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,
        center=True,
    )(wav)

    mel = mel.squeeze(0).clamp_min(1e-10)
    S_db = T.AmplitudeToDB("power", 80)(mel)  # log-power
    return S_db, hop


def plot_indices_with_phone_annotations(indices, phones, S_bg=None):
    indices = np.asarray(indices)

    # ----- RLE indices -----
    change = np.where(indices[1:] != indices[:-1])[0] + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(indices)]
    values = indices[starts]
    lengths = ends - starts

    fig, ax = plt.subplots(figsize=(16, 3))

    # ----- Optional spectrogram background -----
    ymin = indices.min() - 2
    ymax = indices.max() + 1

    if S_bg is not None:
        # S_bg expected shape [F, T] where T should match len(indices) (or close)
        S_bg = np.asarray(S_bg)

        # If time dims mismatch, crop to the overlapping region
        T = min(S_bg.shape[1], len(indices))
        S_bg = S_bg[:, :T]

        ax.imshow(
            S_bg,
            origin="lower",
            aspect="auto",
            extent=[0, T, ymin, ymax],  # stretch freq axis into your index y-range
            interpolation="nearest",
            alpha=0.5,
            zorder=0,
        )

    # Put everything else above the background
    ax.set_zorder(1)
    ax.patch.set_alpha(0.0)

    # ----- MIDI-like blocks -----
    for v, s, L in zip(values, starts, lengths):
        ax.broken_barh([(s, L)], (v - 0.4, 0.8), zorder=2)

    # ----- Phones: spans + annotations -----
    for p in phones:
        if p["token"] in {"[PAD]", "<pad>", "|"}:
            continue
        t0 = p["start"]
        t1 = p["end"]

        ax.axvspan(t0, t1, facecolor="none", edgecolor="black", linewidth=0.2, zorder=3)

        ax.annotate(
            p["token"],
            ((t0 + t1) / 2, indices.min() - 4),
            ha="center",
            va="top",
            fontsize=8,
            rotation=0,
            annotation_clip=False,
            zorder=4,
        )

    # ----- Layout -----
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_xlim(0, len(indices))
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Index")

    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":

    TOKEN_TYPE = "phones"
    # Example call:
    bns, Ts, file_ids, labels, centroids = load_bns_and_clusters()  # [T] at 50Hz

    Ts = [0] + Ts
    Ts = torch.cumsum(torch.tensor(Ts), dim=0)

    plt.show()

    for file_id, label in zip(
        file_ids,
        [labels[Ts[i - 1] : Ts[i]] for i in range(1, len(Ts))],
    ):
        wav, sr = torchaudio.load(
            Path(__file__).parent / "audio" / f"{file_id}.wav"
        )  # wav: [C, T]
        phones = torch.load(
            Path(__file__).parent / "alignment" / TOKEN_TYPE / f"{file_id}.pt"
        )

        S_db, hop = mel_spectrogram_50hz(wav, sr)
        S_bg = S_db.cpu().numpy()

        plot_indices_with_phone_annotations(label, phones, S_bg=S_bg)
        plt.savefig(Path(__file__).parent / "plots" / "spec" / f"{file_id}.png")
