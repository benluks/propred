from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from clustering.clusters import load_bns_and_clusters, spectral_order

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


def plot_indices_with_phone_annotations(
    indices,
    phones,
    dist_norm,
    S_bg=None,
    cmap_name="plasma",
    note_h=0.8,  # <-- the only knob you care about
    inches_per_lane=0.18,  # auto-scale figure height (leave this alone)
):
    indices = np.asarray(indices)

    # ----- RLE indices -----
    change = np.where(indices[1:] != indices[:-1])[0] + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(indices)]
    values = indices[starts]
    lengths = ends - starts

    # ----- y-range -----
    y_min_val = int(indices.min())
    y_max_val = int(indices.max())
    n_lanes = (y_max_val - y_min_val) + 1

    # pad so phone text fits below
    ymin = y_min_val - 2
    ymax = y_max_val + 1

    # ----- auto figure height based on lanes + note thickness -----
    fig_h = max(2.0, n_lanes * inches_per_lane * (note_h / 0.8))
    fig, ax = plt.subplots(figsize=(16, fig_h))

    # ----- Optional spectrogram background -----
    if S_bg is not None:
        S_bg = np.asarray(S_bg)
        T = min(S_bg.shape[1], len(indices))
        S_bg = S_bg[:, :T]
        ax.imshow(
            S_bg,
            origin="lower",
            aspect="auto",
            extent=[0, T, ymin, ymax],
            interpolation="nearest",
            alpha=0.5,
            zorder=0,
        )
        ax.set_zorder(1)
        ax.patch.set_alpha(0.0)

    cmap = plt.get_cmap(cmap_name)

    # ----- MIDI-like blocks -----
    half = note_h / 2
    for v, s, L in zip(values, starts, lengths):
        color = cmap(dist_norm[int(v)])
        ax.broken_barh(
            [(s, L)], (v - half, note_h), zorder=2, facecolors=color, edgecolors="none"
        )

    # ----- Phones: spans + annotations -----
    for p in phones:
        if p["token"] in {"[PAD]", "<pad>", "|"}:
            continue
        t0, t1 = p["start"], p["end"]

        ax.axvspan(t0, t1, facecolor="none", edgecolor="black", linewidth=0.2, zorder=3)

        ax.annotate(
            p["token"],
            ((t0 + t1) / 2, y_min_val - 3),  # anchored below lanes
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
    perm, inv_perm, adj_dist_norm = spectral_order(centroids)

    Ts = [0] + Ts
    Ts = torch.cumsum(torch.tensor(Ts), dim=0)

    labels = inv_perm[labels]

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

        plot_indices_with_phone_annotations(label, phones, adj_dist_norm, S_bg=S_bg)
        plt.savefig(Path(__file__).parent / "plots" / "dist" / f"{file_id}.png")
