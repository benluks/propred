from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

from clustering.clusters import load_bns_and_clusters


def plot_indices_with_phone_annotations(indices, phones):
    import numpy as np
    import matplotlib.pyplot as plt

    indices = np.asarray(indices)

    # ----- RLE indices -----
    change = np.where(indices[1:] != indices[:-1])[0] + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(indices)]
    values = indices[starts]
    lengths = ends - starts

    fig, ax = plt.subplots(figsize=(16, 3))

    # ----- MIDI-like blocks -----
    for v, s, L in zip(values, starts, lengths):
        ax.broken_barh([(s, L)], (v - 0.4, 0.8))

    # ----- Phones: spans + annotations -----
    for p in phones:

        if p["token"] in {"[PAD]", "<pad>", "|"}:
            continue
        t0 = p["start"]
        t1 = p["end"]

        # phone boundary shading (like torchaudio example)
        ax.axvspan(t0, t1, facecolor="none", edgecolor="black", linewidth=0.2)

        # token label
        ax.annotate(
            p["token"],
            ((t0 + t1) / 2, indices.min() - 4),
            ha="center",
            va="top",
            fontsize=8,
            rotation=0,
            annotation_clip=False,
        )

    # ----- Layout -----
    ax.set_xticks([])
    ax.set_xlabel("")

    ax.set_xlim(0, len(indices))
    ax.set_ylim(indices.min() - 2, indices.max() + 1)
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
        phones = torch.load(
            Path(__file__).parent / "alignment" / TOKEN_TYPE / f"{file_id}.pt"
        )
        plot_indices_with_phone_annotations(label, phones)
        plt.savefig(Path(__file__).parent / "plots" / TOKEN_TYPE / f"{file_id}.png")
