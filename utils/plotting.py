from pathlib import Path
from typing import Iterable
import torch
import matplotlib.pyplot as plt


def plot_vuv_and_logf0(
    pt_files: Iterable[str | Path],
    *,
    show_ref: bool = False,
    use_probs: bool = False,
    figsize=(12, 3),
    alpha_pred=1.0,
    alpha_ref=0.6,
):
    """
    Plot predicted VUV + log-F0 for one or more .pt files.

    Args:
        pt_files: iterable of paths to .pt files
        show_ref: overlay reference f0/vuv if present
        use_probs: plot vuv probabilities instead of binary vuv
        figsize: base figure size per utterance
    """
    pt_files = list(pt_files)

    for path in pt_files:
        path = Path(path)
        data = torch.load(path, map_location="cpu")

        log_f0 = data["pred_log_f0"].cpu()
        vuv = (
            data["pred_vuv_prob"].cpu()
            if use_probs and "pred_vuv_prob" in data
            else data["pred_vuv"].cpu()
        )

        T = log_f0.numel()
        t = torch.arange(T)

        fig, (ax_f0, ax_vuv) = plt.subplots(
            2, 1, figsize=(figsize[0], figsize[1] * 2), sharex=True
        )

        # ---- log-F0 ----
        log_f0[log_f0 == 0] = torch.nan
        log_f0[~vuv.bool()] = torch.nan
        ax_f0.plot(t, log_f0, label="pred log-f0", alpha=alpha_pred)
        ax_f0.set_ylabel("log F0")
        ax_f0.set_title(path.stem)

        if show_ref and "ref_feats" in data and "f0" in data["ref_feats"]:
            ref_f0 = torch.log(torch.as_tensor(data["ref_feats"]["f0"]).clamp_min(1e-6))
            ref_f0[ref_f0 <= 0] = torch.nan
            ax_f0.plot(t, ref_f0[:T], "--", label="ref log-f0", alpha=alpha_ref)

        ax_f0.legend(loc="upper right")

        # ---- VUV ----
        ax_vuv.plot(t, vuv, label="pred vuv", alpha=alpha_pred)
        ax_vuv.set_ylim(-0.05, 1.05)
        ax_vuv.set_ylabel("VUV")
        ax_vuv.set_xlabel("Frame")

        if show_ref and "ref_feats" in data and "vuv" in data["ref_feats"]:
            ref_vuv = torch.as_tensor(data["ref_feats"]["vuv"])
            ax_vuv.plot(t, ref_vuv[:T], "--", label="ref vuv", alpha=alpha_ref)

        ax_vuv.legend(loc="upper right")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    plot_vuv_and_logf0(
        [
            "/Users/ben/dev/propred/data/LJSpeech-1.1/predictions/version_2/LJ001-0023.pt"
        ],
        show_ref=True,
    )
