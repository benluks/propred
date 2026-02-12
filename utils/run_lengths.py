import matplotlib.pyplot as plt
import torch


import torch


def rle_encode_1d(x: torch.Tensor):
    """
    x: 1D int tensor of length T (frame-level indices)
    returns:
      values:    (R,) int tensor
      durations: (R,) long tensor (run lengths in frames)
      starts:    (R,) long tensor (start frame index)
      ends:      (R,) long tensor (end frame index, exclusive)
    """
    if x.ndim != 1:
        raise ValueError(f"Expected 1D tensor, got shape {tuple(x.shape)}")
    if x.numel() == 0:
        # edge case: empty sequence
        empty = torch.empty(0, dtype=x.dtype, device=x.device)
        emptyL = torch.empty(0, dtype=torch.long, device=x.device)
        return empty, emptyL, emptyL, emptyL

    # where value changes (boundaries between runs)
    change = torch.nonzero(x[1:] != x[:-1], as_tuple=False).flatten() + 1

    starts = torch.cat([torch.tensor([0], device=x.device), change])
    ends = torch.cat([change, torch.tensor([x.numel()], device=x.device)])

    values = x[starts]
    durations = (ends - starts).to(torch.long)

    return values, durations, starts, ends


def get_run_lengths(x):
    change = torch.nonzero(x[1:] != x[:-1], as_tuple=False).flatten() + 1

    starts = torch.cat([torch.tensor([0]), change])
    ends = torch.cat([change, torch.tensor([len(x)])])

    run_lengths = ends - starts

    return run_lengths


def expand_by_duration(values: torch.Tensor, durations: torch.Tensor) -> torch.Tensor:
    """
    values:    (R,) int tensor (run values)
    durations: (R,) int tensor (run lengths in frames)
    returns:   (sum(durations),) int tensor
    """
    ndim = values.ndim
    values = values.to(torch.long).flatten()
    durations = durations.to(torch.long).flatten()

    if values.numel() != durations.numel():
        raise ValueError("values and durations must have same length")

    if (durations < 0).any():
        raise ValueError("durations must be non-negative")

    expanded = torch.repeat_interleave(values, durations)
    if ndim == 2:
        return expanded.unsqueeze(0)
    return expanded


def plot_run_lengths(run_lengths):
    hist = torch.bincount(run_lengths)

    lengths = torch.arange(len(hist))
    counts = hist

    mask = counts > 0
    lengths = lengths[mask]
    counts = counts[mask]

    plt.figure(figsize=(6, 4))
    plt.bar(lengths.numpy(), counts.numpy(), width=0.8)
    plt.xlabel("Run length")
    plt.ylabel("Count")
    plt.title("Histogram of run lengths")
    plt.tight_layout()
    plt.savefig("/Users/ben/dev/propred/clustering/run_lengths/lj-speech.png")


if __name__ == "__main__":
    labels = torch.load("/Users/ben/dev/propred/data/LJSpeech-1.1/labels.pt")
    run_lengths = get_run_lengths(labels)
    plot_run_lengths(run_lengths)
