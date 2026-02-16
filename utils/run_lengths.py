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


def singleton_kill(values, lengths, k=1):
    """
    values:  [R]
    lengths: [R]
    k: max length to consider a glitch (1 = singleton)

    returns:
        new_values, new_lengths
    """
    values = values.clone()
    lengths = lengths.clone()

    R = len(values)
    for i in range(1, R - 1):
        if lengths[i] <= k:
            if values[i - 1] == values[i + 1]:
                # kill the singleton
                values[i] = values[i - 1]

    return values, lengths


def expand_by_duration(
    values_1d: torch.Tensor, durations_1d: torch.Tensor
) -> torch.Tensor:
    values = values_1d.to(torch.long).flatten()
    durations = durations_1d.to(torch.long).flatten()
    if values.numel() != durations.numel():
        raise ValueError("values and durations must have same length")
    if (durations < 0).any():
        raise ValueError("durations must be non-negative")
    return torch.repeat_interleave(values, durations)


@torch.no_grad()
def quantize_to_indices(x: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """
    x:       [B, D, T] float
    centers: [K, D] float  (e.g., torch.from_numpy(kmeans.cluster_centers_))

    returns:
      idx: [B, T] long  (nearest center per frame)
    """
    if x.ndim != 3:
        raise ValueError(f"Expected [B,D,T], got {tuple(x.shape)}")
    if centers.ndim != 2:
        raise ValueError(f"Expected centers [K,D], got {tuple(centers.shape)}")

    B, D, T = x.shape
    K, Dc = centers.shape
    if Dc != D:
        raise ValueError(f"Center dim {Dc} != x dim {D}")

    # [B,T,D]
    xt = x.permute(0, 2, 1).contiguous()

    # Compute squared Euclidean distance efficiently:
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    a2 = (xt**2).sum(dim=-1, keepdim=True)  # [B,T,1]
    b2 = (centers**2).sum(dim=-1).view(1, 1, K)  # [1,1,K]
    ab = xt @ centers.t()  # [B,T,K]
    dist2 = a2 + b2 - 2 * ab  # [B,T,K]

    idx = dist2.argmin(dim=-1).to(torch.long)  # [B,T]
    return idx


def rle_encode_1d(x: torch.Tensor):
    if x.ndim != 1:
        raise ValueError(f"Expected 1D tensor, got shape {tuple(x.shape)}")
    if x.numel() == 0:
        empty = torch.empty(0, dtype=x.dtype, device=x.device)
        emptyL = torch.empty(0, dtype=torch.long, device=x.device)
        return empty, emptyL, emptyL, emptyL

    change = torch.nonzero(x[1:] != x[:-1], as_tuple=False).flatten() + 1
    starts = torch.cat([torch.tensor([0], device=x.device), change])
    ends = torch.cat([change, torch.tensor([x.numel()], device=x.device)])

    values = x[starts]
    durations = (ends - starts).to(torch.long)
    return values, durations, starts, ends


def rle_encode_batch(idx_bt: torch.Tensor, pad_value: int = -1):
    """
    idx_bt: [B, T] long

    returns padded:
      values:   [B, Rmax] long
      lengths:  [B, Rmax] long
      rmask:    [B, Rmax] bool  (True where real run exists)
    """
    if idx_bt.ndim != 2:
        raise ValueError(f"Expected [B,T], got {tuple(idx_bt.shape)}")

    B, T = idx_bt.shape
    values_list, lengths_list = [], []
    Rmax = 0
    for b in range(B):
        v, l, _, _ = rle_encode_1d(idx_bt[b])
        values_list.append(v)
        lengths_list.append(l)
        Rmax = max(Rmax, v.numel())

    device = idx_bt.device
    values = torch.full((B, Rmax), pad_value, dtype=torch.long, device=device)
    lengths = torch.zeros((B, Rmax), dtype=torch.long, device=device)
    rmask = torch.zeros((B, Rmax), dtype=torch.bool, device=device)

    for b in range(B):
        R = values_list[b].numel()
        if R == 0:
            continue
        values[b, :R] = values_list[b].to(torch.long)
        lengths[b, :R] = lengths_list[b].to(torch.long)
        rmask[b, :R] = True

    return values, lengths, rmask


def expand_batch(values: torch.Tensor, pred_lengths: torch.Tensor, rmask: torch.Tensor):
    """
    values:       [B, Rmax]
    pred_lengths: [B, Rmax] (long)
    rmask:        [B, Rmax] bool

    returns:
      idx_list: list of length B, each is [T'_b] long (ragged)
    """
    B, Rmax = values.shape
    out = []
    for b in range(B):
        v = values[b][rmask[b]]
        l = pred_lengths[b][rmask[b]]
        out.append(expand_by_duration(v, l))
    return out


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
