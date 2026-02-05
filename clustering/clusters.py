from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
import torch


BNS_PATH = Path(__file__).parent / "bns"


import torch


@torch.no_grad()
def cluster(x):
    if torch.is_tensor(x):
        x = x.cpu().numpy()
    kmeans = KMeans(n_clusters=48, random_state=0, n_init="auto").fit(x)

    return torch.from_numpy(kmeans.labels_), torch.from_numpy(kmeans.cluster_centers_)


def run_length(arr):
    arr = np.asarray(arr)
    change_idx = np.flatnonzero(np.diff(arr)) + 1
    idx = np.r_[0, change_idx, len(arr)]
    lengths = np.diff(idx)
    values = arr[idx[:-1]]
    return values, lengths


def run_lengths(inv: torch.Tensor):
    """
    inv: [T] integer tensor
    Returns:
        values: unique value per run
        lengths: length of each run
    """
    # Find where value changes
    change = torch.ones_like(inv, dtype=torch.bool)
    change[1:] = inv[1:] != inv[:-1]

    # Indices where new run starts
    run_starts = torch.nonzero(change, as_tuple=False).flatten()

    # Run lengths via diff
    run_ends = torch.cat(
        [run_starts[1:], torch.tensor([inv.numel()], device=inv.device)]
    )
    lengths = run_ends - run_starts

    values = inv[run_starts]

    return values, lengths


def load_bns_and_clusters():

    chunks = []
    Ts = []
    file_ids = []

    for f in BNS_PATH.glob("*.pt"):
        bn = torch.load(f).transpose(0, 1)
        T, _ = bn.shape
        Ts.append(T)
        chunks.append(bn)
        file_ids.append(f.stem)

    x = torch.cat(chunks).to("cpu")
    labels, centroids = cluster(x)

    return x, Ts, file_ids, labels, centroids


if __name__ == "__main__":

    chunks = []
    Ts = []

    for f in BNS_PATH.glob("*.pt"):
        bn = torch.load(f).transpose(0, 1)
        T, _ = bn.shape
        Ts.append(T)
        chunks.append(bn)

    x = torch.cat(chunks)
    labels, centroids = cluster(x)
