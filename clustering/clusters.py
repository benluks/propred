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


@torch.no_grad()
def spectral_order(centroids: torch.Tensor, k_nn: int = 8, sigma: float | None = None):
    """
    centroids: [K, D] float tensor (cpu/mps/cuda)
    returns: perm [K] long tensor (indices into centroids) giving the order
    """
    K = centroids.size(0)
    D = torch.cdist(centroids, centroids)  # [K,K], euclidean

    # choose sigma from median of non-diagonal distances if not provided
    if sigma is None:
        dvals = D[~torch.eye(K, dtype=torch.bool, device=D.device)]
        sigma = dvals.median().clamp_min(1e-8)

    # affinity (Gaussian kernel)
    W = torch.exp(-(D * D) / (2 * sigma * sigma))

    # sparsify with kNN to avoid everything connecting to everything
    W.fill_diagonal_(0.0)
    knn = torch.topk(W, k=min(k_nn, K - 1), dim=1).indices
    mask = torch.zeros_like(W, dtype=torch.bool)
    mask.scatter_(1, knn, True)
    mask = mask | mask.t()  # make symmetric
    W = W * mask

    # graph Laplacian L = diag(deg) - W
    deg = W.sum(dim=1)
    L = torch.diag(deg) - W

    # eigen-decomposition (symmetric)
    evals, evecs = torch.linalg.eigh(L)

    # Fiedler vector = eigenvector of 2nd smallest eigenvalue
    f = evecs[:, 1]
    perm = torch.argsort(f)

    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(len(perm), device=perm.device)

    # get distances
    ordered = centroids[perm]
    adj_dist = torch.norm(ordered[1:] - ordered[:-1], dim=1)  # [K-1]

    adj_dist_norm = torch.zeros(adj_dist.shape[0] + 1)
    adj_dist_norm[1:] = adj_dist
    adj_dist_norm = torch.cumsum(adj_dist_norm, 0)
    adj_dist_norm = adj_dist_norm / adj_dist_norm.max()

    return perm, inv_perm, adj_dist_norm


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
