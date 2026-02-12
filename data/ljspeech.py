from pathlib import Path
import torch
from torch.utils.data import Dataset

from utils.run_lengths import rle_encode_1d


class DurationsDataset(Dataset):
    def __init__(
        self, root: str | Path, pattern: str = "*.pt", return_meta: bool = False
    ):
        self.root = Path(root)
        self.files = sorted(self.root.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matched {pattern} in {self.root}")
        self.return_meta = return_meta

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        path = self.files[i]
        x = torch.load(path)  # expected: 1D integer tensor

        # If it’s saved as something else (e.g., dict), adapt here
        # e.g., x = x["indices"]

        x = x.to(torch.long).flatten()  # enforce 1D

        values, durations, starts, ends = rle_encode_1d(x)

        if self.return_meta:
            meta = {
                "path": str(path),
                "T": int(x.numel()),
                "R": int(values.numel()),
                "starts": starts,
                "ends": ends,
            }
            return values, durations, meta

        return values, durations


class PitchDataset(Dataset):
    def __init__(
        self,
        bn_root: str | Path,
        feats_root: str | Path,
        pattern: str = "*.pt",
    ):
        self.bn_root = Path(bn_root)
        self.bn_files = sorted(self.bn_root.glob(pattern))
        if not self.bn_files:
            raise FileNotFoundError(f"No files matched {pattern} in {self.root}")

        self.feats_root = Path(feats_root)
        if not self.feats_root.exists():
            raise FileNotFoundError(f"feats_root not found: {self.feats_root}")

    def __len__(self):
        return len(self.bn_files)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        bn_path = self.bn_files[i]
        # expected: 1D integer tensor
        indices = torch.load(bn_path).to(torch.long).flatten()
        # loads dict containing keys 'f0', 'vuv', and 'energy'
        feats = torch.load(self.feats_root / bn_path.name)

        indices_numel = indices.numel()

        min_length = min(indices_numel, *[v.numel() for _, v in feats.items()])

        for k, v in feats.items():
            feats[k] = v[:min_length]

        return indices[:min_length], feats
