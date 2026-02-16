from pathlib import Path
import torch
from torch.utils.data import Dataset

from data.utils import load_audio
from utils.run_lengths import rle_encode_1d, singleton_kill


class DurationsDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        pattern: str = "*.pt",
        embs_folder="emb_ids",
        split=None,
        return_wav=True,
        wavs_folder="wavs",
        wavs_pattern="*.wav",
        target_sr=16000,
        kill_singletons: int = 0,
    ):

        self.root = Path(root)
        self.split = split
        self.embs_path = self.root / embs_folder
        if self.split:
            self.embs_path = self.embs_path / split
        self.files = sorted(self.embs_path.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matched {pattern} in {self.embs_path}")

        self.return_wav = return_wav
        if self.return_wav:
            self.wavs = sorted((self.root / wavs_folder).glob(wavs_pattern))
            assert len(self.wavs) == len(self.files)
        self.target_sr = target_sr

        self.kill_singletons = kill_singletons

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        path = self.files[i]
        utt_id = path.stem
        x = torch.load(path).to(torch.long).flatten()

        values, durations, *_ = rle_encode_1d(x)
        if self.kill_singletons > 0:
            values, durations = singleton_kill(
                values, durations, k=self.kill_singletons
            )

        if self.return_wav:
            wav = load_audio(self.wavs[i], self.target_sr)
            return values, durations, wav, utt_id

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
