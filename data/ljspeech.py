from pathlib import Path
from typing import Union
import torch
from torch.types import FileLike
from torch.utils.data import Dataset

from data.utils import load_audio
from propred.utils.run_lengths import rle_encode_1d, singleton_kill


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
        spk_id=False,
        spk_id_path: Union[FileLike, dict, None] = None,
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

        self.spk_id = spk_id
        if self.spk_id:
            if spk_id_path is None:
                raise ValueError("spk_id_path must be provided if spk_id is True")
            self.spk_id_path = Path(spk_id_path)
            if not self.spk_id_path.exists():
                raise FileNotFoundError(f"spk_id_path not found: {self.spk_id_path}")
            # dict of speaker ids, keys are ids, values are contiguous integers where max is num speakers - 1
            self.spk_id_map = self._load_spk_id_map(self.spk_id_path)

    def __len__(self):
        return len(self.files)

    def _parse_spk_id(self, utt_id: str) -> int:
        return utt_id.split("-")[0]

    def _load_spk_id_map(self, input) -> dict:
        if isinstance(input, dict):
            return input
        elif isinstance(input, (str, Path)):
            path = Path(input)
            if not path.exists():
                raise FileNotFoundError(f"spk_id_path not found: {path}")
            return torch.load(path)
        else:
            raise ValueError(
                "spk_id_path must be a dict or a path to a file containing a dict"
            )

    def __getitem__(self, i: int):
        path = self.files[i]
        utt_id = path.stem
        x = torch.load(path).to(torch.long).flatten()

        values, durations, *_ = rle_encode_1d(x)
        if self.kill_singletons > 0:
            values, durations = singleton_kill(
                values, durations, k=self.kill_singletons
            )

        ret = (values, durations)
        if self.return_wav:
            wav = load_audio(self.wavs[i], self.target_sr)
            ret += (wav, utt_id)

        if self.spk_id:
            ret += (self.spk_id_map[self._parse_spk_id(utt_id)],)

        return ret


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
