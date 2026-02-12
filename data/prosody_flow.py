import json
from operator import itemgetter
from typing import List
from datasets import load_dataset
from librosa import pyin
from librosa.feature import rms
import matplotlib.pyplot as plt
import numpy as np

import torchaudio.transforms as T
from datasets import IterableDataset
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

FINAL_RATE = 50
TARGET_SR = 16000
F_MIN = 60
F_MAX = 400


def transforms(
    y: np.ndarray,
    sample_rate: int,
    speaker_id: str,
    do_log=True,
    speaker_normalized: bool = False,
    speaker_dict: dict = None,
):
    hop_length = sample_rate // FINAL_RATE
    f0, vuv, _ = pyin(
        y=y,
        sr=sample_rate,
        hop_length=hop_length,
        fmin=F_MIN,
        fmax=F_MAX,
    )

    f0 = torch.from_numpy(f0).float()
    vuv = torch.from_numpy(vuv).bool()
    f0[~vuv] = 0

    energy = torch.from_numpy(rms(y=y, hop_length=hop_length)[0]).float()

    if do_log:
        f0[vuv] = torch.log(f0[vuv])

    if speaker_normalized:
        assert (
            speaker_dict is not None
        ), "`speaker_normalized` is set to `True`, but `speaker_dict` is `None`"
        # log speaker mean normalized f0
        speaker_mu = speaker_dict["mu_logf0"][int(speaker_id)]
        speaker_sigma = max(speaker_dict["sigma_logf0"][int(speaker_id)], 1e-6)

        f0[vuv] = (f0[vuv] - speaker_mu) / speaker_sigma

    return f0, vuv.to(torch.float32), energy


def preprocess(ex, speaker_dict, **kwargs):
    y = ex["audio"]["array"]
    sample_rate = ex["audio"].get_all_samples().sample_rate

    
    wav = torch.from_numpy(y).float()
    if wav.ndim == 2:  # [C, T]
        wav = wav.mean(dim=0)  # or pick channel 0
    wav = T.Resample(sample_rate, TARGET_SR)(wav)
    ex["wav"] = wav

    speaker_id = ex["speaker_id"]
    ex["speaker_id"] = speaker_dict[str(ex["speaker_id"])]

    f0, vuv, energy = transforms(ex["wav"].numpy(), TARGET_SR, speaker_id, **kwargs)

    ex["f0"] = f0
    ex["vuv"] = vuv
    ex["energy"] = energy

    return ex


def pad_tensor_lists(tensor_lists: List[List[torch.tensor]]):
    return [
        pad_sequence(tensor_list, batch_first=True, padding_value=0.0)
        for tensor_list in tensor_lists
    ]


def collate(batch):

    wavs = [ex["wav"] for ex in batch]
    f0s = [ex["f0"] for ex in batch]
    vuvs = [ex["vuv"] for ex in batch]
    ens = [ex["energy"] for ex in batch]
    speaker_id = torch.tensor([int(ex["speaker_id"]) for ex in batch], dtype=torch.long)

    wav_len = torch.tensor([w.numel() for w in wavs], dtype=torch.long)
    frame_len = torch.tensor([x.numel() for x in f0s], dtype=torch.long)

    wav, f0, vuv, energy = pad_tensor_lists([wavs, f0s, vuvs, ens])
    wav_pad_mask = (
        torch.arange(wav.shape[1])[None, :] >= wav_len[:, None]
    )  # (B, T_wav) True=pad
    frame_pad_mask = (
        torch.arange(f0.shape[1])[None, :] >= frame_len[:, None]
    )  # (B, Tf_max)

    return {
        "wav": wav,
        "wav_len": wav_len,
        "wav_pad_mask": wav_pad_mask,
        "f0": f0,
        "vuv": vuv,
        "energy": energy,
        "frame_len": frame_len,
        "frame_pad_mask": frame_pad_mask,
        "speaker_id": speaker_id,
    }


def plot_f0_and_energy(f0, energy):
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(f0)
    plt.ylabel("F0")
    plt.title("Fundamental Frequency")

    plt.subplot(2, 1, 2)
    plt.plot(energy)
    plt.ylabel("Energy")
    plt.xlabel("Frame")
    plt.title("RMS Energy")

    plt.tight_layout()
    plt.show()


class HFStreamingWrapper(IterableDataset):
    def __init__(self, hf_iterable):
        self.hf_iterable = hf_iterable

    def __iter__(self):
        it = iter(self.hf_iterable)
        info = get_worker_info()
        if info is None:
            yield from it
        else:
            for i, ex in enumerate(it):
                if i % info.num_workers == info.id:
                    yield ex


if __name__ == "__main__":

    ds = load_dataset("mythicinfinity/libritts", "clean", streaming=True)
    speaker_dict = json.load(open("f0_stats/speaker_dict.json"))
    train_ds = ds["train.clean.100"].map(
        preprocess, fn_kwargs={"speaker_dict": speaker_dict}
    )

    train_loader = DataLoader(
        HFStreamingWrapper(train_ds), batch_size=16, num_workers=4, collate_fn=collate
    )

    batch = next(iter(train_loader))

    # plot the log mean-normalized f[0]
    f0 = batch["f0"][0].clone()
    f0[f0 == 0] = torch.nan
    plot_f0_and_energy(f0, batch["energy"][0])
