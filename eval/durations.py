import os
from pathlib import Path
import sys

import torch
import torch.nn as nn
import lightning as L
import torchaudio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from clustering.generate_bns import load_bn_extractor
from data.ljspeech import DurationsDataset
from data.utils import warp_f0_by_durations
from train import DurationRegressor
from utils.run_lengths import expand_by_duration

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


@torch.inference_mode
class Converter(nn.Module):
    def __init__(
        self,
        dp_ckpt="/Users/ben/dev/propred/lightning_logs/duration/adapted.ckpt",
        target_speaker="6081",
        device="mps",
    ):

        super().__init__()

        self.model = load_bn_extractor().to(device)
        self.model.eval()
        self.duration_predictor = DurationRegressor.load_from_checkpoint(dp_ckpt).to(
            device
        )
        self.duration_predictor.eval()
        self.target_speaker = target_speaker

    @torch.no_grad
    def forward(self, values, values_mask, wav, orig_durations, C=1.0):

        values, values_mask, wav, orig_durations = (
            values.to("mps"),
            values_mask.to("mps"),
            wav.to("mps"),
            orig_durations.to("mps"),
        )
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)

        pred_durations = self.duration_predictor(values, values_mask)
        durations = torch.round(C * pred_durations + (1 - C) * orig_durations)

        bn_ids = expand_by_duration(values, durations)
        bn = self.duration_predictor.model.embedding(bn_ids).transpose(1, 2)

        # methods from: https://github.com/deep-privacy/SA-toolkit/blob/master/egs/vc/libritts/local/tuning/hifigan.py
        f0 = self.model.get_f0(wav)
        warped_f0 = warp_f0_by_durations(
            f0, orig_durations, durations, in_log_domain=False
        ).reshape(1, 1, -1)
        spk_id = self.model.get_spk_id(wav, self.target_speaker)
        return self.model._forward(warped_f0, bn, spk_id).squeeze(0)


if __name__ == "__main__":
    ds = DurationsDataset("/Users/ben/dev/propred/data/LJSpeech-1.1/")
    converter = Converter(target_speaker="1069")

    values, orig_durations, wav, utt_id = ds[0]
    values = values.unsqueeze(0)
    y = converter(values, torch.ones_like(values), wav, orig_durations)
    torchaudio.save(f"{utt_id}_warped.wav", y.to("cpu"), 16_000)
    y = converter(values, torch.ones_like(values), wav, orig_durations, C=0)
    torchaudio.save(f"{utt_id}.wav", y.to("cpu"), 16_000)
