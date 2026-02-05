from pathlib import Path
import torch
import torchaudio
import torchaudio.transforms as T
from datasets import load_dataset
from torchcodec import AudioSamples

from data.data import HFStreamingWrapper


TARGET_SR = 16000
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.mps.is_available() else "cpu")
)


def get_iter_data(split="train.clean.100"):
    name = split.split(".")[1]
    ds = load_dataset("mythicinfinity/libritts", name, streaming=True)
    train_ds = HFStreamingWrapper(ds[split])

    return iter(train_ds)


def load_audio(audio: AudioSamples, target_sr=TARGET_SR):
    samples = audio.get_all_samples()
    audio_data = samples.data
    sr = samples.sample_rate
    return T.Resample(sr, target_sr)(audio_data)


if __name__ == "__main__":
    ds_iter = get_iter_data()
    for i in range(16):
        sample = next(ds_iter)
        audio = load_audio(sample["audio"])
        torchaudio.save(
            Path(__file__).parent / "audio" / f"{sample['id']}.wav",
            audio,
            sample_rate=TARGET_SR,
            format="wav",
        )
