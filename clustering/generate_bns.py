from pathlib import Path
from datasets import load_dataset
import datasets
import torch
import torchaudio.transforms as T
from tqdm import tqdm
from torchcodec import AudioSamples

from data.prosody_flow import HFStreamingWrapper

TARGET_SR = 16_000
DEVICE = "mps"


def load_audio(audio: AudioSamples, target_sr=TARGET_SR):
    samples = audio.get_all_samples()
    audio_data = samples.data
    sr = samples.sample_rate
    return T.Resample(sr, target_sr)(audio_data).to(DEVICE)


def load_bn_extractor():

    bn_extractor = torch.hub.load(
        "deep-privacy/SA-toolkit",
        "anonymization",
        tag_version="hifigan_bn_tdnnf_wav2vec2_vq_48_v1",
        trust_repo=True,
    )
    bn_extractor.eval()
    for parameter in bn_extractor.parameters():
        parameter.requires_grad = False

    return bn_extractor.to(DEVICE)


def get_bn(bn_extractor, audio):
    audio = load_audio(audio).float()
    with torch.no_grad():
        return bn_extractor.get_bn(audio)


if __name__ == "__main__":

    ds = load_dataset("mythicinfinity/libritts", "clean", streaming=True)
    train_ds = HFStreamingWrapper(ds["train.clean.100"])

    ds_iter = iter(train_ds)

    bn_extractor = load_bn_extractor()

    for i in tqdm(range(16)):
        sample = next(ds_iter)
        bn = get_bn(bn_extractor, sample["audio"])

        torch.save(bn.squeeze(0), Path("clustering") / "bns" / sample["id"])
