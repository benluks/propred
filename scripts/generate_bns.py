import argparse
from pathlib import Path
import sys
import torch
import torchaudio
from torchaudio.datasets import LIBRITTS, LJSPEECH, LIBRISPEECH
import torchaudio.transforms as T
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from clustering.generate_bns import load_bn_extractor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-d",
        "--dataset",
        choices=["libritts", "ljspeech", "librispeech"],
        default="libritts",
    )
    ap.add_argument(
        "--root",
        type=str,
        default="./data",
        help="path containing `LibriTTS` folder with data",
    )
    ap.add_argument(
        "--split",
        type=str,
        default="test-clean",
        help="Dataset split: LibriTTS url or LibriSpeech url (e.g., train-clean-100)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="./data",
        help="directory to store per-utt .pt latents",
    )
    ap.add_argument(
        "--sr", type=int, default=16000, help="target sample rate for caching"
    )
    ap.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.mps.is_available() else "cpu")
        ),
    )
    args = ap.parse_args()

    datasets = {
        "libritts": {"class": LIBRITTS, "folder": "LibriTTS"},
        "librispeech": {"class": LIBRISPEECH, "folder": "LibriSpeech"},
        "ljspeech": {"class": LJSPEECH, "folder": "LJSpeech-1.1"},
    }

    kwargs = {}
    if args.dataset in ("libritts", "librispeech"):
        kwargs["url"] = args.split

    ds = datasets[args.dataset]["class"](root=args.root, **kwargs)

    if args.dataset == "ljspeech":

        def __getitem__(self, n):
            fileid = self._flist[n][0]
            fileid_audio = self._path / (fileid + ".wav")

            waveform, sample_rate = torchaudio.load(fileid_audio)

            return waveform, sample_rate, fileid

        ds.__getitem__ = __getitem__

    bn_extractor = load_bn_extractor().to(args.device)
    outdir: Path = Path(args.out_dir) / datasets[args.dataset]["folder"] / "latents"
    if kwargs.get("url", None) is not None:
        outdir = outdir / kwargs["url"]

    outdir.mkdir(parents=True, exist_ok=True)

    for i, (waveform, sr, *_, utt_id) in enumerate(tqdm(ds)):
        x = T.Resample(sr, args.sr)(waveform).to(args.device)
        with torch.no_grad():
            reps = bn_extractor.get_bn(x)
        reps = reps.squeeze(0).permute(0, 1)

        if args.dataset == "ljspeech":
            utt_id = ds._flist[i][0]
        elif args.dataset == "librispeech":
            utt_id = ds._walker[i]

        torch.save(reps, outdir / f"{utt_id}.pt")


if __name__ == "__main__":
    main()
