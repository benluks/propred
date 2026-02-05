# Use a pipeline as a high-level helper
from pathlib import Path
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline


DEVICE = "mps"


def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis


def get_emission(audio, model, device=DEVICE):
    with torch.inference_mode():
        waveform, _ = torchaudio.load(audio)
        emissions = model(waveform.to(device)).logits
        emissions = torch.log_softmax(emissions, dim=-1)

    return emissions[0].cpu().detach()


def get_token_spans(tokens):
    """
    Run-length encode token sequence.

    Returns:
        list of dicts:
            token
            start
            end
            duration
    """

    spans = []

    if not tokens:
        return spans

    current = tokens[0]
    start = 0

    for i in range(1, len(tokens)):
        if tokens[i] != current:
            spans.append(
                {"token": current, "start": start, "end": i, "duration": i - start}
            )

            current = tokens[i]
            start = i

    # final span
    spans.append(
        {
            "token": current,
            "start": start,
            "end": len(tokens),
            "duration": len(tokens) - start,
        }
    )

    return spans


def get_alignment(audio, model, tokenizer):
    emission = get_emission(audio, model)
    greedy = emission.argmax(dim=-1)
    greedy_decoded = tokenizer.convert_ids_to_tokens(greedy)
    tok_spans = get_token_spans(greedy_decoded)

    return tok_spans


if __name__ == "__main__":

    # pipe = pipeline(
    #     "automatic-speech-recognition", model="Bluecast/wav2vec2-Phoneme", device="mps"
    # )

    processor = Wav2Vec2Processor.from_pretrained(
        "facebook/wav2vec2-base-960h", device="mps"
    )
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to(DEVICE)

    for f in (Path(__file__).parent / "audio").glob("*.wav"):
        id = f.stem

        alignment = get_alignment(f, model, processor.tokenizer)
        torch.save(
            alignment, Path(__file__).parent / "alignment" / "chars" / f"{id}.pt"
        )
