from collections.abc import Iterable
from pathlib import Path
import torch
import torch.nn as nn

import sys

from utils.config import load_spk_id_map

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # noqa: F821
sys.path.insert(0, str(PROJECT_ROOT))

from model.conv import CNN


class DurationPredictor(nn.Module):

    def __init__(
        self,
        embeddings_path,
        output_dim=1,
        do_log=False,
        spk_embedding_dim=None,
        spk_id_map=None,
        **conv_kwargs,
    ):
        super().__init__()
        embeddings_matrix = torch.load(embeddings_path)
        num_embeddings, embedding_dim = embeddings_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embeddings_matrix,
            freeze=True,
            padding_idx=conv_kwargs.pop("padding_idx", 0),
        )
        for p in self.embedding.parameters():
            p.requires_grad = False

        input_dim = embedding_dim

        self.use_spk_id = spk_id_map is not None
        if self.use_spk_id:
            spk_id_map = load_spk_id_map(spk_id_map)
            num_speakers = len(spk_id_map)
            self.spk_embedding = nn.Embedding(num_speakers, spk_embedding_dim)
            input_dim += spk_embedding_dim

        # for posterity
        self.encoder = CNN(input_dim, **conv_kwargs)

        self.proj = nn.Linear(self.encoder.output_dim, output_dim)
        self.do_log = do_log

    def forward(self, x, mask, spk_ids=None, **kwargs):

        with torch.no_grad():
            x = self.embedding(x * mask.detach().to(x.dtype))
        x = x.transpose(1, 2)

        if self.use_spk_id:
            assert (
                spk_ids is not None
            ), "spk_ids must be provided if DurationPredictor is initialized with `use_spk_id=True`"
            *_, T = x.shape
            # [B, D_spk]
            spk_emb = self.spk_embedding(spk_ids)
            spk_emb = spk_emb.unsqueeze(-1).expand(-1, -1, T)
            x = torch.cat([x, spk_emb], dim=1)

        x_mask = mask.unsqueeze(1)

        x = self.encoder(x, x_mask)

        x = self.proj(x.transpose(1, 2)).squeeze(-1)
        return x * x_mask.squeeze(1)


if __name__ == "__main__":
    from hyperpyyaml import load_hyperpyyaml

    cfg = load_hyperpyyaml(open("configs/log_spkid.yaml"))["model"]
    num_speakers = len(torch.load(cfg["spk_id_path"]))
    model = DurationPredictor(**cfg, num_speakers=num_speakers)

    input()
    x = torch.tensor([[1, 2, 3, 4, 0], [5, 6, 7, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    out = model(x, mask)
    print(out.shape)
