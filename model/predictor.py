from collections.abc import Iterable
from pathlib import Path
import torch
import torch.nn as nn

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.conv_decoder import ConvDecoder


class DurationPredictor(nn.Module):

    def __init__(
        self,
        embeddings_path,
        output_dim=1,
        n_layers=1,
        do_log=False,
        use_spk_id=False,
        num_speakers=None,
        spk_embedding_dim=None,
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

        self.use_spk_id = use_spk_id
        if self.use_spk_id:
            self.spk_embedding = nn.Embedding(num_speakers, spk_embedding_dim)
            input_dim += spk_embedding_dim

        # for posterity
        if n_layers == 1:
            self.conv = ConvDecoder(input_dim, **conv_kwargs)
            features = self.conv.filter_channels
        else:
            conv = []

            for _ in range(n_layers):
                layer = ConvDecoder(input_dim, **conv_kwargs)
                conv.append(layer)
                input_dim = layer.filter_channels

            self.conv = nn.Sequential(*conv)
            features = layer.filter_channels

        self.proj = self.proj = nn.Linear(features, output_dim)
        self.do_log = do_log

    def forward(self, x, mask, spk_ids=None):

        with torch.no_grad():
            x = self.embedding(x * mask.detach().to(x.dtype))
        x = x.transpose(1, 2)

        if self.use_spk_id:
            assert (
                spk_ids is not None
            ), "spk_ids must be provided if DurationPredictor is initialized with `use_spk_id=True`"
            B, T = spk_ids.shape
            # [B, D_spk]
            spk_emb = self.spk_embedding(spk_ids)
            spk_emb = spk_emb.unsqueeze(-1).expand(-1, -1, T)
            x = torch.cat([x, spk_emb], dim=1)

        x_mask = mask.unsqueeze(1)

        if not isinstance(self.conv, Iterable):
            x = self.conv(x, x_mask)
        else:
            for layer in self.conv:
                res = x
                x = layer(x, x_mask)
                x = (x + res) * x_mask

        x = self.proj(x.transpose(1, 2)).squeeze(-1)
        return x * x_mask.squeeze(1)


class ProsodyPredictor(nn.Module):
    """
    Wrapper that:
      1) embeds speaker ids (from an index) + optionally projects reps
      2) concatenates reps and speaker embedding along channel dim -> [B, D_rep + D_spk, T]
      3) feeds trunk (duration-predictor-style conv stack)
      4) predicts log-f0, vuv, rms energy as [B, T]
    """

    def __init__(
        self,
        embeddings_path: str | Path,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        rep_proj_dim: int | None = None,  # set to rep_dim to keep same; or None to skip
        heads=["f0", "vuv"],
    ):
        super().__init__()

        embeddings_matrix = torch.load(embeddings_path)
        self.embedding = nn.Embedding.from_pretrained(embeddings_matrix, freeze=True)
        for p in self.embedding.parameters():
            p.requires_grad = False

        # Optional rep projection (handy if your reps are huge and you want a smaller trunk)
        emb_dim = embeddings_matrix.shape[-1]
        if rep_proj_dim is not None:
            self.rep_proj = nn.Conv1d(emb_dim, rep_proj_dim, kernel_size=1)
            emb_dim = rep_proj_dim
        else:
            self.rep_proj = None

        self.conv = ConvDecoder(emb_dim, filter_channels, kernel_size, p_dropout)

        self.heads = []
        for head_feature in heads:
            head = nn.Conv1d(filter_channels, 1, kernel_size=1)
            setattr(
                self,
                f"{head_feature}_head",
                head,
            )
            self.heads.append(head)

    def forward(self, x, x_mask):
        """
        reps:   [B, D_rep, T]
        spk_id: [B] (int64 speaker indices)
        x_mask: [B, 1, T] (1 for valid frames, 0 for padding)

        returns:
          log_f0: [B, T]
          vuv:    [B, T] (logits or prob depending on vuv_output)
          rms:    [B, T]
        """

        with torch.no_grad():
            x = self.embedding(x * x_mask)
        x = x.transpose(1, 2)

        if self.rep_proj is not None:
            x = self.rep_proj(x)
        x_mask = x_mask.unsqueeze(1)
        x = self.conv(x, x_mask)

        head_outputs = [(head(x) * x_mask).squeeze(1) for head in self.heads]
        return tuple(head_outputs)


if __name__ == "__main__":

    pp = ProsodyPredictor(
        "/Users/ben/dev/propred/data/LJSpeech-1.1/embeddings.pt", 256, 3, 0.1
    )
    x = torch.randint(47, (1, 403))
    x_mask = torch.ones_like(x)

    y = pp(x, x_mask)
    print("hi")
