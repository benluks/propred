from pathlib import Path
import torch
import torch.nn as nn
import logging

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from model.conv_decoder import ConvDecoder


class DurationPredictor(nn.Module):

    def __init__(self, embeddings_path, output_dim=1, **conv_kwargs):
        super().__init__()
        embeddings_matrix = torch.load(embeddings_path)
        num_embeddings, embedding_dim = embeddings_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(embeddings_matrix, freeze=True)
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.conv = ConvDecoder(embedding_dim, **conv_kwargs)
        self.proj = self.proj = nn.Linear(self.conv.filter_channels, output_dim)

    def forward(self, x, x_mask):

        x_mask = x_mask.unsqueeze(1)

        with torch.no_grad():
            x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.conv(x, x_mask)

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
            x = self.embedding(x)
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
    print('hi')
