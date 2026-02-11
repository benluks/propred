import logging
import torch
import torch.nn as nn

"""
Taken from MatchaTTS: https://github.com/shivammehta25/Matcha-TTS/blob/main/matcha/models/components/text_encoder.py
"""


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = torch.nn.Parameter(torch.ones(channels))
        self.beta = torch.nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ConvDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        filter_channels=256,
        kernel_size=3,
        p_dropout=0.1,
        output_dim=1,
    ):
        super().__init__()
        self.filter_channels = filter_channels
        self.p_dropout = p_dropout

        self.drop = torch.nn.Dropout(p_dropout)
        self.conv_1 = torch.nn.Conv1d(
            input_dim, filter_channels, kernel_size, padding=(kernel_size - 1) // 2
        )
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = torch.nn.Conv1d(
            filter_channels,
            filter_channels,
            kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.norm_2 = LayerNorm(filter_channels)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)

        return x


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


if __name__ == "__main__":
    dp = DurationPredictor(
        "/Users/ben/dev/propred/data/LJSpeech-1.1/embeddings.pt",
    )
    x = torch.randint(47, (1, 403))
    x_mask = torch.ones_like(x)

    y = dp(x, x_mask)
