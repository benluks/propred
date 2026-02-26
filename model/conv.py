from typing import List

import torch
import torch.nn as nn

from utils.config import load_spk_id_map

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


class Conv(nn.Module):
    def __init__(
        self, input_dim, filter_channels=256, kernel_size=3, p_dropout=0.1, **kwargs
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


class CNN(nn.Module):
    def __init__(
        self,
        input_dim=256,
        filter_channels: List = [256],
        **kwargs,
    ):
        super().__init__()

        layers = []
        projections = []

        if isinstance(filter_channels, int):
            filter_channels = [filter_channels]

        for i, filter_channels in enumerate(filter_channels):
            layer = Conv(input_dim, filter_channels=filter_channels, **kwargs)
            layers.append(layer)
            if i == 0:
                projections.append(None)
            elif input_dim == filter_channels:
                projections.append(nn.Identity())
            else:
                projections.append(
                    torch.nn.Conv1d(input_dim, filter_channels, kernel_size=1)
                )

            input_dim = layer.filter_channels

        self.layers = nn.Sequential(*layers)
        if len(layers) > 1:
            self.projections = nn.ModuleList(projections)
        self.output_dim = input_dim

    def forward(self, x, x_mask):

        for i, (layer, proj) in enumerate(zip(self.layers, self.projections)):
            if i > 0:
                res = proj(x) * x_mask
            x = layer(x, x_mask)
            if i > 0:
                x = (x + res) * x_mask

        return x
