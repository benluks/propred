from typing import List

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
        input_dim,
        filter_channels: List = [256],
        **kwargs,
    ):
        super().__init__()
        conv = []
        # for posterity

        conv = []

        if isinstance(filter_channels, int):
            filter_channels = [filter_channels]

        for filter_channels in filter_channels:
            layer = ConvDecoder(input_dim, filter_channels=filter_channels, **kwargs)
            conv.append(layer)
            input_dim = layer.filter_channels

        self.conv = nn.Sequential(*conv)

    def forward(self, x, x_mask):

        for layer in self.conv:
            res = x
            x = layer(x, x_mask)
            x = (x + res) * x_mask

        return x
