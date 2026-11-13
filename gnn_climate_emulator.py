#!/usr/bin/env python3
# gnn_climate_emulator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, LayerNorm


class TriangularPrismMessagePassingBlock(nn.Module):
    """Message passing block simulating horizontal & vertical flux between prism faces."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1 = GCNConv(in_features, out_features)
        self.norm1 = LayerNorm(out_features)
        self.conv2 = GCNConv(out_features, out_features)
        self.norm2 = LayerNorm(out_features)
        self.act = nn.GELU()

    def forward(self, x, edge_index):
        residual = x if x.shape[-1] == self.conv1.out_channels else None
        h = self.act(self.norm1(self.conv1(x, edge_index)))
        h = self.norm2(self.conv2(h, edge_index))
        if residual is not None:
            h = h + residual
        return self.act(h)


class TriangularGNNClimateEmulator(nn.Module):
    def __init__(self, in_channels=12, hidden_channels=64, out_channels=4):
        super().__init__()
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.block1 = TriangularPrismMessagePassingBlock(hidden_channels, hidden_channels)
        self.block2 = TriangularPrismMessagePassingBlock(hidden_channels, hidden_channels * 2)
        self.block3 = TriangularPrismMessagePassingBlock(hidden_channels * 2, hidden_channels)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.GELU(),
            nn.Linear(32, out_channels)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = F.gelu(self.encoder(x))
        h = self.block1(h, edge_index)
        h = self.block2(h, edge_index)
        h = self.block3(h, edge_index)
        return self.decoder(h)
