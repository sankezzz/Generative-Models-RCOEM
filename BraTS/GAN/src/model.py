# model.py
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class ResBlockGen(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)

class Generator(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        self.init_size = 7
        self.l1 = nn.Sequential(nn.Linear(z_dim, 1024 * self.init_size ** 2))
        
        # 7 -> 14 -> 28 -> 56 -> 112 -> 224
        self.blocks = nn.Sequential(
            ResBlockGen(1024, 512),  # 14x14
            ResBlockGen(512, 256),   # 28x28
            ResBlockGen(256, 128),   # 56x56
            ResBlockGen(128, 64),    # 112x112
            ResBlockGen(64, 32),     # 224x224
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 1024, self.init_size, self.init_size)
        out = self.blocks(out)
        return self.final(out)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        def d_block(in_filters, out_filters, downsample=True):
            layers = [
                spectral_norm(nn.Conv2d(in_filters, out_filters, 4, 2 if downsample else 1, 1)),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            return layers

        # 224 -> 112 -> 56 -> 28 -> 14 -> 7
        self.model = nn.Sequential(
            *d_block(1, 64),
            *d_block(64, 128),
            *d_block(128, 256),
            *d_block(256, 512),
            *d_block(512, 1024),
            spectral_norm(nn.Conv2d(1024, 1, 7, 1, 0)) # Outputs a 1x1 scalar
        )

    def forward(self, x):
        out = self.model(x)
        return out.view(-1) # No Sigmoid for WGAN!