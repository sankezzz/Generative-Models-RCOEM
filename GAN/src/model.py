import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, downsample=False):
        super().__init__()
        self.upsample = upsample
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection handling
        self.shortcut = nn.Sequential()
        if in_channels != out_channels or upsample or downsample:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        if self.upsample:
            h = torch.nn.functional.interpolate(h, scale_factor=2)
            x = torch.nn.functional.interpolate(x, scale_factor=2)
        
        h = self.bn2(self.conv2(h))
        
        if self.downsample:
            h = torch.nn.functional.avg_pool2d(h, 2)
            x = torch.nn.functional.avg_pool2d(x, 2)
            
        return h + self.shortcut(x)

class Generator(nn.Module):
    def __init__(self, z_dim=128):
        super().__init__()
        # Initial projection: 128 -> 512x7x7
        self.linear = nn.Linear(z_dim, 512 * 7 * 7)
        
        # Upsampling blocks (7 -> 14 -> 28 -> 56 -> 112 -> 224)
        self.blocks = nn.Sequential(
            ResBlock(512, 256, upsample=True), # 14x14
            ResBlock(256, 128, upsample=True), # 28x28
            ResBlock(128, 64, upsample=True),  # 56x56
            ResBlock(64, 32, upsample=True),   # 112x112
            ResBlock(32, 16, upsample=True),   # 224x224
        )
        
        self.final = nn.Sequential(
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Tanh() # Output -1 to 1
        )

    def forward(self, z):
        x = self.linear(z)
        x = x.view(-1, 512, 7, 7)
        x = self.blocks(x)
        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Input 224x224
        self.initial = nn.Conv2d(1, 16, 3, 1, 1) # 224
        
        # Downsampling blocks (224 -> 112 -> 56 -> 28 -> 14 -> 7)
        self.blocks = nn.Sequential(
            ResBlock(16, 32, downsample=True),   # 112
            ResBlock(32, 64, downsample=True),   # 56
            ResBlock(64, 128, downsample=True),  # 28
            ResBlock(128, 256, downsample=True), # 14
            ResBlock(256, 512, downsample=True), # 7
        )
        
        self.final = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1),
            nn.Sigmoid() # Real (1) vs Fake (0)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.blocks(x)
        return self.final(x)