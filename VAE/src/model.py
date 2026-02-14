import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNetVAE(nn.Module):
    def __init__(self, latent_dim=128):
        super(ResNetVAE, self).__init__()
        
        # --- Encoder ---
        # Load ResNet18 and modify first layer for 1-channel input
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final FC layer
        self.encoder_backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Latent projections
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)

        # --- Decoder ---
        # Upsample from latent_dim back to 224x224
        self.decoder_input = nn.Linear(latent_dim, 512 * 7 * 7)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), # 14x14
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 28x28
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 56x56
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 112x112
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),    # 224x224
            nn.Sigmoid() # Output pixels 0-1
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # Encode
        features = self.encoder_backbone(x)
        features = features.view(features.size(0), -1)
        
        mu = self.fc_mu(features)
        logvar = self.fc_var(features)
        
        # Sample
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoder_input = self.decoder_input(z)
        decoder_input = decoder_input.view(-1, 512, 7, 7)
        reconstruction = self.decoder(decoder_input)
        
        return reconstruction, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    # MSE Loss (Reconstruction)
    mse = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return mse + (beta * kld), mse, kld