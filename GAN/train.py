# train.py
import torch
import torch.optim as optim
import csv
import os
import time
from tqdm import tqdm
from torchvision.utils import save_image

from src.dataset import get_dataloaders
from src.model import Generator, Discriminator

# --- Config ---
# Pointing exactly to the structure in your screenshot
DATASET_PATH = "Processed_BraTS_AnomalyDetection/Train/Healthy" 

BATCH_SIZE = 16 # Keep this low (16 or 32) as WGAN-GP with Spectral Norm is heavy on VRAM
EPOCHS = 200    
LR = 0.0002
Z_DIM = 128
CRITIC_ITERATIONS = 5 
LAMBDA_GP = 10        
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOG_FILE = "wgan_training_log.csv"
IMG_SAVE_DIR = "generated_images"
os.makedirs(IMG_SAVE_DIR, exist_ok=True)

def gradient_penalty(critic, real, fake, device):
    """Calculates the WGAN-GP gradient penalty."""
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)
    
    # Calculate critic scores
    mixed_scores = critic(interpolated_images)
    
    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean((gradient_norm - 1) ** 2)
    return penalty

def main():
    loader = get_dataloaders(DATASET_PATH, BATCH_SIZE)
    
    gen = Generator(Z_DIM).to(DEVICE)
    critic = Discriminator().to(DEVICE)
    
    # WGAN-GP optimal betas
    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LR, betas=(0.0, 0.9))
    
    fixed_noise = torch.randn(16, Z_DIM).to(DEVICE)
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Epoch", "Critic_Loss", "Gen_Loss"])

    print(f"Starting WGAN-GP Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        loop = tqdm(loader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for batch_idx, (real, _) in enumerate(loop):
            real = real.to(DEVICE)
            cur_batch_size = real.shape[0]
            
            # ---------------------
            # Train Critic
            # ---------------------
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(cur_batch_size, Z_DIM).to(DEVICE)
                fake = gen(noise)
                
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                
                gp = gradient_penalty(critic, real, fake, DEVICE)
                # WGAN Loss: Maximize E[critic(real)] - E[critic(fake)]
                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
                
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()
                
            # ---------------------
            # Train Generator
            # ---------------------
            gen_fake = critic(fake).reshape(-1)
            # Maximize E[critic(fake)]
            loss_gen = -torch.mean(gen_fake)
            
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
            loop.set_postfix(C_loss=loss_critic.item(), G_loss=loss_gen.item())

        # Save Visual Snapshot
        if (epoch+1) % 2 == 0:
            with torch.no_grad():
                fake_images = gen(fixed_noise)
                fake_images = (fake_images * 0.5) + 0.5 # De-normalize back to 0-1
                save_image(fake_images, f"{IMG_SAVE_DIR}/epoch_{epoch+1}.png", nrow=4)
                
        # Basic logging
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, loss_critic.item(), loss_gen.item()])

if __name__ == "__main__":
    main()