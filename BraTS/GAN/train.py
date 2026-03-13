import torch
import torch.optim as optim
import csv
import os
import time
from tqdm import tqdm
from torchvision.utils import save_image

from src.dataset import get_dataloaders
from src.model import Generator, Discriminator
from src.utils import get_d_accuracy 

# --- Config ---
DATASET_PATH = "../Dataset/Processed_BraTS_AnomalyDetection/Train/Healthy" 
BATCH_SIZE = 32 
EPOCHS = 200    
LR = 0.0002
Z_DIM = 128
CRITIC_ITERATIONS = 5 
LAMBDA_GP = 10        
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File Paths
LOG_FILE = "wgan_training_log.csv"
IMG_SAVE_DIR = "generated_images"
CHECKPOINT_GEN = "generator_checkpoint.pth"
CHECKPOINT_CRITIC = "critic_checkpoint.pth"
BEST_GEN = "best_generator.pth"

os.makedirs(IMG_SAVE_DIR, exist_ok=True)

def gradient_penalty(critic, real, fake, device):
    """Calculates the WGAN-GP gradient penalty."""
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake * (1 - alpha)
    mixed_scores = critic(interpolated_images)
    
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    
    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    return torch.mean((gradient_norm - 1) ** 2)

def save_checkpoint(model, optimizer, filename):
    """Saves weights and optimizer state."""
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer, lr):
    """Loads weights and optimizer state to resume training."""
    if os.path.exists(filename):
        print(f"=> Loading checkpoint: {filename}")
        checkpoint = torch.load(filename, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        
        # Ensure the learning rate is correct after loading
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return True
    return False

def main():
    loader = get_dataloaders(DATASET_PATH, BATCH_SIZE)
    gen = Generator(Z_DIM).to(DEVICE)
    critic = Discriminator().to(DEVICE)
    
    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LR, betas=(0.0, 0.9))
    
    # --- AUTO-RESUME LOGIC ---
    start_epoch = 0
    if load_checkpoint(CHECKPOINT_GEN, gen, opt_gen, LR) and \
       load_checkpoint(CHECKPOINT_CRITIC, critic, opt_critic, LR):
        # Try to determine the last epoch from the log file
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    start_epoch = int(lines[-1].split(',')[0])
                    print(f"=> Resuming from Epoch {start_epoch + 1}")

    fixed_noise = torch.randn(16, Z_DIM).to(DEVICE)
    best_diff = float('inf')
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Epoch", "D_Loss", "G_Loss", "D_Accuracy", "Img_Variance", "Time(s)", "Latency(ms)", "VRAM(MB)"])

    for epoch in range(start_epoch, EPOCHS):
        epoch_start_time = time.time()
        loop = tqdm(loader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for batch_idx, (real, _) in enumerate(loop):
            step_start = time.time()
            real = real.to(DEVICE)
            
            # Train Critic
            for _ in range(CRITIC_ITERATIONS):
                noise = torch.randn(real.shape[0], Z_DIM).to(DEVICE)
                fake = gen(noise)
                critic_real = critic(real).reshape(-1)
                critic_fake = critic(fake).reshape(-1)
                gp = gradient_penalty(critic, real, fake, DEVICE)
                loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()
                
            # Train Generator
            gen_fake = critic(fake).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
            latency_ms = (time.time() - step_start) * 1000
            d_acc = get_d_accuracy(critic_real, critic_fake)
            img_var = torch.var(fake).item()
            loop.set_postfix(C_loss=f"{loss_critic.item():.3f}", G_loss=f"{loss_gen.item():.3f}")

        # Save Checkpoints
        save_checkpoint(gen, opt_gen, CHECKPOINT_GEN)
        save_checkpoint(critic, opt_critic, CHECKPOINT_CRITIC)

        # Track Best Model
        current_diff = abs(d_acc - 0.5)
        if current_diff < best_diff:
            best_diff = current_diff
            torch.save(gen.state_dict(), BEST_GEN)

        # Performance and Logging
        vram_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024 * 1024) if torch.cuda.is_available() else 0
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([epoch + 1, f"{loss_critic.item():.3f}", f"{loss_gen.item():.3f}", f"{d_acc:.3f}", f"{img_var:.3f}", f"{(time.time()-epoch_start_time):.3f}", f"{latency_ms:.3f}", f"{vram_mb:.3f}"])
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(DEVICE)

if __name__ == "__main__":
    main()