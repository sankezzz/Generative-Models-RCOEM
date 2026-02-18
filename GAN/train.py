import torch
import torch.nn as nn
import torch.optim as optim
import csv
import os
import time
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from thop import profile 

from src.dataset import get_dataloaders
from src.model import Generator, Discriminator
from src.utils import initialize_weights, get_d_accuracy

# --- Config ---
DATASET_PATH = "../Dataset/brisc2025" 
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0002
Z_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Files
LOG_FILE = "gan_training_log.csv"
CHECKPOINT_PATH = "gan_latest.pth"          # <--- RESUME FILE (Overwritten)
BEST_MODEL_PATH = "best_generator.pth"      # <--- BEST MODEL (Saved only when good)
IMG_SAVE_DIR = "generated_images"

os.makedirs(IMG_SAVE_DIR, exist_ok=True)

def get_generator_flops(model, z_dim, device):
    """Calculates GFLOPS for the Generator."""
    dummy_input = torch.randn(1, z_dim).to(device)
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    return (2 * macs) / 1e9 

def save_checkpoint(gen, disc, opt_gen, opt_disc, epoch, filename=CHECKPOINT_PATH):
    """Saves full training state for resuming."""
    state = {
        'epoch': epoch,
        'gen_state_dict': gen.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'opt_gen_state_dict': opt_gen.state_dict(),
        'opt_disc_state_dict': opt_disc.state_dict(),
    }
    torch.save(state, filename)

def load_checkpoint(gen, disc, opt_gen, opt_disc, filename=CHECKPOINT_PATH):
    if os.path.exists(filename):
        print(f"--> Found checkpoint: {filename}. Resuming...")
        checkpoint = torch.load(filename, map_location=DEVICE)
        gen.load_state_dict(checkpoint['gen_state_dict'])
        disc.load_state_dict(checkpoint['disc_state_dict'])
        opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
        opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
        return checkpoint['epoch'] + 1
    return 1

def main():
    loader = get_dataloaders(DATASET_PATH, BATCH_SIZE)
    
    gen = Generator(Z_DIM).to(DEVICE)
    disc = Discriminator().to(DEVICE)
    initialize_weights(gen)
    initialize_weights(disc)
    
    opt_gen = optim.Adam(gen.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LR, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    fixed_noise = torch.randn(16, Z_DIM).to(DEVICE)
    
    # Static Metrics
    try:
        g_flops = get_generator_flops(gen, Z_DIM, DEVICE)
    except:
        g_flops = 0.0

    # CSV Headers
    headers = ["Epoch", "D_Loss", "G_Loss", "D_Accuracy", "Img_Variance", "Time(s)", "Latency(ms)", "VRAM(MB)", "GFLOPS"]
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(headers)

    start_epoch = load_checkpoint(gen, disc, opt_gen, opt_disc)
    best_variance = 0.0 # Heuristic for "Best" model (Highest diversity without collapse)

    print(f"Starting GAN Training on {DEVICE}...")

    for epoch in range(start_epoch - 1, EPOCHS):
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        gen.train()
        disc.train()
        
        epoch_d_loss = 0
        epoch_g_loss = 0
        epoch_acc = 0
        epoch_var = 0
        
        loop = tqdm(loader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for batch_idx, (real, _) in enumerate(loop):
            real = real.to(DEVICE)
            cur_batch_size = real.shape[0]
            
            # --- Train Discriminator ---
            noise = torch.randn(cur_batch_size, Z_DIM).to(DEVICE)
            fake = gen(noise)
            
            disc_real = disc(real).reshape(-1)
            loss_d_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake.detach()).reshape(-1)
            loss_d_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_d = (loss_d_real + loss_d_fake) / 2
            
            disc.zero_grad()
            loss_d.backward()
            opt_disc.step()
            
            # --- Train Generator ---
            output = disc(fake).reshape(-1)
            loss_g = criterion(output, torch.ones_like(output))
            
            gen.zero_grad()
            loss_g.backward()
            opt_gen.step()
            
            # Metrics
            epoch_d_loss += loss_d.item()
            epoch_g_loss += loss_g.item()
            epoch_acc += get_d_accuracy(disc_real, disc_fake)
            epoch_var += torch.var(fake).item()

            loop.set_postfix(acc=epoch_acc/(batch_idx+1))

        # --- End of Epoch Metrics ---
        avg_d_loss = epoch_d_loss / len(loader)
        avg_g_loss = epoch_g_loss / len(loader)
        avg_acc = epoch_acc / len(loader)
        avg_var = epoch_var / len(loader)
        duration = time.time() - start_time
        
        # Latency & VRAM
        start_infer = time.time()
        with torch.no_grad():
            _ = gen(fixed_noise[0:1]) 
        latency_ms = (time.time() - start_infer) * 1000
        
        vram_usage = 0
        if torch.cuda.is_available():
            vram_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)

        status = "🟢 OK"
        if avg_acc > 0.95: status = "🔴 D Winning"
        if avg_acc < 0.05: status = "🔴 G Winning"
        if avg_var < 0.001: status = "💀 COLLAPSED"
        
        print(f" -> D_Acc: {avg_acc:.3f} | Latency: {latency_ms:.3f}ms | VRAM: {vram_usage:.3f}MB | Status: {status}")
        
        # --- SAVE LOGIC ---
        
        # 1. ALWAYS save Resume Checkpoint (Overwrites previous)
        save_checkpoint(gen, disc, opt_gen, opt_disc, epoch+1)
        
        # 2. INTELLIGENT "Best Model" Save
        # Condition: Model must be stable (Acc 0.4-0.6) AND have high variance (Diversity)
        if 0.40 < avg_acc < 0.60 and avg_var > best_variance:
            best_variance = avg_var
            torch.save(gen.state_dict(), BEST_MODEL_PATH)
            print(f"    💎 New Best Generator Found! (Stable Acc & High Var: {avg_var:.4f})")

        # 3. Log to CSV
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch+1, 
                f"{avg_d_loss:.3f}", 
                f"{avg_g_loss:.3f}", 
                f"{avg_acc:.3f}", 
                f"{avg_var:.3f}", 
                f"{duration:.3f}", 
                f"{latency_ms:.3f}", 
                f"{vram_usage:.3f}", 
                f"{g_flops:.3f}"
            ])

        # 4. Save Visual Snapshot (Overwrites old snapshot to save space)
        if (epoch+1) % 5 == 0:
            with torch.no_grad():
                fake_images = gen(fixed_noise)
                fake_images = (fake_images * 0.5) + 0.5
                save_image(fake_images, f"{IMG_SAVE_DIR}/latest_generated.png")

if __name__ == "__main__":
    main()