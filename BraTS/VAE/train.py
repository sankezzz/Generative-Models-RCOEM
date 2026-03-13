import time
import torch
import torch.optim as optim
import csv
import os
from tqdm import tqdm
from pytorch_msssim import ssim # Added to track our new structural metric

from src.dataset import get_dataloaders
from src.model import ResNetVAE, vae_loss_function 
from src.utils import calculate_psnr, get_flops

# --- Configuration ---
TRAIN_DIR = "../Dataset/Processed_BraTS_AnomalyDetection/Train/Healthy"
VAL_DIR = "../Dataset/Processed_BraTS_AnomalyDetection/Val/Healthy"

BATCH_SIZE = 32
EPOCHS = 100 
LR = 1e-4     
LATENT_DIM = 256  
BETA = 0.1        
ALPHA = 0.85 # The SSIM/L1 balance parameter
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File Paths
LOG_FILE = "vae_anomaly_log.csv"
CHECKPOINT_PATH = "vae_checkpoint.pth"
BEST_SSIM_MODEL_PATH = "best_anomaly_vae_ssim.pth" # Renamed to reflect new save logic

def save_checkpoint(model, optimizer, epoch, best_ssim, filename=CHECKPOINT_PATH):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_ssim': best_ssim # Track SSIM instead of PSNR
    }
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename=CHECKPOINT_PATH):
    if os.path.exists(filename):
        print(f"--> Found checkpoint: {filename}")
        checkpoint = torch.load(filename, map_location=DEVICE)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_ssim = checkpoint.get('best_val_ssim', 0.0)
            
            print(f"--> Resuming from Epoch {start_epoch} (Best Val SSIM: {best_val_ssim:.4f})")
            return start_epoch, best_val_ssim
        else:
            model.load_state_dict(checkpoint)
            return 1, 0.0
    return 1, 0.0

def main():
    print(f"Training Edge-Aware Anomaly VAE on {DEVICE}")
    
    train_loader, val_loader = get_dataloaders(TRAIN_DIR, VAL_DIR, BATCH_SIZE)
    
    model = ResNetVAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    try:
        flops_g = get_flops(model, device=DEVICE)
        print(f"Model Complexity: {flops_g:.2f} GFLOPs")
    except:
        flops_g = 0.0

    # Added Train_SSIM and Val_SSIM to logs
    headers = [
        "Epoch", "Train_Loss", "Recon_Loss", "KLD", "Train_PSNR", "Train_SSIM",
        "Val_Recon", "Val_PSNR", "Val_SSIM", "Time(s)", "Latency(ms)", "VRAM(MB)", "FLOPS(G)"
    ]
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            csv.writer(f).writerow(headers)

    start_epoch, best_val_ssim = load_checkpoint(model, optimizer)

    try:
        for epoch in range(start_epoch - 1, EPOCHS):
            start_time = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            # --- 1. TRAINING LOOP ---
            model.train()
            train_total_loss = 0
            train_recon_loss = 0
            train_kld_loss = 0
            train_psnr_total = 0
            train_ssim_total = 0
            
            train_loop = tqdm(train_loader, leave=False)
            train_loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}] Train")
            
            for data, _ in train_loop:
                data = data.to(DEVICE)
                optimizer.zero_grad()
                
                recon_batch, mu, logvar = model(data)
                
                # Using the new loss function with alpha
                loss, recon, kld = vae_loss_function(recon_batch, data, mu, logvar, beta=BETA, alpha=ALPHA)
                
                loss.backward()
                optimizer.step()
                
                train_total_loss += loss.item()
                train_recon_loss += recon.item()
                train_kld_loss += kld.item()
                
                train_psnr_total += calculate_psnr(recon_batch, data).item()
                train_ssim_total += ssim(recon_batch, data, data_range=1.0, size_average=True).item()

            avg_train_loss = train_total_loss / len(train_loader)
            avg_train_recon = train_recon_loss / len(train_loader)
            avg_train_kld = train_kld_loss / len(train_loader)
            avg_train_psnr = train_psnr_total / len(train_loader)
            avg_train_ssim = train_ssim_total / len(train_loader)

            # --- 2. VALIDATION LOOP ---
            model.eval()
            val_recon_loss = 0
            val_psnr_total = 0
            val_ssim_total = 0
            
            with torch.no_grad():
                for data_val, _ in val_loader:
                    data_val = data_val.to(DEVICE)
                    recon_batch_val, mu_val, logvar_val = model(data_val)
                    
                    _, recon_val, _ = vae_loss_function(recon_batch_val, data_val, mu_val, logvar_val, beta=BETA, alpha=ALPHA)
                    
                    val_recon_loss += recon_val.item()
                    val_psnr_total += calculate_psnr(recon_batch_val, data_val).item()
                    val_ssim_total += ssim(recon_batch_val, data_val, data_range=1.0, size_average=True).item()

            avg_val_recon = val_recon_loss / len(val_loader)
            avg_val_psnr = val_psnr_total / len(val_loader)
            avg_val_ssim = val_ssim_total / len(val_loader)
            
            epoch_time = time.time() - start_time

            # --- 3. HARDWARE METRICS ---
            infer_start = time.time()
            with torch.no_grad(): 
                _ = model(data[0:1]) 
            inference_latency = (time.time() - infer_start) * 1000 
            
            vram_usage = 0
            if torch.cuda.is_available():
                vram_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)

            # --- 4. LOGGING ---
            print(f"Epoch {epoch+1:03d}/{EPOCHS} | "
                  f"Loss: {avg_train_loss:.2f} (Recon: {avg_train_recon:.2f}, KLD: {avg_train_kld:.2f}) | "
                  f"Train SSIM: {avg_train_ssim:.4f} | "
                  f"Val SSIM: {avg_val_ssim:.4f} (PSNR: {avg_val_psnr:.2f}dB) | "
                  f"{epoch_time:.1f}s")

            with open(LOG_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow([
                    epoch + 1, 
                    f"{avg_train_loss:.4f}", 
                    f"{avg_train_recon:.4f}", 
                    f"{avg_train_kld:.4f}", 
                    f"{avg_train_psnr:.2f}", 
                    f"{avg_train_ssim:.4f}",
                    f"{avg_val_recon:.4f}", 
                    f"{avg_val_psnr:.2f}", 
                    f"{avg_val_ssim:.4f}",
                    f"{epoch_time:.1f}",
                    f"{inference_latency:.2f}",
                    f"{vram_usage:.2f}",
                    f"{flops_g:.2f}"
                ])

            # --- 5. SAVE LOGIC (NOW BASED ON SSIM) ---
            if avg_val_ssim > best_val_ssim:
                best_val_ssim = avg_val_ssim
                torch.save(model.state_dict(), BEST_SSIM_MODEL_PATH)
                print(f"  💎 New Best Structural Integrity! Val SSIM: {best_val_ssim:.4f} -> Saved {BEST_SSIM_MODEL_PATH}")
            
            save_checkpoint(model, optimizer, epoch + 1, best_val_ssim)

    except KeyboardInterrupt:
        print("\n\n⚠️ Training Interrupted! Saving Emergency Checkpoint...")
        save_checkpoint(model, optimizer, epoch, best_val_ssim)

if __name__ == "__main__":
    main()