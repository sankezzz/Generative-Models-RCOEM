import time
import torch
import torch.optim as optim
import csv
import os
from tqdm import tqdm
from src.dataset import get_dataloaders
from src.model import ResNetVAE, vae_loss_function 
from src.utils import calculate_psnr, get_flops, probe_accuracy_auc

# --- Configuration ---
DATASET_PATH = "../Dataset/brisc2025" 
BATCH_SIZE = 32
EPOCHS = 100  # Updated to 100 since you are continuing
LR = 1e-4     # <--- LOWERED LR for fine-tuning (as discussed)
LATENT_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File Paths
LOG_FILE = "training_log.csv"
CHECKPOINT_PATH = "checkpoint.pth"

# Two separate best models
BEST_ACC_MODEL_PATH = "best_acc_brisc_vae.pth"
BEST_PSNR_MODEL_PATH = "best_psnr_brisc_vae.pth"

def save_checkpoint(model, optimizer, epoch, best_acc, best_psnr, filename=CHECKPOINT_PATH):
    """Saves training state including both best scores."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_accuracy': best_acc,
        'best_val_psnr': best_psnr
    }
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename=CHECKPOINT_PATH):
    """Robust loader that handles old and new checkpoints."""
    if os.path.exists(filename):
        print(f"--> Found checkpoint: {filename}")
        checkpoint = torch.load(filename, map_location=DEVICE)
        
        # Check if it's a full checkpoint or just weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
            # Safely get values (defaults to 0.0 for old checkpoints)
            best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
            best_val_psnr = checkpoint.get('best_val_psnr', 0.0)
            
            print(f"--> Resuming from Epoch {start_epoch}")
            print(f"    (Best Acc: {best_val_accuracy:.4f} | Best PSNR: {best_val_psnr:.2f})")
            return start_epoch, best_val_accuracy, best_val_psnr
        else:
            # Fallback for legacy weight files
            print("--> Detected legacy weight file. Starting fresh optimizer.")
            model.load_state_dict(checkpoint)
            return 1, 0.0, 0.0
    else:
        print("--> No checkpoint found. Starting from scratch.")
        return 1, 0.0, 0.0

def main():
    print(f"Training on {DEVICE}")
    
    # 1. Prepare Data
    train_loader, test_loader = get_dataloaders(DATASET_PATH, BATCH_SIZE)
    
    # 2. Initialize Model & Optimizer
    model = ResNetVAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3. Metrics & Logging
    flops_g = get_flops(model, device=DEVICE)
    print(f"Model Complexity: {flops_g:.2f} GFLOPs")

    headers = ["Epoch", "Train_Loss", "Train_PSNR", "Val_Accuracy", "Val_AUC", "Time(s)", "Latency(ms)", "VRAM(MB)", "FLOPS(G)"]
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            csv.writer(f).writerow(headers)

    # 4. Resume
    start_epoch, best_val_accuracy, best_val_psnr = load_checkpoint(model, optimizer)

    # --- Training Loop ---
    try:
        for epoch in range(start_epoch - 1, EPOCHS):
            model.train()
            start_time = time.time()
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            train_loss = 0
            total_psnr = 0
            
            loop = tqdm(train_loader, leave=True)
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            
            for batch_idx, (data, _) in enumerate(loop):
                data = data.to(DEVICE)
                optimizer.zero_grad()
                
                recon_batch, mu, logvar = model(data)
                loss, mse, kld = vae_loss_function(recon_batch, data, mu, logvar)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                current_psnr = calculate_psnr(recon_batch, data).item()
                total_psnr += current_psnr
                
                loop.set_postfix(loss=loss.item(), psnr=current_psnr)

            # End of Epoch Stats
            epoch_duration = time.time() - start_time
            infer_start = time.time()
            with torch.no_grad(): _ = model(data[0:1]) 
            inference_latency = (time.time() - infer_start) * 1000 
            
            vram_usage = 0
            if torch.cuda.is_available():
                vram_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)

            val_acc, val_auc = probe_accuracy_auc(model, test_loader, DEVICE)
            avg_loss = train_loss / len(train_loader)
            avg_psnr = total_psnr / len(train_loader)

            print(f" -> Acc: {val_acc:.4f} | PSNR: {avg_psnr:.2f} | Time: {epoch_duration:.2f}s")

            # Logging
            with open(LOG_FILE, mode='a', newline='') as f:
                csv.writer(f).writerow([
                    epoch + 1, f"{avg_loss:.4f}", f"{avg_psnr:.2f}", f"{val_acc:.4f}",
                    f"{val_auc:.4f}", f"{epoch_duration:.2f}", f"{inference_latency:.2f}",
                    f"{vram_usage:.2f}", f"{flops_g:.2f}"
                ])

            # --- SAVE LOGIC (Dual Saving) ---
            
            # 1. Save if ACCURACY improves
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                torch.save(model.state_dict(), BEST_ACC_MODEL_PATH)
                print(f"   🏆 New Best Accuracy! ({best_val_accuracy:.4f}) -> Saved {BEST_ACC_MODEL_PATH}")

            # 2. Save if PSNR improves (NEW!)
            if avg_psnr > best_val_psnr:
                best_val_psnr = avg_psnr
                torch.save(model.state_dict(), BEST_PSNR_MODEL_PATH)
                print(f"   💎 New Best Quality! ({best_val_psnr:.2f} dB) -> Saved {BEST_PSNR_MODEL_PATH}")
            
            # 3. Always update checkpoint
            save_checkpoint(model, optimizer, epoch + 1, best_val_accuracy, best_val_psnr)

    except KeyboardInterrupt:
        print("\n\n⚠️ Training Interrupted! Saving Emergency Checkpoint...")
        save_checkpoint(model, optimizer, epoch, best_val_accuracy, best_val_psnr)
        print("✅ Checkpoint saved.")

    print("Training Complete.")

if __name__ == "__main__":
    main()