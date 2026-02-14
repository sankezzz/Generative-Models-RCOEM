import time
import torch
import torch.optim as optim
import csv
import os
from tqdm import tqdm  # <--- NEW IMPORT
from src.dataset import get_dataloaders
from src.model import ResNetVAE, vae_loss_function 
from src.utils import calculate_psnr, get_flops, probe_accuracy_auc

# --- Configuration ---
DATASET_PATH = "../Dataset/brisc2025" 
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
LATENT_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_FILE = "training_log.csv"
BEST_MODEL_PATH = "best_brisc_vae.pth"

def main():
    print(f"Training on {DEVICE}")
    
    # 1. Prepare Data
    train_loader, test_loader = get_dataloaders(DATASET_PATH, BATCH_SIZE)
    
    # 2. Initialize Model
    model = ResNetVAE(latent_dim=LATENT_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3. Calculate Static Metrics (FLOPS)
    flops_g = get_flops(model, device=DEVICE)
    print(f"Model Complexity: {flops_g:.2f} GFLOPs")

    # 4. Initialize CSV Logging
    headers = ["Epoch", "Train_Loss", "Train_PSNR", "Val_Accuracy", "Val_AUC", "Time(s)", "Latency(ms)", "VRAM(MB)", "FLOPS(G)"]
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    best_val_accuracy = 0.0

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        train_loss = 0
        total_psnr = 0
        
        # --- TQDM Progress Bar Start ---
        # We wrap the train_loader with tqdm to create the bar
        loop = tqdm(train_loader, leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for batch_idx, (data, _) in enumerate(loop):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss, mse, kld = vae_loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            current_psnr = calculate_psnr(recon_batch, data).item()
            total_psnr += current_psnr
            
            # Update the progress bar text immediately
            loop.set_postfix(loss=loss.item(), psnr=current_psnr)
        # --- TQDM Progress Bar End ---

        # End of Epoch Timing
        epoch_duration = time.time() - start_time
        
        # Inference Latency Check
        infer_start = time.time()
        with torch.no_grad():
            _ = model(data[0:1]) 
        inference_latency = (time.time() - infer_start) * 1000 

        # VRAM Usage
        vram_usage = 0
        if torch.cuda.is_available():
            vram_usage = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Validation Probe
        # (We print this on a new line so it doesn't mess up the progress bar)
        val_acc, val_auc = probe_accuracy_auc(model, test_loader, DEVICE)

        avg_loss = train_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)

        # Print Summary
        print(f" -> Val Acc: {val_acc:.4f} | AUC: {val_auc:.4f} | Time: {epoch_duration:.2f}s")

        # Log to CSV
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{avg_loss:.4f}", f"{avg_psnr:.2f}", f"{val_acc:.4f}",
                f"{val_auc:.4f}", f"{epoch_duration:.2f}", f"{inference_latency:.2f}",
                f"{vram_usage:.2f}", f"{flops_g:.2f}"
            ])

        # Save Best Model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"   --> New Best Model Saved! (Acc: {best_val_accuracy:.4f})")

    print("Training Complete.")

if __name__ == "__main__":
    main()