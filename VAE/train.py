import time
import torch
import torch.optim as optim
import csv
import os
from src.dataset import get_dataloaders
from src.model import ResNetVAE, vae_loss_function  # Or SimpleVAE if you chose that
from src.utils import calculate_psnr, get_flops, probe_accuracy_auc

# --- Configuration ---
DATASET_PATH = "data/brisc2025" 
BATCH_SIZE = 16
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
    
    # Create file and write headers if it doesn't exist
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    best_val_accuracy = 0.0

    # --- Training Loop ---
    for epoch in range(EPOCHS):
        model.train()
        start_time = time.time()
        
        # Reset VRAM tracking
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        train_loss = 0
        total_psnr = 0
        
        # Training Step
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(DEVICE)
            optimizer.zero_grad()
            
            recon_batch, mu, logvar = model(data)
            loss, mse, kld = vae_loss_function(recon_batch, data, mu, logvar)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            total_psnr += calculate_psnr(recon_batch, data).item()

        # End of Epoch Timing
        epoch_duration = time.time() - start_time
        
        # Inference Latency Check (One pass)
        infer_start = time.time()
        with torch.no_grad():
            _ = model(data[0:1]) # Single image inference
        inference_latency = (time.time() - infer_start) * 1000 # ms

        # VRAM Usage
        vram_usage = 0
        if torch.cuda.is_available():
            vram_usage = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB

        # Validation & Probing
        # We use Accuracy as the metric to decide "Best Model"
        val_acc, val_auc = probe_accuracy_auc(model, test_loader, DEVICE)

        avg_loss = train_loss / len(train_loader)
        avg_psnr = total_psnr / len(train_loader)

        # Console Log
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Loss: {avg_loss:.4f} | PSNR: {avg_psnr:.2f}dB | "
              f"Latent Acc: {val_acc:.4f} | AUC: {val_auc:.4f}")

        # --- CSV Logging ---
        with open(LOG_FILE, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{avg_loss:.4f}",
                f"{avg_psnr:.2f}",
                f"{val_acc:.4f}",
                f"{val_auc:.4f}",
                f"{epoch_duration:.2f}",
                f"{inference_latency:.2f}",
                f"{vram_usage:.2f}",
                f"{flops_g:.2f}"
            ])

        # --- Save Best Model ---
        # We save if the Unsupervised Latent Accuracy improves
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"--> Best Model Saved (Accuracy: {best_val_accuracy:.4f})")

    print("Training Complete. Check 'training_log.csv' for details.")

if __name__ == "__main__":
    main()