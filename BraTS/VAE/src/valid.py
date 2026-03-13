"""
anomaly_validation.py
─────────────────────
Computes live ROC-AUC during training so the model is saved based on
actual anomaly detection performance, not just reconstruction quality.

Called once per epoch from train.py (after the warmup period).
"""

import torch
import torch.nn.functional as F
import numpy as np
from pytorch_msssim import ssim
from sklearn.metrics import roc_auc_score


def compute_anomaly_score(model, dataloader, device):
    """
    Returns one scalar anomaly score per image across the full dataloader.

    Score = 0.85 * (1 - SSIM) + 0.15 * L1  — same formula as training loss.
    Higher score = more anomalous.

    FIX — per-image SSIM:
    ─────────────────────
    pytorch_msssim's size_average=False does NOT reliably return a (B,)
    tensor — its behaviour depends on the library version and can return
    a scalar or a wrong shape. The correct approach is to loop over each
    image in the batch individually on the (1,1,H,W) slice. Slightly
    slower but guaranteed correct for every version of the library.
    """
    model.eval()
    scores = []

    with torch.no_grad():
        for data, _ in dataloader:
            data  = data.to(device)       # (B, 1, H, W)
            recon, _, _ = model(data)     # (B, 1, H, W)

            for i in range(data.size(0)):
                img_real  = data[i:i+1]   # (1, 1, H, W)
                img_recon = recon[i:i+1]  # (1, 1, H, W)

                ssim_val  = ssim(img_recon, img_real, data_range=1.0, size_average=True)
                ssim_loss = 1.0 - ssim_val.item()
                l1_loss   = F.l1_loss(img_recon, img_real, reduction='mean').item()

                scores.append(0.85 * ssim_loss + 0.15 * l1_loss)

    return np.array(scores)


def validate_anomaly_separation(model, healthy_loader, abnormal_loader, device):
    """
    Computes ROC-AUC between healthy and abnormal reconstruction scores.

    Returns:
        auc           (float) – ROC-AUC. 0.5 = random, 1.0 = perfect.
        healthy_mean  (float) – mean score on healthy images  (want LOW)
        abnormal_mean (float) – mean score on abnormal images (want HIGH)
    """
    healthy_scores  = compute_anomaly_score(model, healthy_loader,  device)
    abnormal_scores = compute_anomaly_score(model, abnormal_loader, device)

    y_true  = np.concatenate([
        np.zeros(len(healthy_scores)),
        np.ones(len(abnormal_scores))
    ])
    y_score = np.concatenate([healthy_scores, abnormal_scores])

    if len(np.unique(y_true)) < 2:
        return 0.5, float(np.mean(healthy_scores)), float(np.mean(abnormal_scores))

    auc = roc_auc_score(y_true, y_score)
    return float(auc), float(np.mean(healthy_scores)), float(np.mean(abnormal_scores))