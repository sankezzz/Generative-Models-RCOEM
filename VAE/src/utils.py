import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from thop import profile

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def get_flops(model, input_size=(1, 1, 224, 224), device='cpu'):
    dummy_input = torch.randn(input_size).to(device)
    macs, params = profile(model, inputs=(dummy_input, ), verbose=False)
    # 1 MAC = 2 FLOPs usually
    flops_g = (2 * macs) / 1e9
    return flops_g

def probe_accuracy_auc(model, dataloader, device):
    """
    Extracts latent vectors (z) and trains a quick Logistic Regression
    to see if the unsupervised model learned class features.
    """
    model.eval()
    latents = []
    labels = []
    
    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            mu, _ = model.encoder_backbone(imgs), None # Helper to get features
            # NOTE: We need to run the full encoder part manually to get 'mu'
            # Or just call model forward and ignore recon. 
            # Better way:
            features = model.encoder_backbone(imgs)
            features = features.view(features.size(0), -1)
            mu = model.fc_mu(features)
            
            latents.append(mu.cpu().numpy())
            labels.append(lbls.numpy())

    X = np.concatenate(latents)
    y = np.concatenate(labels)

    # Train a quick linear classifier
    # max_iter is low to keep it fast during training
    clf = LogisticRegression(max_iter=100, solver='liblinear') 
    clf.fit(X, y)
    
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    
    acc = accuracy_score(y, preds)
    
    # Handle AUC for multiclass (One-vs-Rest)
    try:
        auc = roc_auc_score(y, probs, multi_class='ovr')
    except:
        auc = 0.5 # Fallback if only 1 class present in batch
        
    return acc, auc