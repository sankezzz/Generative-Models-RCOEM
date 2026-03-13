import torch
import numpy as np

def initialize_weights(model):
    # Initializes weights for stability
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0)

def get_d_accuracy(pred_real, pred_fake):
    """
    In WGAN, we don't use 0.5. We check if the critic correctly 
    assigns HIGHER scores to real images than to fake images.
    """
    # Calculate how many real samples got a score higher than the average fake score
    # or simply check if pred_real > pred_fake on average.
    # A simple robust metric:
    real_correct = (pred_real > 0).float().mean().item()
    fake_correct = (pred_fake < 0).float().mean().item()
    
    return (real_correct + fake_correct) / 2