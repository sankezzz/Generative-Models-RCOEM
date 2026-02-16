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
    # How many Reals did it call Real? (>0.5)
    acc_real = (pred_real > 0.5).float().mean().item()
    # How many Fakes did it call Fake? (<0.5)
    acc_fake = (pred_fake < 0.5).float().mean().item()
    return (acc_real + acc_fake) / 2