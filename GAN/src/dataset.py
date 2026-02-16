import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(root_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # GANs prefer -1 to 1 range (Tanh)
    ])

    # Assumes dataset is at root_dir/classification_task/train
    train_dir = os.path.join(root_dir, 'classification_task', 'train')
    
    # We load everything as one big dataset for GAN training
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=4, pin_memory=True, drop_last=True)
    
    return loader