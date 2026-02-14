import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(root_dir, batch_size=32, img_size=224):
    """
    Expects root_dir to be the path to 'brisc2025'.
    It will look for 'classification_task/train' and 'classification_task/test'.
    """
    
    # Define transformations: Resize, Grayscale, ToTensor
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(), 
        # No normalization needed for simple VAE, 0-1 range is best for Sigmoid output
    ])

    train_dir = os.path.join(root_dir, 'classification_task', 'train')
    test_dir = os.path.join(root_dir, 'classification_task', 'test')

    # ImageFolder assumes structure: root/class_name/image.jpg
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader