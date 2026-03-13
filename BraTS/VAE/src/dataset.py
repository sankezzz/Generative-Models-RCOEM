import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HealthyBraTSDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L') # Ensure Grayscale
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0 

def get_dataloaders(train_dir, val_dir, batch_size=32, img_size=224):
    # Train: Heavy augmentation to prevent memorization
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)), 
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])

    # Validation: NO augmentation, only deterministic resizing
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_dataset = HealthyBraTSDataset(root_dir=train_dir, transform=train_transform)
    val_dataset = HealthyBraTSDataset(root_dir=val_dir, transform=val_transform)
    
    print(f"✅ Datasets Loaded: {len(train_dataset)} Train | {len(val_dataset)} Val images.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True, drop_last=False)
    
    return train_loader, val_loader