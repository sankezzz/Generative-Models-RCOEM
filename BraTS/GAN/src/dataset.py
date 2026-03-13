import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class BraTSGANDataset(Dataset):
    def __init__(self, root_dir, img_size=224):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        
        # Transformations specific for GAN (Tanh activation needs -1 to 1 range)
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5), # Add slight augmentation to prevent Mode Collapse
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('L') # 'L' strictly enforces 1-channel Grayscale
        
        image = self.transform(image)
        return image, 0 # GANs don't need labels, return 0 as a dummy

def get_dataloaders(train_dir, batch_size=32, img_size=224):
    dataset = BraTSGANDataset(root_dir=train_dir, img_size=img_size)
    
    print(f"✅ GAN Dataset Loaded: {len(dataset)} Healthy Train images.")
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                        num_workers=4, pin_memory=True, drop_last=True)
    
    return loader