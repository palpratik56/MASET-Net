import torch
from torch.utils.data import Dataset
import numpy as np

class CVCDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

class ISICDataset(CVCDataset):
    pass  # Extend dataset functionality if needed
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get all image and mask file names
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.mask_names = [f for f in os.listdir(mask_dir) if f.endswith('.png')]
        
        # Sort and take only the first 50% of images and masks
        self.image_names = sorted(self.image_names)[:len(self.image_names) // 2]
        self.mask_names = sorted(self.mask_names)[:len(self.mask_names) // 2]
        
        # Check if there are any images after filtering
        if len(self.image_names) == 0:
            raise ValueError(f"No images found in the directory: {image_dir}")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            mask = transforms.Resize((256, 256))(mask)
            mask = transforms.ToTensor()(mask)
        
        return image, mask
      
class KVASDataset(CVCDataset):
    pass  # Extend dataset functionality if needed
    def __init__(self, image_folder, mask_folder=None, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform
        # Filter files by valid image extensions and sort to align images and masks
        self.image_names = sorted([f for f in os.listdir(image_folder) if f.endswith('jpg')])
        if mask_folder:
            self.mask_names = sorted([f for f in os.listdir(mask_folder) if f.endswith('jpg')])
        else:
            self.mask_names = None

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.image_names[idx])
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.mask_folder:
            # Load corresponding mask
            mask_name = os.path.join(self.mask_folder, self.mask_names[idx])
            mask = Image.open(mask_name).convert('L')  # Convert mask to grayscale

            if self.transform:
                mask = self.transform(mask)
            return image, mask
        else:
            return image
