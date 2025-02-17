import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
import os
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, transforms=None, transforms_mask=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.transforms_mask = transforms_mask
        self.image_files = [f for f in os.listdir(os.path.join(root_dir,'images')) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root_dir,'images', self.image_files[idx]))
        mask = Image.open(os.path.join(self.root_dir,'masks', self.image_files[idx].replace('-org.jpg','-gt.pbm')))

        if np.random.rand() < 0.2:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            
        if np.random.rand() < 0.2 :
            angle = np.random.uniform(-45,45)
            image = image.rotate(angle)
            mask = mask.rotate(angle)

        if self.transforms:
            image = self.transforms(image)
            #mask = self.transforms_mask(mask)
            
            
            
            mask = torch.squeeze(self.transforms_mask(mask)).long()

            #print("masks :", mask.shape)
            #print("images :", image.shape)

        return image, mask


def get_dataloaders(root_dir, batch_size=16, valid_size=0.1, num_workers=0):
    # Définir les transformations à appliquer aux images
    transform = transforms.Compose([transforms.Resize((256,256)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    
    transform_mask = transforms.Compose([transforms.Resize((256,256)),
                                transforms.ToTensor()])
    
                                    
    dataset = SegmentationDataset(root_dir, transforms=transform, transforms_mask=transform_mask)

    # Séparer les données en ensemble d'entraînement et ensemble de validation
    if (valid_size):
        train_data, val_data = train_test_split(dataset, test_size=valid_size, random_state=42)
    else:
        train_data = dataset
        val_data = []

    # Créer les dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


