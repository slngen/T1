'''
Author: CT
Date: 2023-11-06 10:15
LastEditors: CT
LastEditTime: 2023-11-11 09:31
'''
import os
import random
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import torch

from Config import config

random.seed(config.seed)

class BuildingChangeDetectionDataset(Dataset):
    def __init__(self, root, img_size=64, transform=None):
        self.root = root
        self.img_size = img_size
        self.transform = transform
        
        self.before_image = Image.open(os.path.join(root, 'before', 'before.tif'))
        self.after_image = Image.open(os.path.join(root, 'after', 'after.tif'))
        self.label_image = Image.open(os.path.join(root, 'change label', 'change_label.tif'))
        
        self.width, self.height = self.before_image.size
        self.n_patches_x = (self.width + img_size - 1) // img_size
        self.n_patches_y = (self.height + img_size - 1) // img_size
        self.n_patches = self.n_patches_x * self.n_patches_y
        
        before_image = np.array(self.before_image)
        after_image = np.array(self.after_image)
        self.combined_image = np.concatenate([before_image, after_image], axis=-1)
        
        self.combined_image = np.array(self.combined_image)
        self.label_image = np.array(self.label_image)
        
    def __len__(self):
        return self.n_patches
    
    def __getitem__(self, idx):
        x = (idx % self.n_patches_x) * self.img_size
        y = (idx // self.n_patches_x) * self.img_size
        
        centerPatch = self.extract_patch(self.combined_image, x, y)
        patch_label = self.extract_patch(self.label_image, x, y, is_label=True)
        
        directions = torch.tensor([[0, -1], [0, 1], [-1, 0], [1, 0]])
        direction = random.choice(directions)
        x_neighbor = x + direction[0] * self.img_size
        y_neighbor = y + direction[1] * self.img_size
        
        auxiliaryPatch = self.extract_patch(self.combined_image, x_neighbor, y_neighbor)
        
        if self.transform is not None:
            centerPatch = self.transform(centerPatch)
            auxiliaryPatch = self.transform(auxiliaryPatch)
        
        data = torch.stack([centerPatch, auxiliaryPatch])
        return data, patch_label, direction
    
    def extract_patch(self, image, x, y, is_label=False):
        if x < 0 or y < 0 or x + self.img_size > self.width or y + self.img_size > self.height:
            if is_label:
                patch = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            else:
                patch = np.zeros((self.img_size, self.img_size, 6), dtype=np.uint8)
        else:
            if is_label:
                patch = image[y:y+self.img_size, x:x+self.img_size]
            else:
                patch = image[y:y+self.img_size, x:x+self.img_size]
        return patch

def create_Dataset():
    root = 'W:\T1\Dataset\WHU-BCD\Raw'
    img_size = 64
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = BuildingChangeDetectionDataset(root, img_size, transform)
    return dataset

if __name__ == "__main__":
    train_dataloader = create_Dataset(batch_size=config.batch_size, mode="train")
    test_dataloader = create_Dataset(batch_size=config.batch_size, mode="test")

    for i, (data, patch_label, direction) in enumerate(train_dataloader):
        print('Train:', data.size(), patch_label.size(), direction.size())
        if i == 3:
            break

    for i, (data, patch_label, direction) in enumerate(test_dataloader):
        print('Test:', data.size(), patch_label.size(), direction.size())
        if i == 3:
            break
