import os
import random
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import torch

from Config import config

class BuildingChangeDetectionDataset(Dataset):
    def __init__(self, root, img_size=64, transform=None):
        self.root = root
        self.img_size = img_size
        self.transform = transform
        
        # 加载图像
        self.before_image = Image.open(os.path.join(root, 'before', 'before.tif'))
        self.after_image = Image.open(os.path.join(root, 'after', 'after.tif'))
        self.label_image = Image.open(os.path.join(root, 'change label', 'change_label.tif'))
        
        # 计算有多少个patch
        self.width, self.height = self.before_image.size
        self.n_patches_x = (self.width + img_size - 1) // img_size
        self.n_patches_y = (self.height + img_size - 1) // img_size
        self.n_patches = self.n_patches_x * self.n_patches_y
        
        # 将before和after图像拼接
        before_image = np.array(self.before_image)
        after_image = np.array(self.after_image)
        self.combined_image = np.concatenate([before_image, after_image], axis=-1)
        
        # 将图像转为numpy array
        self.combined_image = np.array(self.combined_image)
        self.label_image = np.array(self.label_image)
        
    def __len__(self):
        return self.n_patches
    
    def __getitem__(self, idx):
        # 计算patch的坐标
        x = (idx % self.n_patches_x) * self.img_size
        y = (idx // self.n_patches_x) * self.img_size
        
        # 提取patch
        patch_image = self.extract_patch(self.combined_image, x, y)
        patch_label = self.extract_patch(self.label_image, x, y, is_label=True)
        
        # 随机选择一个相邻的patch
        direction = random.choice([0, 1, 2, 3])  # 0: up, 1: down, 2: left, 3: right
        if direction == 0:
            x_neighbor = x
            y_neighbor = y - self.img_size
        elif direction == 1:
            x_neighbor = x
            y_neighbor = y + self.img_size
        elif direction == 2:
            x_neighbor = x - self.img_size
            y_neighbor = y
        else:  # direction == 3
            x_neighbor = x + self.img_size
            y_neighbor = y
        
        neighbor_image = self.extract_patch(self.combined_image, x_neighbor, y_neighbor)
        
        if self.transform is not None:
            patch_image = self.transform(patch_image)
            neighbor_image = self.transform(neighbor_image)
        
        data = torch.stack([patch_image, neighbor_image])
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

def create_Dataset(batch_size, mode="train", shuffle=True):
    root = '/Code/T1/Dataset/WHU-BCD/Raw'  # 请将此路径改为你的数据根目录
    img_size = 64
    train_ratio = 0.7
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = BuildingChangeDetectionDataset(root, img_size, transform)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    if mode == "train":
        dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    elif mode == "test":
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        raise ValueError("Mode must be 'train' or 'test'")
    
    return dataloader

if __name__ == "__main__":
    train_dataloader = create_Dataset(batch_size=config.batch_size, mode="train")
    test_dataloader = create_Dataset(batch_size=config.batch_size, mode="test")

    for i, (data, patch_label, direction) in enumerate(train_dataloader):
        print('Train:', data.size(), patch_label.size(), direction)
        if i == 3:  # 读取四个batch后停止
            break

    for i, (data, patch_label, direction) in enumerate(test_dataloader):
        print('Test:', data.size(), patch_label.size(), direction)
        if i == 3:  # 读取四个batch后停止
            break