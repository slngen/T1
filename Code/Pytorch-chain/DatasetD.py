'''
Author: CT
Date: 2022-12-09 10:36
LastEditors: CT
LastEditTime: 2023-08-29 09:34
'''
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
random.seed(3)

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from Config import config
from Utilizes import task_info

def compute_center_decreasing_weights(mask):
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    weights = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    weights = 1 - weights
    weights *= mask / 255.0
    return weights



class Datasets(Dataset):
    def __init__(self, speed_flag, mode, train_rate):
        self.img_size = config.image_size
        self.input_channels = config.input_dim
        self.dataset_path_Dict = task_info.dataset_path_Dict
        self.task_flag_List = task_info.get_task_list()
        self.data_List = []
        self.label_graph_mode = config.label_graph_mode
        end_channel = 0
        
        # torchvision transforms
        if mode == "train":
            self.transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor()
            ])

        if "WHU-BCD" in self.task_flag_List:
            begin_channel = end_channel
            end_channel += config.task_channels_decoder["WHU-BCD"]
            WHU_BCD_data_path = self.dataset_path_Dict["WHU-BCD"]
            WHU_BCD_data_path = os.path.join(WHU_BCD_data_path, "Train" if mode == "train" else "Eval")
            self.files = os.listdir(os.path.join(WHU_BCD_data_path, "Label"))

            for file in self.files:
                self.data_List.append({
                    "path": {
                        "A": os.path.join(WHU_BCD_data_path, "A", file), 
                        "B": os.path.join(WHU_BCD_data_path, "B", file)
                    },
                    "label": os.path.join(WHU_BCD_data_path, "Label", file), 
                    "task_flag": task_info.encode_task("WHU-BCD"),
                    "end_channel": end_channel,
                    "begin_channel": begin_channel
                })
            
    def __len__(self):
        return len(self.data_List)

    def __getitem__(self, idx):
        data = self.data_List[idx]

        # Load images using torchvision
        image_A = self.transforms(Image.open(data["path"]["A"]).convert('RGB'))
        image_B = self.transforms(Image.open(data["path"]["B"]).convert('RGB'))
        image = torch.cat([image_A, image_B], dim=0)

        label = self.transforms(Image.open(data["label"]).convert('RGB'))
        white_mask = (label[0] > 0.9) & (label[1] > 0.9) & (label[2] > 0.9)
        binary_label = white_mask.float()

        label_List = []

        if "full" in self.label_graph_mode:
            label_List.append(binary_label.squeeze().long())

        # Edge label
        if "edge" in self.label_graph_mode:
            white_mask_np = (white_mask.numpy() * 255).astype(np.uint8)
            edge_label = compute_center_decreasing_weights(white_mask_np)
            label_List.append(torch.Tensor(edge_label))
            # # 可视化weights和原始label
            # if (label.max()==1):
            #     plt.figure(figsize=(10, 5))
                
            #     plt.subplot(1, 2, 1)
            #     plt.imshow(white_mask_np, cmap='gray')
            #     plt.title('Original Label')
                
            #     plt.subplot(1, 2, 2)
            #     plt.imshow(edge_label, cmap='hot')
            #     plt.title('Weights')
                
            #     plt.show()

        return image, label_List, data["task_flag"]


def create_Dataset(batch_size, shuffle=True, speed_flag=False, mode="train", train_rate=0.7):
    datasets = Datasets(speed_flag, mode, train_rate)
    datasets = DataLoader(datasets, 
                          batch_size=batch_size, 
                          shuffle=shuffle, 
                          num_workers=config.num_parallel_workers,
                          sampler=DistributedSampler(datasets))
    return datasets

if __name__ == "__main__":
    datasets = create_Dataset(batch_size=8, shuffle=True)
    for data in datasets:
        pass