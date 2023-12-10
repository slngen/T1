'''
Author: CT
Date: 2023-12-10 12:26
LastEditors: CT
LastEditTime: 2023-12-10 15:02
'''
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import random
import torch
import os
import matplotlib.pyplot as plt

from Config import config
from Backbone import Backbone
from Dataset import create_Dataset

random.seed(config.seed)
torch.manual_seed(config.seed)

# Path
ckpt_path = r"Models\Unet\2023-11-30_22-36--Dice--unet--x16--r33--s64--posnone\unet-E470-0.8926.ckpt"
inference_path = os.path.dirname(ckpt_path).replace("Models", "Inferences")
os.makedirs(inference_path, exist_ok=True)

# Model
model = Backbone()
net_state = torch.load(ckpt_path)
model.load_state_dict(net_state)
model.eval()
model.to(config.device)

# Dataset
dataset = create_Dataset()
inference_dataset = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

whole_image_mask = np.zeros((dataset.height, dataset.width))

# Inference
patch_idx = 0
with torch.no_grad():
    for images, labels, directions in tqdm(inference_dataset, desc="Processing"):
        batch_size = images.size(0)
        images = images.to(config.device)
        outputs = model(images, directions)

        output_classes = torch.argmax(outputs, dim=1)
        output_binaries = (output_classes == 1).float()
        original_images = images[:, 0, :, :, :].cpu().numpy()

        for i in range(images.size(0)):
            patch_x = (patch_idx % dataset.n_patches_x) * config.image_size
            patch_y = (patch_idx // dataset.n_patches_x) * config.image_size
            patch_idx += 1

            output_image_np = output_binaries[i].squeeze().cpu().numpy() * 255
            if np.all(output_image_np == 0):
                continue
        
            valid_patch_x = min(patch_x + config.image_size, dataset.width)
            valid_patch_y = min(patch_y + config.image_size, dataset.height)
            whole_image_mask[patch_y:valid_patch_y, patch_x:valid_patch_x] = output_image_np[:valid_patch_y-patch_y, :valid_patch_x-patch_x]

whole_image_path = os.path.join(inference_path, "whole.png")
plt.imsave(whole_image_path, whole_image_mask, cmap='gray')