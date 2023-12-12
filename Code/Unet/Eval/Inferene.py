from torch.utils.data import DataLoader, random_split
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
inference_path = os.path.join(os.path.dirname(ckpt_path).replace("Models", "Inferences"), "split")
os.makedirs(inference_path, exist_ok=True)

# Model
model = Backbone()
net_state = torch.load(ckpt_path)
model.load_state_dict(net_state)
model.eval()
model.to(config.device)

# Dataset
dataset = create_Dataset()
train_ratio = 0.7
train_size = int(len(dataset) * train_ratio)
_, inference_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
inference_dataset = DataLoader(inference_dataset, batch_size=config.batch_size, shuffle=False)

# Inference
with torch.no_grad():
    for idx, (images, labels, directions) in enumerate(tqdm(inference_dataset, desc="Processing")):
        images = images.to(config.device)
        outputs = model(images, directions)

        output_classes = torch.argmax(outputs, dim=1)
        output_binaries = (output_classes == 1).float()
        original_images = images[:, 0, :, :, :].cpu().numpy()

        for i in range(images.size(0)):
            label_image_np = labels[i].squeeze().cpu().numpy() * 255
            output_image_np = output_binaries[i].squeeze().cpu().numpy() * 255

            if np.all(label_image_np == 0) and np.all(output_image_np == 0):
                continue

            stage_images = [original_images[i, j*3:(j+1)*3, :, :].transpose(1, 2, 0) for j in range(2)]

            fig, axs = plt.subplots(2, 2, figsize=(12, 8))
            axs[0, 0].imshow(stage_images[0])
            axs[0, 0].set_title("Before")
            axs[0, 0].axis('off')

            axs[0, 1].imshow(stage_images[1])
            axs[0, 1].set_title("After")
            axs[0, 1].axis('off')

            axs[1, 0].imshow(label_image_np, cmap='gray')
            axs[1, 0].set_title("Label")
            axs[1, 0].axis('off')

            axs[1, 1].imshow(output_image_np, cmap='gray')
            axs[1, 1].set_title("Prediction")
            axs[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(inference_path, f"{idx}_{i}.png"))
            plt.close()