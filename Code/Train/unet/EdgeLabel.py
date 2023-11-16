import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_center_decreasing_weights(mask):
    # 对mask进行距离变换，得到的是每个前景像素到最近边界的距离
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)

    # 归一化权重
    weights = cv2.normalize(dist_transform, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    weights = (1 - weights) * mask / 255.0  # 使得背景像素权重为0，前景像素权重为其到边界的距离

    return weights

def visualize_center_decreasing_weights(directory, img_size, save_path):
    file_list = [f for f in os.listdir(directory) if f.endswith('.tif') or f.endswith('.jpg')]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for f in tqdm(file_list):
        mask_path = os.path.join(directory, f)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # 确保图像大小为img_size x img_size
        mask = cv2.resize(mask, (img_size, img_size))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # 如果图像全黑，跳过
        if np.sum(mask)==0:
            continue

        weights = compute_center_decreasing_weights(mask)

        # 保存可视化图像
        output_path = os.path.join(save_path, f"{f}")
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.title("Original Mask")
        plt.imshow(mask, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title("Center Decreasing Weights")
        plt.imshow(weights, cmap='hot')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

# 调用函数
directory = r"W:\Data\T1\CD\WHU-BCD\Eval\label" # 替换为你的文件夹路径
img_size = 512  # 或其他你想要设定的大小
save_path = r"W:\Data\T1\CD\WHU-BCD\Eval\Edgelabel"  # 替换为你想要保存效果图的路径
visualize_center_decreasing_weights(directory, img_size, save_path)