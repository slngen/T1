'''
Author: CT
Date: 2023-08-14 17:51
LastEditors: CT
LastEditTime: 2023-08-14 17:52
'''
'''
Author: CT
Date: 2023-08-06 20:34
LastEditors: CT
LastEditTime: 2023-08-14 17:24
'''
import os
import cv2
import numpy as np
from PIL import Image
from skimage import transform
from sklearn.metrics import f1_score 

import random
random.seed(3)

import torch
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.transforms as py_vision

from Utilizes import task_info
from Config import config
from Backbone import Backbone

ckpt_path = r"W:\Code\T1\Baseline\Models\2023-07-30_10-14--Cx64\C-E290-0.9447.ckpt"
dataset_path = r"W:\Data\T1\CD\WHU-BCD\Eval"

class Datasets(Dataset):
    def __init__(self, dataset_path ):
        self.input_channels = config.input_dim
        self.data_List = []
        self.toTensor = py_vision.ToTensor()
        '''
        WHU-BCD
        '''
        self.files = os.listdir(os.path.join(dataset_path, "label"))

        self.data_List = []
        for file in self.files:
            self.data_List.append(
                            {
                                "path": {
                                            "A": os.path.join(dataset_path, "A", file), 
                                            "B": os.path.join(dataset_path, "B", file), 
                                        },
                                "label": os.path.join(dataset_path, "label", file), 
                            }
                        )
            
    def __len__(self):
        return len(self.data_List)

    def __getitem__(self, idx):
        data = self.data_List[idx]
        # Image
        image_path_A = data["path"]["A"]
        image_path_B = data["path"]["B"]
        image_A = Image.open(image_path_A).convert('RGB')
        image_B = Image.open(image_path_B).convert('RGB')
        image_A = self.toTensor(image_A)
        image_B = self.toTensor(image_B)
        image = np.concatenate([image_A, image_B], axis=0)
        image = torch.Tensor(image)
        # Label
        label = Image.open(data["label"]).convert('RGB')
        label_tensor = self.toTensor(label)
        # 检测白色像素
        white_mask = (label_tensor[0] > 0.9) & (label_tensor[1] > 0.9) & (label_tensor[2] > 0.9)
        # 初始化一个全0的tensor
        binary_label = torch.zeros_like(label_tensor[0])
        # 将检测到的白色像素设置为1
        binary_label[white_mask ] = 1.0
        return image, binary_label

def create_Dataset(dataset_path, batch_size=1, shuffle=False):
    datasets = Datasets(dataset_path)
    datasets = DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_parallel_workers)
    return datasets


# 定义滑动窗口分割函数
def sliding_window(image, stride=32, window_size=(64,64)):
    """
    返回一个包含所有子窗口的列表
    """
    windows = []
    for x in range(0, image.shape[2] - window_size[0] + stride, stride):
        for y in range(0, image.shape[3] - window_size[1] + stride, stride):
            windows.append(image[:,:,x:x+window_size[0], y:y+window_size[1]])
    return windows

def show_images(prediction, ground_truth):
    # 转换预测为白色和黑色图像
    pred_img = torch.zeros(3, prediction.shape[1], prediction.shape[2])
    pred_img = prediction  # 红色通道
    pred_img = (pred_img * 255).byte()  # 转换到0-255范围并转为byte类型

    # 转换真实标签为白色和黑色图像
    gt_img = torch.zeros(3, ground_truth.shape[0], ground_truth.shape[1])
    gt_img[:, ground_truth == 1] = 1
    gt_img = (gt_img * 255).byte()


    # 使用matplotlib展示
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.imshow(pred_img.permute(1, 2, 0).numpy())
    ax1.set_title('Prediction')
    ax2.imshow(gt_img.permute(1, 2, 0).numpy())
    ax2.set_title('Ground Truth')

    plt.show()

if __name__=='__main__':
    '''
    Dataset
    '''
    eval_dataset = create_Dataset(dataset_path)
    '''
    Network
    '''
    net = Backbone()
    net_state = torch.load(ckpt_path)
    net.load_state_dict(net_state)
    net.eval()
    index = 0
    '''
    Eval
    '''
    f1_scores = []
    for image, label in tqdm(eval_dataset):
        # 分割图像
        windows = sliding_window(image)
        # 使用模型预测
        predictions = []
        with torch.no_grad():
            for window in windows:
                output = net(window)[0]
                output = F.softmax(output, dim=1)
                predictions.append(output)
        # 创建一个空的预测图
        prediction_map = torch.zeros((1,2,512,512), dtype=torch.float32)
        # 将预测结果填入预测图
        idx = 0
        for x in range(0, image.shape[2] - 64 + 32, 32):
            for y in range(0, image.shape[3] - 64 + 32, 32):
                prediction_map[:,:,x:x+64, y:y+64] += predictions[idx]
                idx += 1
        # 根据预测图得到最终预测结果
        final_prediction = torch.argmax(prediction_map, dim=1)
        # 显示预测结果和真实标签
        # show_images(final_prediction.cpu(), label.squeeze().cpu())
        # 计算F1分数
        true_label = label.squeeze().cpu().numpy().flatten()
        pred_label = final_prediction.squeeze().cpu().numpy().flatten()
        f1 = f1_score(true_label, pred_label)
        if (f1>0.1):
            f1_scores.append(f1)
        # 输出当前图片的F1分数和目前的平均F1分数
        print(f"Current Image F1 Score: {f1:.4f}")
        print(f"Average F1 Score So Far: {np.mean(f1_scores):.4f}")
    # 最后输出所有图片的平均F1分数
    print(f"Final Average F1 Score: {np.mean(f1_scores):.4f}")