'''
Author: CT
Date: 2023-08-06 20:34
LastEditors: CT
LastEditTime: 2023-08-16 08:58
'''
import os
import numpy as np
from PIL import Image

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

from Config import config
from Backbone import Backbone

ckpt_path = r"W:\Code\T1\Baseline\Models\2023-08-15_15-49--Dx64\D-E200-0.8208.ckpt"
patch_flag = False
if patch_flag:
    dataset_path = r"W:\Data\T1\CD\WHU-BCD-Patch\Eval"
else:
    dataset_path = r"W:\Data\T1\CD\WHU-BCD\Eval"

class Datasets(Dataset):
    def __init__(self, dataset_path):
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

def create_Dataset(dataset_path, batch_size=config.batch_size, shuffle=True):
    datasets = Datasets(dataset_path)
    datasets = DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_parallel_workers)
    return datasets


# 定义滑动窗口分割函数
def sliding_window(image, stride=64, window_size=(64,64)):
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
    pred_img = prediction
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
    # Initialize global counters for TP, FP, FN
    total_TP = 0
    total_FP = 0
    total_FN = 0
    confusion_matrix = np.array([[0, 0], [0, 0]])  # [TN, FP], [FN, TP]     
    for image, label in tqdm(eval_dataset):     
        # 分割图像
        if patch_flag:
            windows = [image]
        else:
            image = F.interpolate(image, size=(64, 64), mode='bilinear', align_corners=False)
            windows = sliding_window(image)
        # 使用模型预测
        predictions = []
        with torch.no_grad():
            for window in windows:
                output = net(window)[0]
                output = F.softmax(output, dim=1)
                predictions.append(output)
                
        # 创建一个空的预测图
        # if patch_flag:
        #     prediction_map = predictions[0]
        # else:
        #     prediction_map = torch.zeros((1,2,512,512), dtype=torch.float32)
        #     # 将预测结果填入预测图
        #     idx = 0
        #     for x in range(0, image.shape[2] - 64 + 32, 32):
        #         for y in range(0, image.shape[3] - 64 + 32, 32):
        #             prediction_map[:,:,x:x+64, y:y+64] += predictions[idx]
        #             idx += 1
        prediction_map = predictions[0]

        # 根据预测图得到最终预测结果
        if not patch_flag:
            prediction_map = F.interpolate(prediction_map, size=(512, 512), mode='bilinear', align_corners=False)
        final_prediction = torch.argmax(prediction_map, dim=1)
        
        # 显示预测结果和真实标签
        true_label = label.squeeze().cpu().numpy().flatten().astype(np.int64)
        pred_label = final_prediction.squeeze().cpu().numpy().flatten()
        # show_images(final_prediction.cpu(), label.squeeze().cpu())
        TP = np.sum((true_label == 1) & (pred_label == 1))
        FP = np.sum((true_label == 0) & (pred_label == 1))
        FN = np.sum((true_label == 1) & (pred_label == 0))
        TN = np.sum((true_label == 0) & (pred_label == 0))

        confusion_matrix[0][1] += FP
        confusion_matrix[1][0] += FN
        confusion_matrix[1][1] += TP
        confusion_matrix[0][0] += TN

        correctly_predicted = np.sum(true_label == pred_label)
        accuracy = correctly_predicted / len(true_label)

        # Calculate current accumulated F1 score using the updated confusion matrix
        precision = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1]) if (confusion_matrix[1][1] + confusion_matrix[0][1]) != 0 else 0
        recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[1][0]) if (confusion_matrix[1][1] + confusion_matrix[1][0]) != 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        # 输出当前图片的F1分数
        print(f"F1 -> {f1:.4f}, ACC -> {accuracy:.4f}")

    # Print final confusion matrix
    print(confusion_matrix)