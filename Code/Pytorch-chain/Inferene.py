'''
Author: CT
Date: 2023-07-23 13:53
LastEditors: CT
LastEditTime: 2023-08-06 20:21
'''
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

from Config import config
from Backbone import Backbone
from Dataset import create_Dataset

ckpt_path = r"/Code/T1/Models/2023-10-08_06-27--DiceL2-32x64tox64/L2-32-E290-0.9174.ckpt"

def plot_images(original_A, original_B, predicted, label, index):
    # 输入检查
    assert original_A.shape == (3, 64, 64)
    assert original_B.shape == (3, 64, 64)
    assert predicted.shape == (64, 64)
    assert label.shape == (64, 64)

    if not torch.any(label):
        return

    # 将原图张量转换为numpy数组，并转置通道维度
    original_A = original_A.numpy().transpose((1, 2, 0))
    original_B = original_B.numpy().transpose((1, 2, 0))
    # 将预测和标签图转换为numpy数组
    predicted = predicted.numpy()
    label = label.numpy()

    # 创建预测和标签图的RGB表示
    predicted_rgb = np.zeros((64, 64, 3))  # 创建一个全0的数组
    predicted_rgb[predicted == 1] = [1, 0, 0]  # 将1值设为红色

    label_rgb = np.zeros((64, 64, 3))  # 创建一个全0的数组
    label_rgb[label == 1] = [1, 0, 0]  # 将1值设为红色

    # 创建子图
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))

    # 绘制原图A和B
    axs[0, 0].imshow(original_A)
    axs[0, 0].set_title('Original A')
    axs[0, 1].imshow(original_B)
    axs[0, 1].set_title('Original B')

    # 绘制预测和标签图
    axs[1, 0].imshow(predicted_rgb)
    axs[1, 0].set_title('Predicted')
    axs[1, 1].imshow(label_rgb)
    axs[1, 1].set_title('Label')

    # 移除坐标轴
    for ax in axs.flat:
        ax.axis('off')

    # plt.show()
    plt.savefig("/Code/T1/Inference/"+str(index)+".png")
    plt.close(fig)

if __name__=='__main__':
    '''
    Dataset
    '''
    # train_dataset = create_Dataset(batch_size=1, shuffle=True, speed_flag=False, mode="train")
    eval_dataset = create_Dataset(batch_size=1, shuffle=False, speed_flag=False, mode="eval")
    '''
    Network
    '''
    net = Backbone()
    net_state = torch.load(ckpt_path)
    net.load_state_dict(net_state)
    net.eval()
    index = 0
    for image, label_List, task_flag in tqdm(eval_dataset):
        # forward
        output_List = net(image)
        # softmax
        for output_index in range(len(output_List)):
            output_List[output_index] = F.softmax(output_List[output_index], dim=1)
        index += 1
        plot_images(
            image[0,0:3,:,:],
            image[0,3:6,:,:],
            torch.argmax(output_List[0][0],dim=0),
            label_List[0][0],
            index
        )




