import torch
import torchvision.transforms as transforms
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import os

from Backbone import Backbone
from Dataset import create_Dataset

model_paths = [r"W:\Code\T1\Baseline\Models\2023-08-23_21-55--Bx128tox64\B-E290-0.7750.ckpt", 
               r"W:\Code\T1\Baseline\Models\2023-08-29_09-47--Bx64tox64\B-E80-0.7889.ckpt", 
               r"W:\Code\T1\Baseline\Models\2023-08-24_14-30--Bx32tox64\B-E50-0.7828.ckpt"]
scales = [128, 64, 32]
eval_dataset = create_Dataset(batch_size=1, shuffle=False, speed_flag=False, mode="eval")

def reconstruct_image(network_outputs, original_size, scale):
    # 反向缩放的transform
    inverse_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((scale, scale)),
        transforms.ToTensor()
    ])
    # 初始化一个用于存储重构图像的张量
    reconstructed_image = torch.zeros((2, original_size, original_size))
    count = 0 # 用于跟踪当前处理的小块在network_outputs中的索引
    
    for y in range(0, original_size - scale + 1, scale):
        for x in range(0, original_size - scale + 1, scale):
            # 取出一个小块（当前处理的小块）
            block = network_outputs[count]
            # 反向缩放
            block_resized = inverse_transform(block.cpu().detach())
            # 将反向缩放的小块放入重构图像的正确位置
            reconstructed_image[:, y:y+scale, x:x+scale] = block_resized
            count += 1
    
    return reconstructed_image

# 定义评估函数
def evaluate_models(model_paths, dataloader, scales, output_size=(64, 64), input_size=(512, 512)):
    # 初始化模型列表
    models = [load_model_from_path(model_path) for model_path in model_paths]
    total_f1 = 0.0
    total_samples = 0
    for inputs, label_list, _ in dataloader:
        labels = label_list[0]
        if (labels.sum()<10):
            continue
        results = torch.zeros((2, input_size[0], input_size[1]))

        # 对每个尺度的图像进行推理
        for index, scale in enumerate(scales):
            scaled_images = split_and_scale_images(inputs, scale, output_size)
            model = models[index]
            # 对每个模型进行推理
            with torch.no_grad():
                model.eval()
                scores = model(scaled_images)[0]
                scores = F.softmax(scores, dim=1)
                image_scores = reconstruct_image(scores, original_size=input_size[0], scale=scale)
                results += image_scores

        results = torch.argmax(results, dim=0)
        f1 = f1_score(labels.view(-1).cpu().numpy(), results.view(-1).cpu().numpy())
        total_f1 += f1
        total_samples += 1
        print("f1: {:.3f}, avg: {:.3f}".format(f1, total_f1/total_samples))

        # 可视化
        plt.figure(figsize=(10, 5)) 
        plt.subplot(1, 2, 1)
        plt.title('Predicted')
        plt.imshow(results.cpu().numpy(), cmap='gray')
        plt.subplot(1, 2, 2)
        plt.title('Ground Truth')
        plt.imshow(labels.cpu().numpy()[0], cmap='gray')
        # plt.show()
        output_folder = "W:\Code\T1\EvalOut"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(os.path.join(output_folder, str(total_samples)+'.png'))
        plt.close()

def load_model_from_path(model_path):
    model = Backbone()
    model_state = torch.load(model_path)
    model.load_state_dict(model_state)
    return model

import torch
import torch.nn.functional as F

def split_and_scale_images(inputs, scale, output_size):
    # 将图像分割为小块并缩放到指定大小
    scaled_images = []
    
    for image in inputs:
        for y in range(0, image.size(1) - scale + 1, scale):
            for x in range(0, image.size(2) - scale + 1, scale):
                region = image[:, y:y+scale, x:x+scale]

                # 使用PyTorch的F.interpolate来缩放图像
                scaled_region = F.interpolate(region.unsqueeze(0), size=output_size, mode='bilinear', align_corners=False)
                # 移除添加的维度
                scaled_region = scaled_region.squeeze(0)
                scaled_images.append(scaled_region)
    
    return torch.stack(scaled_images)

if __name__ == "__main__":
    evaluate_models(model_paths, eval_dataset, scales)
