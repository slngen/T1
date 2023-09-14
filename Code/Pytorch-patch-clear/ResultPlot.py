'''
Author: CT
Date: 2023-07-30 20:39
LastEditors: CT
LastEditTime: 2023-07-30 20:46
'''
import os
import pandas as pd
import matplotlib.pyplot as plt

# 定义线条样式
line_styles = {'64': '-', '512': '--'}

# 定义线条颜色
colors = {'A': 'red', 'B': 'blue', 'C': 'green'}

# 定义数据的根目录
root_dir = r'W:\Code\T1\Baseline\Logs'  # 请将此处改为你的实际目录路径

# 创建一个空的DataFrame用于存储信息
dataframes = []

# 遍历根目录下的每个文件夹，读取CSV文件，并添加到dataframes列表中
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if os.path.isdir(folder_path):
        df = pd.read_csv(os.path.join(folder_path, "result_info.csv"))
        df['Scheme'] = folder_name.split('--')[1].split('x')[0]  # Add experiment scheme information
        df['Size'] = folder_name.split('x')[1]  # Add image size information
        dataframes.append(df)

# 使用concat函数合并所有的DataFrame
all_data = pd.concat(dataframes, ignore_index=True)

# 对于每一个数据集（'Train'和'Eval'）和指标（'F1', 'Precision', 'Recall'），创建一个图表
for dataset in ['Train', 'Eval']:
    for metric in ['F1', 'Precision', 'Recall']:
        plt.figure(figsize=(10, 5))
        # 遍历每个实验方案和图像大小
        for scheme, color in colors.items():
            for size, linestyle in line_styles.items():
                # 选取该实验方案和图像大小的数据
                subset = all_data[(all_data['Dataset'] == dataset) & 
                                  (all_data['Scheme'] == scheme) & 
                                  (all_data['Size'] == size)]
                # 绘制线条，使用不同的颜色和线条样式
                if not subset.empty:  # Check if subset is empty
                    plt.plot(subset['Epoch'], subset[metric], label=f"{scheme}x{size}", color=color, linestyle=linestyle)
                # 设置X轴和Y轴标签
                plt.xlabel('Epochs')
                plt.ylabel(metric + ' score')
        # 设置图表的标题和图例
        if dataset == "Eval":
            plt.title(f"{metric}")
        else:
            plt.title(f"{dataset} {metric}")
        plt.legend()
        # 保存图表为图片文件
        plt.savefig(os.path.join(r"W:\Code\T1\Baseline\Logs", f"{dataset}_{metric}_plot.png"))
