'''
Author: CT
Date: 2023-07-30 20:09
LastEditors: CT
LastEditTime: 2023-07-30 20:32
'''
import os
import re
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

log_dir = "W:\Code\T1\Baseline\Logs"

for dirpath, dirnames, filenames in os.walk(log_dir):
    for log_dir in tqdm(dirnames):
        if "--" not in log_dir:
            continue

        log_file_path = os.path.join(dirpath, log_dir, "info.log")

        # 创建一个空的DataFrame用于存储信息
        df = pd.DataFrame(columns=['Epoch', 'Dataset', 'F1', 'Precision', 'Recall'])
        data = []

        with open(log_file_path, "r") as file:
            content = file.readlines()

        dataset = None
        epoch = {'Train': -1, 'Eval': -1}
        for line in content:
            if 'train dataset' in line:
                dataset = 'Train'
                epoch[dataset] += 1
            elif 'eval dataset' in line:
                dataset = 'Eval'
                epoch[dataset] += 1

            if line.startswith('F1-->'):
                # 使用正则表达式提取F1，precision和recall的值
                f1_info = re.findall(r"'F1': \[(.*?)\]", line)
                precision_info = re.findall(r"'precision': \[(.*?)\]", line)
                recall_info = re.findall(r"'recall': \[(.*?)\]", line)

                # 若提取到信息，则加入到临时列表中
                if f1_info and precision_info and recall_info:
                    data.append({'Epoch': epoch[dataset] * 10, 
                                'Dataset': dataset, 
                                'F1': float(f1_info[0]), 
                                'Precision': float(precision_info[0]), 
                                'Recall': float(recall_info[0])})

        # 使用concat函数合并数据
        df = pd.concat([df, pd.DataFrame(data)], ignore_index=True)

        # 设置索引列为Epoch
        df.set_index('Epoch', inplace=True)

        # 保存到本地
        df.to_csv(os.path.join(dirpath, log_dir, "result_info.csv"))

        # 画图
        for metric in ['F1', 'Precision', 'Recall']:
            plt.figure(figsize=(10, 5))
            for dataset in ['Train', 'Eval']:
                subset = df[df['Dataset'] == dataset]
                plt.plot(subset[metric], label=dataset)
            plt.title(metric)
            plt.legend()
            plt.savefig(os.path.join(dirpath, log_dir, f"{metric}_plot.png"))
