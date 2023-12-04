'''
Author: CT
Date: 2023-11-16 14:23
LastEditors: CT
LastEditTime: 2023-12-01 23:16
'''
import os
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from Config import config
from Backbone import Backbone
from Dataset import create_Dataset
from Utilizes import Metrics, Metrics_net

random.seed(config.seed)
torch.manual_seed(config.seed)
'''
Path
'''
ckpt_path = r"/Code/T1/Models/Unet/2023-12-02_09-57--Dice--unet--x64--r33--s64--posnone/unet-E480-0.9068.ckpt"
log_path = os.path.join(os.path.dirname(ckpt_path).replace("Models","Logs"), "eval.log")

'''
Model
'''
model = Backbone()
net_state = torch.load(ckpt_path)
model.load_state_dict(net_state)
model.eval()
model.to(config.device)
metricNet = Metrics_net()

'''
Dataset
'''
dataset = create_Dataset()
train_ratio = 0.7
train_size = int(len(dataset) * train_ratio)
train_dataset, eval_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
# train_dataset = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
eval_dataset = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
print("###Dataset eval:",len(eval_dataset.dataset))

'''
Eval
'''
eval_log_file = open(log_path, "w")
with torch.no_grad():
    metricNet.clear()
    for image, label, directions in tqdm(eval_dataset):
        image = image.to(config.device)
        label = label.to(config.device)
        output = model(image, directions)
        metricNet.update(output, label)

    CM = metricNet.get()
    result = Metrics(CM)
    for key, value in result.items():
        print(key, "--> ", value)
        eval_log_file.write(key+"--> "+str(value)+"\n")
        eval_log_file.flush()