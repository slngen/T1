'''
Author: CT
Date: 2023-08-15 10:10
LastEditors: CT
LastEditTime: 2023-09-06 14:09
'''
'''
Author: CT
Date: 2023-04-03 20:04
LastEditors: CT
LastEditTime: 2023-08-15 09:51
'''
from Config import config

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class Loss_net(nn.Module):
    def __init__(self, mode="dice"):
        super(Loss_net, self).__init__()
        self.mode = mode
        self.loss_ce = nn.CrossEntropyLoss()
    
    def dice_loss(self, pred, target):
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def iou_loss(self, pred, target):
        smooth = 1e-5
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou
    
    def forward(self, output, label):
        loss = 0
        if self.mode == "ce":
            loss += self.loss_ce(output, label)
        elif self.mode == "dice":
            pred = F.softmax(output, dim=1)[:, 1, :, :]
            loss += self.dice_loss(pred, label == 1)
        elif self.mode == "iou":
            pred = F.softmax(output, dim=1)[:, 1, :, :]
            loss += self.iou_loss(pred, label == 1)
        return loss

class Metrics_net():
    def __init__(self):
        super().__init__()
        self.n_class = config.class_nums
        self.label_graph_mode = config.label_graph_mode
        self.CM = np.zeros((self.n_class, self.n_class))

    def clear(self):
        self.CM = np.zeros((self.n_class, self.n_class))

    def get(self):
        return self.CM

    def update(self, output, label):
        true_label = label.cpu().numpy().flatten()
        Output = F.softmax(output, dim=1).cpu().numpy()
        pred_label = np.argmax(Output, axis=1).flatten()
        TP = np.sum((true_label == 1) & (pred_label == 1))
        FP = np.sum((true_label == 0) & (pred_label == 1))
        FN = np.sum((true_label == 1) & (pred_label == 0))
        TN = np.sum((true_label == 0) & (pred_label == 0))

        self.CM[0][1] += FP
        self.CM[1][0] += FN
        self.CM[1][1] += TP
        self.CM[0][0] += TN

def Metrics(CM):
    result = {}
    smooth = 1e-5
    if "acc" in config.metrics_List:
        acc_List = []
        acc = 100*np.diag(CM).sum()/(CM.sum()+smooth)
        acc_List.append(acc)
        result["acc"] = acc_List
    if "F1" in config.metrics_List:
        F1_Dict = {
            "F1":[],
            "precision":[],
            "recall":[],
        }
        # macro-F1
        P = np.zeros(config.class_nums)
        R = np.zeros(config.class_nums)
        for cls_index in range(config.class_nums):
            P[cls_index] = CM[cls_index, cls_index] / (CM[:,cls_index].sum()+smooth)
            R[cls_index] = CM[cls_index, cls_index] / (CM[cls_index,:].sum()+smooth)
        P = P[-1]
        R = R[-1]
        F1 = 2*P*R/(P+R+smooth)
        F1_Dict["F1"].append(F1)
        F1_Dict["precision"].append(P)
        F1_Dict["recall"].append(R)
        result["F1"] = F1_Dict
    if "kappa" in config.metrics_List:
        kappa_List = []
        P0 = np.diag(CM).sum()/(CM.sum()+smooth)
        Pe = 0
        for cls_index in range(config.class_nums):
            Pe += CM[:,cls_index].sum()*CM[cls_index,:].sum()
        Pe /= np.power(CM.sum(), 2)
        kappa = (P0-Pe)/(1-Pe)
        kappa_List.append(kappa)
        result["kappa"] = kappa_List
    return result

class Task_Info():
    def __init__(self):
        self.task_flag_Dict = config.task_flag_Dict
        self.dataset_path_Dict = config.dataset_path_Dict
        self.task_channels_decoder = config.task_channels_decoder
        # Cal input channel nums. 
        self.input_channels = 0
        self.task_seq = []
        for task_index, task_list in self.task_flag_Dict.items():
            if len(task_list):
                # print("{}:".format(task_index))
                for task in task_list:
                    self.task_seq.append(task)
        self.task_str = "."+"-".join(self.task_seq)
        # Task index encoder/decoder
        self.task_encoder = {}
        self.task_decoder = {}
        for task_index, task in enumerate(self.task_seq):
            self.task_encoder[task] = task_index
            self.task_decoder[task_index] = task
            

    def encode_task(self, task):
        return self.task_encoder[task]
    
    def decode_task(self, task_index):
        return self.task_decoder[task_index]
    
    def get_task_list(self):
        return self.task_seq

    def get_task_str(self):
        return self.task_str

task_info = Task_Info()