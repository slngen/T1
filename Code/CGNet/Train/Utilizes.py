'''
Author: CT
Date: 2023-08-15 10:10
LastEditors: CT
LastEditTime: 2023-11-06 10:34
'''
from Config import config

import numpy as np
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