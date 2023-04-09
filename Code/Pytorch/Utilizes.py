'''
Author: CT
Date: 2023-04-03 20:04
LastEditors: CT
LastEditTime: 2023-04-09 22:40
'''
from Config import config

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Loss_net(nn.Module):
    def __init__(self):
        super(Loss_net, self).__init__()
        self.loss_ce = nn.CrossEntropyLoss()

    def forward(self, PL_List, label_List):
        loss = 0
        for PLout in PL_List:
            for label_index, label in enumerate(label_List):
                loss += self.loss_ce(PLout[:,label_index*2:label_index*2+config.class_nums,:,:], label)
        return loss

class Metrics_net():
    def __init__(self):
        super().__init__()
        self.n_class = config.class_nums
        self.label_graph_mode = config.label_graph_mode
        self.CM_List = [np.zeros((self.n_class, self.n_class)) for _ in range(len(self.label_graph_mode))]

    def clear(self):
        self.CM_List = [[np.zeros((self.n_class, self.n_class)) for _ in range(6)] for _ in range(len(self.label_graph_mode))]

    def get(self):
        return self.CM_List

    def update(self, output_List, label_List):
        for PLout_index, PLout in enumerate(output_List):
            for label_index, label in enumerate(label_List):
                Output = F.softmax(PLout[:,label_index*2:label_index*2+config.class_nums,:,:], dim=1).cpu().numpy()
                Prediction = np.argmax(Output, axis=1).flatten()
                Label = label.cpu().numpy().flatten()
                cm = np.bincount(self.n_class * Label + Prediction, minlength=self.n_class*self.n_class).reshape(self.n_class, self.n_class)
                self.CM_List[label_index][PLout_index] += cm

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
        # P = P.mean()
        # R = R.mean()
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
        # print("="*10, "Task Info", "="*10)
        # # Check channel mode
        # if config.channel_mode not in ["overlap", "order"]:
        #     print("channel mode error!")
        #     raise NotImplementedError
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
                    # print("\t{},\n".format(task))
                    if config.channel_mode == "overlap":
                        self.input_channels = max(self.input_channels, self.task_channels_decoder[task])
                    elif config.channel_mode == "order":
                        self.input_channels = self.input_channels + self.task_channels_decoder[task]
        self.task_str = "."+"-".join(self.task_seq)
        # Task index encoder/decoder
        self.task_encoder = {}
        self.task_decoder = {}
        for task_index, task in enumerate(self.task_seq):
            self.task_encoder[task] = task_index
            self.task_decoder[task_index] = task
            
    def get_input_channels(self):
        return self.input_channels

    def encode_task(self, task):
        return self.task_encoder[task]
    
    def decode_task(self, task_index):
        return self.task_decoder[task_index]
    
    def get_task_list(self):
        return self.task_seq

    def get_task_str(self):
        return self.task_str

task_info = Task_Info()