'''
Author: CT
Date: 2022-12-09 10:36
LastEditors: CT
LastEditTime: 2023-11-16 15:40
'''
from easydict import EasyDict as ed

config = ed({
    "device":"cuda:1",  # CPU or cuda
    "class_nums":2,
    # Dataset
    "data_path":"/Code/T1/Datasets/WHU-BCD",
    "image_size":256,
    "num_parallel_workers":4,
    "batch_size": 16,
    "input_dim": 6,
    "seed": 33,
    # Model
    "pretrained":False,
    "resume":r"", 
    # Train & Eval
    "eval_epochs":10,
    "start_eval_epochs":0,
    "eval_traindata":True,
    "epoch_size": 501,
    "loss_monitor_step":50,
    "metrics_List":["acc", "F1"],  # "acc", "F1", "kappa"
    "save_metrics_List":["F1"],
    # Log
    "save_model_path":"/Code/T1/Models/DeepLabv3",
    "log_path":"/Code/T1/Logs/DeepLabv3",
    # LR
    "lr_init":5e-4,
    "lr_max":5e-4,
    "lr_end":5e-5,
    "warmup_epochs":0,
})
