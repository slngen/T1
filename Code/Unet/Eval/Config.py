'''
Author: CT
Date: 2022-12-09 10:36
LastEditors: CT
LastEditTime: 2023-12-10 15:03
'''
from easydict import EasyDict as ed

config = ed({
    "device":"cuda",  # CPU or cuda
    "class_nums":2,
    # Dataset
    "data_path":"Datasets/WHU-BCD",
    "image_size":64,
    "num_parallel_workers":4,
    "batch_size": 64,
    "input_dim": 6,
    "seed": 33,
    # Model
    "pretrained":False,
    "resume":r"", 
    "pos_mode": "none", # add ,none or multi
    # Train & Eval
    "eval_epochs":10,
    "start_eval_epochs":0,
    "eval_traindata":True,
    "epoch_size": 501,
    "loss_monitor_step":50,
    "metrics_List":["acc", "F1"],  # "acc", "F1", "kappa"
    "save_metrics_List":["F1"],
    # Log
    "save_model_path":"Models",
    "log_path":"Logs",
    # LR
    "lr_init":5e-4,
    "lr_max":5e-4,
    "lr_end":5e-5,
    "warmup_epochs":0,
    # Backbone
    "features":[16, 32, 64]
})
