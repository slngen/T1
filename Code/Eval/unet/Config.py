'''
Author: CT
Date: 2022-12-09 10:36
LastEditors: CT
LastEditTime: 2023-11-16 15:41
'''
from easydict import EasyDict as ed

config = ed({
    "device":"cuda",  # CPU or cuda
    "class_nums":2,
    # Task
    "task_flag_Dict":{
        "SC":[],  #  "UCMLU"
        "OD":[],  #  "RSOD-Aircraft"
        "CD":["CDD-BCD"],  #  "WHU-BCD", "CDD-BCD"
        "SS":[],  #  "GID"
    },
    # Task channels decoder
    "task_channels_decoder":{
        "UCMLU":3,
        "RSOD-Aircraft":3,
        "GID":3,
        "WHU-BCD":6,
        "CDD":6,
    },
    # Dataset
    "image_size":64,
    "num_parallel_workers":4,
    "batch_size": 64,
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
    "save_model_path":"W:\T1\Models",
    "log_path":"W:\T1\Logs",
    # LR
    "lr_init":5e-4,
    "lr_max":5e-4,
    "lr_end":5e-5,
    "warmup_epochs":0,
    # Backbone
    "features":[32, 64, 128]
})
