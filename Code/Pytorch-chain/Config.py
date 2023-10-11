'''
Author: CT
Date: 2022-12-09 10:36
LastEditors: CT
LastEditTime: 2023-09-13 11:05
'''
from easydict import EasyDict as ed

config = ed({
    "device":"cuda:1",  # CPU or cuda
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
    "scale_factor": 2,
    "loss_weight":1,
    # Vote
    "deep_supervision": True,
    # Dataset
    "label_graph_mode":["full"],  # "full", "edge"
    "image_size":64,
    "raw_size":64,
    "dataset_path_Dict": {
                "RSOD-Aircraft": "/root/CT/ModelArts/T1/Datasets/RSOD-OD/aircraft",
                "UCMLU": "/root/CT/ModelArts/T1/Datasets/UCMerced_LandUse",
                "WHU-BCD": "/Code/T1/Dataset/WHU-BCD/split_64",
                "CDD-BCD": "/Code/T1/Dataset/CDD-BCD/split_64",
        },
    "num_parallel_workers":8,
    "batch_size": 32,
    "speed_up_nums":128,
    # Model
    "pretrained":False,
    "resume":r"", 
    # Train & Eval
    "eval_epochs":3,
    "start_eval_epochs":0,
    "eval_traindata":True,
    "epoch_size": 201,
    "loss_monitor_step":50,
    "metrics_List":["acc", "F1"],  # "acc", "F1", "kappa"
    "save_metrics_List":["F1"],
    # Log
    "save_model_path":"/Code/T1/Models",
    "log_path":"/Code/T1/Logs",
    # LR
    "lr_init":5e-4,
    "lr_max":5e-4,
    "lr_end":5e-5,
    "warmup_epochs":0,
    # Backbone
    "backbone_type": "L3-8",
    "input_dim":6,
    "layer_nums":3,
    "backbone_dims":[8,16,32,64]
})
