'''
Author: CT
Date: 2022-12-09 10:36
LastEditors: CT
LastEditTime: 2023-09-04 09:43
'''
from easydict import EasyDict as ed

config = ed({
    "device":"cuda",  # CPU or cuda
    "class_nums":2,
    # Task
    "task_flag_Dict":{
        "SC":[],  #  "UCMLU"
        "OD":[],  #  "RSOD-Aircraft"
        "CD":["WHU-BCD"],  #  "WHU-BCD", "CDD"
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
    "image_size":512,
    "raw_size":512,
    "dataset_path_Dict": {
                "RSOD-Aircraft": "/root/CT/ModelArts/T1/Datasets/RSOD-OD/aircraft",
                "UCMLU": "/root/CT/ModelArts/T1/Datasets/UCMerced_LandUse",
                # "WHU-BCD": "W:/Data/T1/CD/WHU-BCD-Patch-32",
                # "WHU-BCD": "W:/Data/T1/CD/WHU-BCD-Patch-128",
                "WHU-BCD": "W:/Data/T1/CD/WHU-BCD",
                # "WHU-BCD": "W:/Data/T1/CD/WHU-BCD-Patch",
                "CDD": "W:/Data/T1/CD/CDD",
        },
    "num_parallel_workers":8,
    "batch_size": 256,
    "speed_up_nums":128,
    # Model
    "pretrained":False,
    "resume":r"", 
    # Train & Eval
    "eval_epochs":10,
    "start_eval_epochs":0,
    "eval_traindata":True,
    "epoch_size": 301,
    "loss_monitor_step":20,
    "metrics_List":["acc", "F1"],  # "acc", "F1", "kappa"
    "save_metrics_List":["F1"],
    # Log
    "save_model_path":r"W:\Code\T1\Baseline\Models",
    "log_path":r"W:\Code\T1\Baseline\Logs",
    # LR
    "lr_init":1e-4,
    "lr_max":1e-4,
    "lr_end":1e-5,
    "warmup_epochs":0,
    # Backbone
    "backbone_type": "B",
    "input_dim":6,
    "backbone_dims":[64, 128, 256]
})
