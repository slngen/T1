'''
Author: CT
Date: 2022-12-09 10:36
LastEditors: CT
LastEditTime: 2023-04-09 23:17
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
    "channel_mode":"order",  # overlap or order
    "alpha":0.2, 
    # Vote
    "deep_supervision": True,
    "SG_level": "pixel", # "image" or "pixel"
    "SG_dims": 64,
    "decoder_dims": 8,
    # Dataset
    "label_graph_mode":["full"],  # "full", "edge"
    "image_size":512,
    "dataset_path_Dict": {
                "RSOD-Aircraft": "/root/CT/ModelArts/T1/Datasets/RSOD-OD/aircraft",
                "UCMLU": "/root/CT/ModelArts/T1/Datasets/UCMerced_LandUse",
                "WHU-BCD": "W:/Data/T1/CD/WHU-BCD",
                "CDD": "W:/Data/T1/CD/CDD",
        },
    "num_parallel_workers":8,
    "batch_size": 8,
    "speed_up_nums":128,
    # Model
    "pretrained":False,
    "resume":"", 
    # Train & Eval
    "eval_epochs":5,
    "start_eval_epochs":0,
    "eval_traindata":True,
    "epoch_size": 300,
    "loss_monitor_step":10,
    "metrics_List":["acc", "F1"],  # "acc", "F1", "kappa"
    "save_metrics_List":["F1"],
    # Log
    "save_model_path":r"W:\Code\T1\Models",
    "log_path":r"W:\Code\T1\Logs",
    # LR
    "lr_init":5e-5,
    "lr_max":5e-5,
    "lr_end":1e-5,
    "warmup_epochs":0,
    # ProcessLine
    "scale_factor_List":[10],
    # Backbone
    ## norm
    "norm":True,
    ## backbone structure
    "backbones":[
        # A
        {
            "layer_nums":6,
            "conv_nums":[2,4,2,4,2,4],
            "dims":[128,128,256,256,512,512],
            "selfconv_flags":[False,False,False,False,False,False]
        },
        # # B
        # {
        #     "layer_nums":6,
        #     "conv_nums":[2,4,2,4,2,4],
        #     "dims":[64,64,128,128,256,256],
        #     "selfconv_flags":[False,False,False,True,True,True]
        # },
        # # D
        # {
        #     "layer_nums":6,
        #     "conv_nums":[3,3,3,3,3,3],
        #     "dims":[64,64,128,128,256,256],
        #     "selfconv_flags":[False,False,False,True,True,True]
        # },
    ]
})
