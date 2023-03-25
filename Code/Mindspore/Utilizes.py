'''
Author: CT
Date: 2023-01-16 11:50
LastEditors: CT
LastEditTime: 2023-02-08 21:41
'''
from Config import config

class Task_Info():
    def __init__(self):
        print("="*10, "Task Info", "="*10)
        # Check channel mode
        if config.channel_mode not in ["overlap", "order"]:
            print("channel mode error!")
            raise NotImplementedError
        self.task_flag_Dict = config.task_flag_Dict
        self.dataset_path_Dict = config.dataset_path_Dict
        self.task_channels_decoder = config.task_channels_decoder
        # Cal input channel nums. 
        self.input_channels = 0
        self.task_seq = []
        for task_index, task_list in self.task_flag_Dict.items():
            if len(task_list):
                print("{}:".format(task_index))
                for task in task_list:
                    self.task_seq.append(task)
                    print("\t{},\n".format(task))
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

global task_info
task_info = Task_Info()