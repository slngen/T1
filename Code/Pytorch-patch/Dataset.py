'''
Author: CT
Date: 2022-12-09 10:36
LastEditors: CT
LastEditTime: 2023-07-26 08:47
'''
import os
import cv2
import numpy as np
from PIL import Image
from skimage import transform

import random
random.seed(3)

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as py_vision

from Config import config
from Utilizes import task_info

# class BCD_Datasets:
#     def __init__(self, data_path):
#         self.path_imageA_root = os.path.join(data_path, "A")
#         self.path_imageB_root = os.path.join(data_path, "B")
#         self.path_label_root = os.path.join(data_path, "label")

#         self.toTensor = py_vision.py_transforms.ToTensor()
#         self.files = os.listdir(self.path_label_root)

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         imageA_path = os.path.join(self.path_imageA_root, self.files[idx])
#         imageA = Image.open(imageA_path)
#         imageB_path = os.path.join(self.path_imageB_root, self.files[idx])
#         imageB = Image.open(imageB_path)
#         label_path = os.path.join(self.path_label_root, self.files[idx])
#         label = Image.open(label_path).convert('L')

#         imageA = self.toTensor(imageA)
#         imageB = self.toTensor(imageB)
#         label = self.toTensor(label)[0]

#         return [imageA, imageB], label

# class GID_Datasets:
#     def __init__(self, data_path, speed_flag):
#         self.path_image_root = os.path.join(data_path, "image")
#         self.path_label_root = os.path.join(data_path, "label")

#         self.toTensor = py_vision.py_transforms.ToTensor()
#         self.files = os.listdir(self.path_label_root)
#         if speed_flag:
#             self.files = self.files[:2048]

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.path_image_root, self.files[idx])
#         image = Image.open(image_path).convert('RGB')
#         label_path = os.path.join(self.path_label_root, self.files[idx])
#         label = Image.open(label_path).convert('RGB')

#         image = self.toTensor(image)
#         label = self.toTensor(label)
#         # Clear Label
#         label = np.where(label>0.5,1,label)
#         label = np.where(label<0.5,0,label)
#         label_cls = label[0] + 2*label[1] + 4*label[2]
#         label_cls = np.where(label_cls==6,5,label_cls)
#         ### BUG
#         # Some points value of label_cls is out of range?
#         label_cls = np.where(label_cls>=7,0,label_cls)
#         if label_cls.max()>5 or label_cls.min()<0:
#             print("###ERROR###\n"*5,"FILE:{}".format(self.files[idx]),"\n\tLabel.max:{}\n\tLabel.min:{}".format(label_cls.max(), label_cls.min()))
#             raise ValueError

#         return image, label_cls

# class VOC_Datasets:
#     def __init__(self, data_path, speed_flag, mode):
#         self.mode = mode
#         if mode == "train":
#             index_file = os.path.join(data_path,"ImageSets","Segmentation","train.txt")
#         else:
#             index_file = os.path.join(data_path,"ImageSets","Segmentation","val.txt")
#         with open(index_file) as f:
#             self.files = f.read().splitlines()
#             f.close()
        
#         self.path_image_root = os.path.join(data_path, "JPEGImages")
#         self.path_label_root = os.path.join(data_path, "SegmentationClass")

#         self.toTensor = py_vision.py_transforms.ToTensor()

#         if speed_flag:
#             self.files = self.files[:2048]

#     def __len__(self):
#         return len(self.files)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.path_image_root, self.files[idx]+".jpg")
#         image = Image.open(image_path)  # RGB mode
#         label_path = os.path.join(self.path_label_root, self.files[idx]+".png")
#         label = Image.open(label_path)  # P mode
        
#         # image = np.array(image).transpose((2,0,1)) / 255
#         image = self.toTensor(image)
#         label = np.array(label)
#         label = np.where(label==255,0,label)
    
#         if label.max()>config.class_nums or label.min()<0:
#             print("###ERROR###\n"*5,"\tFILE:{}\n\tLabel.max:{}\n\tLabel.min:{}\n\tIs Out Of Label Range!!!".format(self.files[idx],label.max(), label.min()))
#             raise ValueError
        
#         # Resize
#         image = image[:, :config.image_size, :config.image_size]
#         image = np.pad(image, ((0,0), (0, config.image_size-image.shape[1]), (0, config.image_size-image.shape[2])))
#         label = label[:config.image_size, :config.image_size]
#         label = np.pad(label, ((0, config.image_size-label.shape[0]), (0, config.image_size-label.shape[1])))

#         return image, label

class Datasets(Dataset):
    def __init__(self, speed_flag, mode, train_rate):
        self.img_size = config.image_size
        self.channel_mode = config.channel_mode
        self.input_channels = task_info.get_input_channels()
        self.dataset_path_Dict = task_info.dataset_path_Dict
        self.task_flag_List = task_info.get_task_list()
        self.data_List = []
        self.toTensor = py_vision.ToTensor()
        self.label_graph_mode = config.label_graph_mode
        end_channel = 0
        '''
        RSOD-Aircraft
        '''
        if "RSOD-Aircraft" in self.task_flag_List:
            begin_channel = end_channel
            end_channel += config.task_channels_decoder["RSOD-Aircraft"]
            RSOD_data_path = self.dataset_path_Dict["RSOD-Aircraft"]
            image_root_path = os.path.join(RSOD_data_path, "JPEGImages")
            label_root_path = os.path.join(RSOD_data_path, "Annotation", "labels")
            info_file_List = os.listdir(label_root_path)
            info_file_List.sort()

            sub_data_List = []
            for info_file in info_file_List:
                info_label_List = []
                with open(os.path.join(RSOD_data_path, "Annotation", "labels", info_file), "r") as f:
                    info_List = f.readlines()
                    for info in info_List:
                        info = info.strip()
                        _, _, x_min, y_min, x_max, y_max = info.split("\t")
                        info_label_List.append([int(x_min), int(y_min), int(x_max), int(y_max)])
                    f.close()
                    sub_data_List.append({
                        "path": os.path.join(image_root_path, info_file.split(".")[0]+".jpg"), 
                        "label": info_label_List,
                        "task_flag": task_info.encode_task("RSOD-Aircraft"),
                        "end_channel": end_channel,
                        "begin_channel": begin_channel
                    })

            train_nums = int(len(sub_data_List)*train_rate)
            if mode == "train":
                if speed_flag:
                    sub_data_List = sub_data_List[:config.speed_up_nums]
                else:
                    sub_data_List = sub_data_List[:train_nums]
            else:
                if speed_flag:
                    sub_data_List = sub_data_List[-config.speed_up_nums:]
                else:
                    sub_data_List = sub_data_List[train_nums:]
            self.data_List.extend(sub_data_List)
        '''
        UCMLU
        '''
        if "UCMLU" in self.task_flag_List:
            begin_channel = end_channel
            end_channel += config.task_channels_decoder["UCMLU"]
            UCMLU_data_path = self.dataset_path_Dict["UCMLU"]
            self.class_List = os.listdir(UCMLU_data_path)
            self.class_List.sort()
            self.label2id = {}

            sub_data_List = []
            for cls_index in self.class_List:
                if cls_index not in self.label2id:
                    self.label2id[cls_index] = len(self.label2id) + 2
                image_List = os.listdir(os.path.join(UCMLU_data_path, cls_index))
                cls_len = len(image_List)
                train_nums = int(cls_len*train_rate)
                if mode == "train":
                    if speed_flag:
                        image_List = image_List[:config.speed_up_nums]
                    else:
                        image_List = image_List[:train_nums]
                else:
                    if speed_flag:
                        image_List = image_List[-config.speed_up_nums:]
                    else:
                        image_List = image_List[train_nums:]
                for image_name in image_List:
                    sub_data_List.append(
                                {
                                    "path": os.path.join(UCMLU_data_path, cls_index, image_name),
                                    "label": self.label2id[cls_index],
                                    "task_flag": task_info.encode_task("UCMLU"),
                                    "end_channel": end_channel,
                                    "begin_channel": begin_channel
                                }
                            )
                self.data_List.extend(sub_data_List)
        '''
        WHU-BCD
        '''
        if "WHU-BCD" in self.task_flag_List:
            begin_channel = end_channel
            end_channel += config.task_channels_decoder["WHU-BCD"]
            WHU_BCD_data_path = self.dataset_path_Dict["WHU-BCD"]
            if mode== "train":
                WHU_BCD_data_path = os.path.join(WHU_BCD_data_path, "Train")
            else:
                WHU_BCD_data_path = os.path.join(WHU_BCD_data_path, "Eval")
            self.files = os.listdir(os.path.join(WHU_BCD_data_path, "label"))

            sub_data_List = []
            for file in self.files:
                sub_data_List.append(
                                {
                                    "path": {
                                                "A": os.path.join(WHU_BCD_data_path, "A", file), 
                                                "B": os.path.join(WHU_BCD_data_path, "B", file), 
                                            },
                                    "label": os.path.join(WHU_BCD_data_path, "label", file), 
                                    "task_flag": task_info.encode_task("WHU-BCD"),
                                    "end_channel": end_channel,
                                    "begin_channel": begin_channel
                                }
                            )
            train_nums = int(len(sub_data_List)*train_rate)
            if mode == "train":
                if speed_flag:
                    sub_data_List = sub_data_List[:config.speed_up_nums]
                else:
                    sub_data_List = sub_data_List[:train_nums]
            else:
                if speed_flag:
                    sub_data_List = sub_data_List[-config.speed_up_nums:]
                else:
                    sub_data_List = sub_data_List[train_nums:]
            self.data_List.extend(sub_data_List)
        '''
        CDD
        '''
        if "CDD" in self.task_flag_List:
            begin_channel = end_channel
            end_channel += config.task_channels_decoder["CDD"]
            WHU_BCD_data_path = self.dataset_path_Dict["CDD"]
            if mode== "train":
                WHU_BCD_data_path = os.path.join(WHU_BCD_data_path, "train")
            else:
                WHU_BCD_data_path = os.path.join(WHU_BCD_data_path, "val")
            self.files = os.listdir(os.path.join(WHU_BCD_data_path, "OUT"))

            sub_data_List = []
            for file in self.files:
                sub_data_List.append(
                                {
                                    "path": {
                                                "A": os.path.join(WHU_BCD_data_path, "A", file), 
                                                "B": os.path.join(WHU_BCD_data_path, "B", file), 
                                            },
                                    "label": os.path.join(WHU_BCD_data_path, "OUT", file), 
                                    "task_flag": task_info.encode_task("CDD"),
                                    "end_channel": end_channel,
                                    "begin_channel": begin_channel
                                }
                            )
            train_nums = int(len(sub_data_List)*train_rate)
            if mode == "train":
                if speed_flag:
                    sub_data_List = sub_data_List[:config.speed_up_nums]
                else:
                    sub_data_List = sub_data_List[:train_nums]
            else:
                if speed_flag:
                    sub_data_List = sub_data_List[-config.speed_up_nums:]
                else:
                    sub_data_List = sub_data_List[train_nums:]
            self.data_List.extend(sub_data_List)
            
    def __len__(self):
        return len(self.data_List)

    def __getitem__(self, idx):
        # random_x = random.randint(0, 511-64)
        # random_y = random.randint(0, 511-64)
        data = self.data_List[idx]
        # Image
        if task_info.decode_task(data["task_flag"]) in ["RSOD-Aircraft", "UCMLU"]:
            image_path = data["path"]
            image_ = Image.open(image_path).convert('RGB')
            image = self.toTensor(image_)
            image_.close()
            img_shape = image.shape
            image = transform.resize(image, (image.shape[0], self.img_size, self.img_size), order=2)
            if self.channel_mode == "order":
                embed_pad_after = np.zeros((self.input_channels-data["end_channel"], self.img_size, self.img_size), dtype=np.float32)
                embed_pad_before = np.zeros((data["begin_channel"], self.img_size, self.img_size), dtype=np.float32)
                image = np.concatenate((embed_pad_before, image, embed_pad_after), axis=0)
            elif self.channel_mode == "overlap" and self.input_channels!=data["end_channel"]-data["begin_channel"]:
                embed_pad = np.zeros((self.input_channels-data["end_channel"]+data["begin_channel"], self.img_size, self.img_size), dtype=np.float32)
                image = np.concatenate((embed_pad, image), axis=0)
        elif task_info.decode_task(data["task_flag"]) in ["WHU-BCD", "CDD"]:
            image_path_A = data["path"]["A"]
            image_path_B = data["path"]["B"]
            image_A = Image.open(image_path_A).convert('RGB')
            image_B = Image.open(image_path_B).convert('RGB')
            image_A = self.toTensor(image_A)
            image_B = self.toTensor(image_B)
            image = np.concatenate([image_A, image_B], axis=0)
            image = transform.resize(image, (image.shape[0], self.img_size, self.img_size), order=2)
            if self.channel_mode == "order":
                embed_pad_after = np.zeros((self.input_channels-data["end_channel"], self.img_size, self.img_size), dtype=np.float32)
                embed_pad_before = np.zeros((data["begin_channel"], self.img_size, self.img_size), dtype=np.float32)
                image = np.concatenate((embed_pad_before, image, embed_pad_after), axis=0)
            elif self.channel_mode == "overlap" and self.input_channels!=data["end_channel"]-data["begin_channel"]:
                embed_pad = np.zeros((self.input_channels-data["end_channel"]+data["begin_channel"], self.img_size, self.img_size), dtype=np.float32)
                image = np.concatenate((embed_pad, image), axis=0)
        else:
            raise NotImplementedError
        image = torch.Tensor(image)
        # image = image[:,random_x:random_x+64,random_y:random_y+64]
        # Label
        label_List = []
        if task_info.decode_task(data["task_flag"]) == "RSOD-Aircraft":
            label = np.zeros((self.img_size, self.img_size), dtype=np.int32)
            for info_label in data["label"]:
                _, y_min, _, y_max = map(lambda x:int(x*self.img_size/img_shape[1]), info_label)
                x_min, _, x_max, _ = map(lambda x:int(x*self.img_size/img_shape[2]), info_label)
                label[y_min:y_max, x_min:x_max] = 1
        elif task_info.decode_task(data["task_flag"]) == "UCMLU":
            label = np.full((self.img_size, self.img_size), data["label"], dtype=np.int32)
        elif task_info.decode_task(data["task_flag"]) in ["WHU-BCD", "CDD"]:
            label = Image.open(data["label"]).convert('RGB')
            label = self.toTensor(label)
            label = transform.resize(label, (1, self.img_size, self.img_size), order=2)
        else:
            raise NotImplementedError
        
        # label = label[:,random_x:random_x+64,random_y:random_y+64]

        if "full" in self.label_graph_mode:
            label_List.append(label)
        # Edge-Label
        smooth = 1e-5
        if "edge-3" in self.label_graph_mode:
            label_numpy = label.copy()[0]
            maxValue=label_numpy.max()
            label_numpy=label_numpy*255/(maxValue+smooth)
            label_numpy=np.uint8(label_numpy)
            label_edge = cv2.Canny(label_numpy, 0, 255)
            # 创建一个核，用于形态学操作
            kernel = np.ones((3, 3), np.uint8)
            # 使用膨胀操作增加边缘宽度
            label_edge = cv2.dilate(label_edge, kernel, iterations=1)
            label_edge = self.toTensor(label_edge)
            label_List.append(label_edge)
        if "edge-5" in self.label_graph_mode:
            label_numpy = label.copy()[0]
            maxValue=label_numpy.max()
            label_numpy=label_numpy*255/(maxValue+smooth)
            label_numpy=np.uint8(label_numpy)
            label_edge = cv2.Canny(label_numpy, 0, 255)
            # 创建一个核，用于形态学操作
            kernel = np.ones((5, 5), np.uint8)
            # 使用膨胀操作增加边缘宽度
            label_edge = cv2.dilate(label_edge, kernel, iterations=1)
            label_edge = self.toTensor(label_edge)
            label_List.append(label_edge)
        if "edge-7" in self.label_graph_mode:
            label_numpy = label.copy()[0]
            maxValue=label_numpy.max()
            label_numpy=label_numpy*255/(maxValue+smooth)
            label_numpy=np.uint8(label_numpy)
            label_edge = cv2.Canny(label_numpy, 0, 255)
            # 创建一个核，用于形态学操作
            kernel = np.ones((7, 7), np.uint8)
            # 使用膨胀操作增加边缘宽度
            label_edge = cv2.dilate(label_edge, kernel, iterations=1)
            label_edge = self.toTensor(label_edge)
            label_List.append(label_edge)
        # squeeze label in label_List
        label_List = [torch.Tensor(label.squeeze()).long() for label in label_List]
        return image, label_List, data["task_flag"]

def create_Dataset(batch_size, shuffle=True, speed_flag=False, mode="train", train_rate=0.7):
    datasets = Datasets(speed_flag, mode, train_rate)
    datasets = DataLoader(datasets, batch_size=batch_size, shuffle=shuffle, num_workers=config.num_parallel_workers, persistent_workers=True)
    return datasets

if __name__ == "__main__":
    datasets = create_Dataset(batch_size=8, shuffle=True)
    for data in datasets:
        pass