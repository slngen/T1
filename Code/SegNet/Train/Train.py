'''
Author: CT
Date: 2023-04-03 21:24
LastEditors: CT
LastEditTime: 2023-11-17 19:07
'''
import os
import time
import torch
import random
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader, random_split

from Config import config
from Dataset import create_Dataset
from Backbone import Backbone
from Utilizes import Metrics, Metrics_net, Loss_net

random.seed(config.seed)
torch.manual_seed(config.seed)

if __name__=='__main__':
    '''
    Log
    '''
    # get time stamp with format "2021-01-01_00-00"
    time_stamp = time.strftime("%Y-%m-%d_%H-%M", time.localtime()) +"--Dice--segNet--r"+str(config.seed)+"--s"+str(config.image_size)
    # create log path
    log_path = os.path.join(config.log_path, time_stamp)
    os.makedirs(log_path, exist_ok=True)
    # create model save path
    save_model_path = os.path.join(config.save_model_path, time_stamp)
    os.makedirs(save_model_path, exist_ok=True)
    # create info log file
    info_log_path = os.path.join(log_path, "info.log")
    info_log_file = open(info_log_path, "a")
    # write config info
    info_log_file.write("#"*10+"Config"+"#"*10+"\n")
    info_log_file.write(str(config)+"\n")
    '''
    Dataset
    '''
    dataset = create_Dataset()
    train_ratio = 0.7
    train_size = int(len(dataset) * train_ratio)
    train_dataset, eval_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_dataset = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataset = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    print("###Dataset eval:",len(eval_dataset.dataset),"train:",len(train_dataset.dataset))
    '''
    Network
    '''
    if config.resume != "":
        net = Backbone()
        net_state = torch.load(config.resume)
        new_state_dict = OrderedDict()
        for k, v in net_state.items():
            if not k.startswith("embedding.embeddings."):
                new_state_dict[k] = v
        print("### Load Checkpoint -> ",config.resume.split("/")[-1].split("\\")[-1])
        net.load_state_dict(new_state_dict, strict=False)
    else:
        net = Backbone()
    lossNet = Loss_net()
    print(net)
    # wirte info log
    info_log_file.write("\n"+"#"*10+"Network"+"#"*10+"\n")
    info_log_file.write(str(net)+"\n")
    net.to(config.device)
    lossNet.to(config.device)
    metricNet = Metrics_net()
    '''
    Optimizer
    '''
    optimizer = torch.optim.Adam(net.parameters(), config.lr_init)
    '''
    Train
    '''
    best_f1 = 0
    info_log_file.write("\n"+"#"*10+"Training"+"#"*10+"\n")
    for epoch in tqdm(range(config.epoch_size)):
        # init
        metricNet.clear()
        step_nums = len(train_dataset)

        # train
        net.train()
        step = 0
        loss_avg = 0
        for image, label in train_dataset:
            step += 1
            # to device
            image = image.to(config.device)
            label = label.to(config.device)
            # forward
            output = net(image)
            # loss
            loss = lossNet(output, label)
            # average loss
            loss_avg += loss.cpu().item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print
            if step%config.loss_monitor_step==0:
                loss_avg /= config.loss_monitor_step
                print("step {}/{}, epoch {}/{} --> loss:{}".format(step+1, step_nums, epoch+1, config.epoch_size, loss_avg))
                # write info log
                info_log_file.write("step {}/{}, epoch {}/{} --> loss:{}\n".format(step+1, step_nums, epoch+1, config.epoch_size, loss_avg))
                info_log_file.flush()
                # clear loss avg
                loss_avg = 0
                
        # eval
        if epoch%config.eval_epochs==0:
            net.eval()
            with torch.no_grad():
                # train dataset
                print("#"*10, "train dataset", "#"*10)
                # write info log
                info_log_file.write("\n"+"#"*10+"train dataset"+"#"*10+"\n")
                for image, label in train_dataset:
                    # to device
                    image = image.to(config.device)
                    label = label.to(config.device)
                    # forward
                    output = net(image)
                    # metrics
                    metricNet.update(output, label)
                # metrics
                CM = metricNet.get()
                result = Metrics(CM)
                for key, value in result.items():
                    print(key, "--> ", value)
                    # wirte info log
                    info_log_file.write(key+"--> "+str(value)+"\n")
                    if key == "F1":
                        f1_socre = value["F1"][0]

                # eval dataset
                metricNet.clear()
                print("#"*10, "eval dataset", "#"*10)
                # write info log
                info_log_file.write("#"*10+"eval dataset"+"#"*10+"\n")
                for image, label in eval_dataset:
                    # to device
                    image = image.to(config.device)
                    label = label.to(config.device)
                    # forward
                    output = net(image)
                    # metrics
                    metricNet.update(output, label)
                # metrics
                CM = metricNet.get()
                result = Metrics(CM)
                for key, value in result.items():
                    print(key, "--> ", value)
                    # wirte info log
                    info_log_file.write(key+"--> "+str(value)+"\n")
                    info_log_file.flush()
                    if key == "F1":
                        f1_socre = value["F1"][0]
                        if f1_socre > best_f1:
                            best_f1 = f1_socre
                            torch.save(net.state_dict(), os.path.join(save_model_path,"unet-E{}-{:.4f}.ckpt".format(epoch,best_f1)))
                            print("save model!")
                            # write info log
                            info_log_file.write("save model!\n")
                            info_log_file.flush()