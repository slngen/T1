'''
Author: CT
Date: 2023-04-03 21:24
LastEditors: CT
LastEditTime: 2023-04-10 21:27
'''
import os
import time
import torch
from tqdm import tqdm

from Config import config
from Dataset import create_Dataset
from ResNet_FPN import resnet50 
from Utilizes import Metrics, Metrics_net, Loss_net

if __name__=='__main__':
    '''
    Log
    '''
    # get time stamp with format "2021-01-01_00-00"
    time_stamp = time.strftime("%Y-%m-%d_%H-%M", time.localtime())
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
    train_dataset = create_Dataset(batch_size=config.batch_size, shuffle=True, speed_flag=False, mode="train")
    eval_dataset = create_Dataset(batch_size=config.batch_size, shuffle=False, speed_flag=False, mode="eval")
    '''
    Network
    '''
    if config.resume != "":
        net = resnet50(pretrained=False)
        net_state = torch.load(config.resume)
        net.load_state_dict(net_state)
    else:
        net = resnet50(pretrained=config.pretrained)
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
        for image, label_List, task_flag in train_dataset:
            step += 1
            # to device
            image = image.to(config.device)
            label_List = [label.to(config.device) for label in label_List]
            # forward
            output_List = net(image)
            # loss
            loss = lossNet(output_List, label_List)
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
                for image, label_List, task_flag in train_dataset:
                    # to device
                    image = image.to(config.device)
                    label_List = [label.to(config.device) for label in label_List]
                    # forward
                    output_List = net(image)
                    # metrics
                    metricNet.update(output_List, label_List)
                # metrics
                CM_List = metricNet.get()
                result_List = []
                # for PLout_index in range(len((output_List))):
                #     result_List.append([])
                #     for label_index in range(len((label_List))):
                #         result = Metrics(CM_List[label_index][PLout_index])
                #         result_List[PLout_index].append(result)
                #     for key, value in result.items():
                #         print(key, "--> ", value)
                #         # wirte info log
                #         info_log_file.write(key+"--> "+str(value)+"\n")
                #         if key == "F1":
                #             f1_socre = value["F1"][0]
                for PLout_index in range(len((output_List))):
                    print("PLout index: ", PLout_index+2)
                    # wirte info log
                    info_log_file.write("PLout index: "+str(PLout_index+2)+"\n")
                    info_log_file.flush()
                    result_List.append([])
                    result = Metrics(CM_List[PLout_index])
                    result_List[PLout_index].append(result)
                    for key, value in result.items():
                        print(key, "--> ", value)
                        # wirte info log
                        info_log_file.write(key+"--> "+str(value)+"\n")
                        if key == "F1":
                            f1_socre = value["F1"][0]
                # eval dataset
                print("#"*10, "eval dataset", "#"*10)
                # write info log
                info_log_file.write("#"*10+"eval dataset"+"#"*10+"\n")
                for image, label_List, task_flag in eval_dataset:
                    # to device
                    image = image.to(config.device)
                    label_List = [label.to(config.device) for label in label_List]
                    # forward
                    output_List = net(image)
                    # metrics
                    metricNet.update(output_List, label_List)
                # metrics
                CM_List = metricNet.get()
                result_List = []
                # for PLout_index in range(len((output_List))):
                #     result_List.append([])
                #     for label_index in range(len((label_List))):
                #         result = Metrics(CM_List[label_index][PLout_index])
                #         result_List[PLout_index].append(result)
                #     for key, value in result.items():
                #         print(key, "--> ", value)
                #         # wirte info log
                #         info_log_file.write(key+"--> "+str(value)+"\n")
                #         info_log_file.flush()
                #         if key == "F1":
                #             f1_socre = value["F1"][0]
                #             if f1_socre > best_f1:
                #                 best_f1 = f1_socre
                #                 torch.save(net.state_dict(), os.path.join(save_model_path,"ResNet50-FPN-{:.4f}.ckpt".format(best_f1)))
                #                 print("save model!")
                #                 # write info log
                #                 info_log_file.write("save model!\n")
                #                 info_log_file.flush()
                for PLout_index in range(len((output_List))):
                    result_List.append([])
                    result = Metrics(CM_List[PLout_index])
                    result_List[PLout_index].append(result)
                    print("PLout index: ", PLout_index+2)
                    # wirte info log
                    info_log_file.write("PLout index: "+str(PLout_index+2)+"\n")
                    info_log_file.flush()
                    for key, value in result.items():
                        print(key, "--> ", value)
                        # wirte info log
                        info_log_file.write(key+"--> "+str(value)+"\n")
                        info_log_file.flush()
                        if key == "F1":
                            f1_socre = value["F1"][0]
                            if f1_socre > best_f1:
                                best_f1 = f1_socre
                                torch.save(net.state_dict(), os.path.join(save_model_path,"ResNet50-FPN-C{}-{:.4f}.ckpt".format(PLout_index+2,best_f1)))
                                print("save model!")
                                # write info log
                                info_log_file.write("save model!\n")
                                info_log_file.flush()