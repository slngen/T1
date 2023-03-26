'''
Author: CT
Date: 2022-12-09 10:36
LastEditors: CT
LastEditTime: 2023-03-21 20:51
'''
import os
import time
import numpy as np
from tqdm import tqdm

from Nets import *
from Config import config
from Dataset import create_Dataset

from mindspore import nn, context, Tensor, Model, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.train.callback import Callback, LossMonitor

def get_lr(lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_max - (lr_max - lr_end) * (i - warmup_steps) / (total_steps - warmup_steps)
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)

    return lr_each_step

class MonitorCallBack(Callback):
    def __init__(self, model, datasets, Logs_dict):
        self.model = model
        self.datasets = datasets
        self.Logs_dict = Logs_dict

        self.pbar = None
        self.save_metrics_List = config.save_metrics_List
        self.max_eval_metrics = {}
        for save_metrics in self.save_metrics_List:
            self.max_eval_metrics[save_metrics] = 0
        self.eval_epochs = config.eval_epochs
        self.start_eval_epochs = config.start_eval_epochs
        self.n_class = config.class_nums
        self.metrics_List = config.metrics_List
        self.scale_factor_nums = len(config.scale_factor_List)+1
        self.config = config
        self.label_graph_mode = config.label_graph_mode

    def begin(self, run_context):
        cb_params = run_context.original_args()
        self.pbar = tqdm(total=cb_params.epoch_num)

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        print("Epoch--> {}".format(epoch))
        self.Logs_dict["info"].write("\nEpoch--> {}".format(epoch))
        if config.SG_level == "image":
            score_List = cb_params.network.network.score_List
            print("score:")
            self.Logs_dict["info"].write("\nScore:")
            for score_index, score in enumerate(score_List):
                print("\t{}-> {}".format(self.label_graph_mode[score_index],score.asnumpy()))
                self.Logs_dict["info"].write("\n\t{}-> {}".format(self.label_graph_mode[score_index],score.asnumpy()))

            
        self.Logs_dict["info"].flush()
        if epoch % self.eval_epochs == 0 and epoch>=self.start_eval_epochs:
            for phase, dataset in self.datasets.items():
                print("<{}> Phase Result:".format(phase))
                self.Logs_dict["info"].write("\n<{}> Phase Result:".format(phase))
                self.Logs_dict["info"].flush()
                CM_List = self.model.eval(dataset)["CM"]
                result_List = [self.metrics(CM, self.metrics_List) for CM in CM_List]
                for result_index, result in enumerate(result_List):
                    print("#"*10, self.label_graph_mode[result_index], "#"*10)
                    self.Logs_dict["info"].write("\n----- "+self.label_graph_mode[result_index]+" -----")
                    self.Logs_dict["info"].flush()
                    for k, v in result.items():
                        if isinstance(v, dict):  # e.g. F1
                            for kk, vv in v.items():
                                save_flag = True if kk in self.save_metrics_List else False
                                vv = np.array(vv)
                                self.logging(kk, vv, phase, save_flag, cb_params, self.label_graph_mode[result_index])
                        else:
                            save_flag = True if k in self.save_metrics_List else False
                            self.logging(k, v, phase, save_flag, cb_params, self.label_graph_mode[result_index])
        self.pbar.update()

    def metrics(self, CM, metrics_List):
        result = {}
        smooth = 1e-5
        if "acc" in metrics_List:
            acc_List = []
            acc = 100*np.diag(CM).sum()/(CM.sum()+smooth)
            acc_List.append(acc)
            result["acc"] = acc_List
        if "F1" in metrics_List:
            F1_Dict = {
                "F1":[],
                "precision":[],
                "recall":[],
            }
            # macro-F1
            P = np.zeros(self.n_class)
            R = np.zeros(self.n_class)
            for cls_index in range(self.n_class):
                P[cls_index] = CM[cls_index, cls_index] / (CM[:,cls_index].sum()+smooth)
                R[cls_index] = CM[cls_index, cls_index] / (CM[cls_index,:].sum()+smooth)
            # P = P.mean()
            # R = R.mean()
            P = P[-1]
            R = R[-1]
            F1 = 2*P*R/(P+R+smooth)
            F1_Dict["F1"].append(F1)
            F1_Dict["precision"].append(P)
            F1_Dict["recall"].append(R)
            result["F1"] = F1_Dict
        if "kappa" in metrics_List:
            kappa_List = []
            P0 = np.diag(CM).sum()/(CM.sum()+smooth)
            Pe = 0
            for cls_index in range(self.n_class):
                Pe += CM[:,cls_index].sum()*CM[cls_index,:].sum()
            Pe /= np.power(CM.sum(), 2)
            kappa = (P0-Pe)/(1-Pe)
            kappa_List.append(kappa)
            result["kappa"] = kappa_List
        return result
    
    def logging(self, k, v, phase, save_flag, cb_params, label_graph_mode):
        log = self.Logs_dict["{}_{}".format(phase, k)]
        v = np.round(v,3)
        v_str = ", ".join(str(vi) for vi in v)
        print("\t{}:{}".format(k, v_str))
        if label_graph_mode == self.label_graph_mode[0]:
            log.write("\n{}".format(v_str))
        else:
            log.write("\t,{}".format(v_str))
        log.flush()
        self.Logs_dict["info"].write("\n\t{}:{}".format(k, v_str))
        self.Logs_dict["info"].flush()
        if save_flag and phase=="eval":
            v_max = v.max()
            if v_max > self.max_eval_metrics[k]:
                self.max_eval_metrics[k] = v_max
                # save net
                print("##### Save Best <{}> Model!".format(k))
                self.Logs_dict["info"].write("\n##### Save Best <{}> Model!".format(k))
                self.Logs_dict["info"].flush()
                save_checkpoint(
                                save_obj=cb_params.train_network,
                                ckpt_file_name=os.path.join(
                                                    self.config.save_model_path,
                                                    self.Logs_dict["time_stamp"],
                                                    "{}_{}-{}_epoch-{}.ckpt".format(
                                                        label_graph_mode,
                                                        k, 
                                                        v_max, 
                                                        cb_params.cur_epoch_num
                                                        )
                                                    )
                                )

class metrics_CM(nn.Metric):
    def __init__(self):
        super().__init__()
        self.n_class = config.class_nums
        self.scale_factor_nums = len(config.scale_factor_List)+1  # additional 1 for un-weighted vote.
        self.out_nums = config.PL_nums+self.scale_factor_nums
        self.label_graph_mode = config.label_graph_mode
        self.CM_List = [np.zeros((self.n_class, self.n_class)) for _ in range(len(self.label_graph_mode))]
    def clear(self):
        self.CM_List = [np.zeros((self.n_class, self.n_class)) for _ in range(len(self.label_graph_mode))]

    def eval(self):
        return self.CM_List

    def update(self, *data):
        label_List = data[1]
        for PLout_index, PLout in enumerate(data[0]):
            Output = (ops.Softmax(3)(PLout).asnumpy()).astype(np.float32)
            Prediction = np.argmax(Output, axis=3).flatten()
            Label = (label_List[PLout_index]).asnumpy().astype(int).flatten()
            cm = np.bincount(self.n_class * Label + Prediction, minlength=self.n_class*self.n_class).reshape(self.n_class, self.n_class)
            self.CM_List[PLout_index] += cm

if __name__ == '__main__':
    '''
    Context
    '''
    if config.context_mode == "GRAPH":
        context.set_context(device_target=config.device_target, mode=context.GRAPH_MODE)
    else:
        context.set_context(device_target=config.device_target, mode=context.PYNATIVE_MODE)
    '''
    Dataset
    '''
    dataset_train = create_Dataset(
                                    batch_size=config.batch_size, 
                                    shuffle=True,
                                    speed_flag=False,
                                    mode="train")
    if config.eval_traindata:
        dataset_train_backup = create_Dataset(
                                        batch_size=config.batch_size, 
                                        shuffle=True,
                                        speed_flag=False,
                                        mode="train")
    dataset_eval = create_Dataset(        
                                    batch_size=config.batch_size, 
                                    shuffle=True,
                                    speed_flag=False,
                                    mode="eval")
    step_size = dataset_train.get_dataset_size()
    if config.eval_traindata:
        datasets = {"train":dataset_train_backup, "eval":dataset_eval}
    else:
        datasets = {"eval":dataset_eval}
    # for d in dataset_eval:
    #     pass
    '''
    Logs
    '''
    time_stamp = time.strftime('%Y-%m-%d_%H-%M',time.localtime(int(round(time.time()*1000))/1000))
    time_stamp += task_info.get_task_str()
    if not os.path.exists(os.path.join(config.log_path,time_stamp)):
        os.makedirs(os.path.join(config.log_path,time_stamp))
    if not os.path.exists(os.path.join(config.save_model_path,time_stamp)):
        os.makedirs(os.path.join(config.save_model_path,time_stamp))
    if config.eval_traindata:
        phase_List = ["train","eval"]
    else:
        phase_List = ["eval"]
    Logs_dict = {}
    Logs_dict["time_stamp"] = time_stamp
    for phase in phase_List:
        for metrics_index in config.metrics_List:
            if metrics_index == "F1":
                Logs_dict["{}_F1".format(phase)] = open(os.path.join(config.log_path,time_stamp,"{}_F1.txt".format(phase)),"a")
                Logs_dict["{}_precision".format(phase)] = open(os.path.join(config.log_path,time_stamp,"{}_precision.txt".format(phase)),"a")
                Logs_dict["{}_recall".format(phase)] = open(os.path.join(config.log_path,time_stamp,"{}_recall.txt".format(phase)),"a")
            else:
                Logs_dict["{}_{}".format(phase, metrics_index)] = open(os.path.join(config.log_path,time_stamp,"{}_{}.txt".format(phase, metrics_index)),"a")
    log_info = open(os.path.join(config.log_path,time_stamp,"info.txt"),"a")
    Logs_dict["info"] = log_info
    log_info.write("########## Config ##########\n")
    for key in config.keys():
        log_info.write("{}:{}\n".format(key,config[key]))
    log_info.flush()
    '''
    lr
    '''
    lr = get_lr(
                    lr_init=config.lr_init,
                    lr_end=config.lr_end,
                    lr_max=config.lr_max,
                    warmup_epochs=config.warmup_epochs,
                    total_epochs=config.epoch_size,
                    steps_per_epoch=step_size
                )
    np.savetxt(os.path.join(config.log_path, time_stamp, "lr.txt"), lr)
    lr = Tensor(lr)
    '''
    Model
    '''
    train_net = Train_net()
    if config.resume:
        ckpt = load_checkpoint(config.resume)
        load_param_into_net(train_net, ckpt)
    optimizer = nn.Adam([
            {'params': [params for params in train_net.trainable_params() if not "SG" in params.name], "lr": lr},
            {'params': train_net.SG.trainable_params(), "lr": lr},
        ])
    eval_net = Eval_net(train_net)
    withloss_net = WithLoss_net(train_net,Loss_fn())
    print("========== Net ==========\n")
    print(withloss_net)
    model = Model(network=withloss_net, optimizer=optimizer, eval_network=eval_net, metrics={"CM":metrics_CM()})
    '''
    Train
    '''
    print("="*10, "Start Training", "="*10)
    log_info.write(("########## Train ##########\n"))
    log_info.flush()
    model.train(
                    config.epoch_size, 
                    dataset_train, 
                    callbacks=[LossMonitor(config.loss_monitor_step), MonitorCallBack(model, datasets, Logs_dict)]
                )