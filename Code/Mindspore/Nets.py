'''
Author: CT
Date: 2022-12-09 10:36
LastEditors: CT
LastEditTime: 2023-03-26 20:02
'''
from Backbone import Backbone
from Config import config
from Utilizes import task_info

import numpy as np
np.random.seed(3)

import mindspore
from mindspore import Parameter, Tensor, ParameterTuple
import mindspore.nn as nn
import mindspore.ops as ops

class SG_net(nn.Cell):
    def __init__(self, SG_level):
        super(SG_net, self).__init__()
        self.SG_level = SG_level
        self.label_graph_mode = config.label_graph_mode
        if self.SG_level == "image":
            self.score_seed_List = ParameterTuple(
                tuple([Parameter(Tensor(np.ones((1,config.PL_nums)), mindspore.float32), name="score_seed_{}".format(mode_index)) for mode_index in range(len(self.label_graph_mode))])
                )
            self.softmax = ops.Softmax()
            self.SG_Seq = nn.SequentialCell([
                    nn.Dense(config.PL_nums, config.SG_dims),
                    nn.ReLU(),
                    nn.Dense(config.SG_dims, config.PL_nums),
                ])
        elif self.SG_level == "pixel":
            self.SG_Seq = nn.SequentialCell([
                    nn.Conv2d(config.class_nums, config.SG_dims, 3, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=config.SG_dims),
                    nn.Conv2d(config.SG_dims, 3*config.SG_dims, 3, 1),
                    nn.ReLU(),
                    nn.BatchNorm2d(num_features=3*config.SG_dims),
                    nn.Conv2d(3*config.SG_dims, config.class_nums, 3, 1, has_bias=True),
                ])
            self.sigmoid = ops.Sigmoid()
            raise NotImplementedError
        else:
            raise NotImplementedError

    def construct(self, input=0):
        if self.SG_level == "image":
            score_List = []
            for index in range(len(self.score_seed_List)):
                score_List.append(self.SG_Seq(self.score_seed_List[index])[0])
                score_List[index] = self.softmax(score_List[index])
        elif self.SG_level == "pixel":
            score = self.SG_Seq(input)
            score = self.sigmoid(score)
            raise NotImplementedError
        else:
            raise NotImplementedError
        return score_List

class Loss_fn(nn.LossBase):
    def __init__(self, reduction="mean"):
        super(Loss_fn, self).__init__(reduction)
        self.loss_ce = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.loss_mse = nn.MSELoss()
        self.SG_level = config.SG_level
        self.alpha = config.alpha

    def construct(self, PL_List, score_List, label):
        label_List = [label[:,index,:,:,:] for index in range(label.shape[1])]
        label_List = [ops.reshape(label, (-1,)) for label in label_List]
        loss = 0
        for PLout in PL_List:
            for PLout_index, PLout_sub in enumerate(PLout):
                if self.SG_level == "image":
                    PLout_sub = ops.reshape(PLout_sub, (label_List[PLout_index].shape[0], -1))
                    loss += self.get_loss(self.loss_ce(PLout_sub, label_List[PLout_index]))
                elif self.SG_level == "pixel":
                    raise NotImplementedError
                else:
                    raise NotImplementedError
        for score in score_List:
            loss += self.alpha * self.loss_mse(score, Tensor(np.ones(score.shape, dtype=np.float32)/score.shape[-1]))  
        return loss

class Train_net(nn.Cell):
    def __init__(self):
        super(Train_net, self).__init__()
        # Task Info
        self.task_info = task_info
        self.input_channels = self.task_info.get_input_channels()
        # Backbone
        PL_nums = 0
        self.Backbone_List = nn.CellList()
        for backbone_config in config.backbones:
            self.Backbone_List.append(Backbone(backbone_config, self.input_channels))
            # cal PL nums
            PL_nums += backbone_config.layer_nums
        config.PL_nums = PL_nums
        # SG (Score Generator)
        self.SG_level = config.SG_level
        self.SG = SG_net(self.SG_level)
        self.score_List = 0
        self.label_graph_mode = config.label_graph_mode

    def construct(self, images):
        PL_List = []
        for backbone in self.Backbone_List:
            PL_List.extend(backbone(images))
        # Vote
        if self.SG_level == "image":
            self.score_List = self.SG()
            output_List = [0 for _ in range(len(self.score_List))]
            for PL_index, PLout in enumerate(PL_List):
                for PLout_index, PLout_sub in enumerate(PLout):
                    output_List[PLout_index] += PLout_sub * self.score_List[PLout_index][PL_index]
        elif self.SG_level == "pixel":
            for PL_index, PLout in enumerate(PL_List):
                PLout = ops.transpose(PLout, (0,3,1,2))
                score_List = self.SG(PLout)
                weighted_out = PLout * score_List
                weighted_out = ops.transpose(weighted_out, (0,2,3,1))
                output_List += weighted_out
                raise NotImplementedError
        else:
            raise NotImplementedError
        PL_List.append(output_List)
        # Deep Supervision
        if not config.deep_supervision:
            PL_List = PL_List[-1:]
        return PL_List, self.score_List

class WithLoss_net(nn.Cell):
    def __init__(self, network, loss_fn):
        super(WithLoss_net, self).__init__(auto_prefix=False)
        self.network = network
        self.loss_fn = loss_fn

    def construct(self, *data):
        label = data[1].astype(mindspore.int32, copy=False)
        PL_List, score_List = self.network(data[0])
        loss = self.loss_fn(PL_List, score_List, label)
        return loss

class Eval_net(nn.Cell):
    def __init__(self, network):
        super(Eval_net, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, *data):
        label = data[1].astype(mindspore.int32, copy=False)
        label_List = [label[:,index,:,:,:] for index in range(label.shape[1])]
        label_List = [ops.reshape(label, (-1,)) for label in label_List]
        PL_List, score_List = self.network(data[0])
        Task_Flag = data[2]
        return PL_List[-1], label_List, Task_Flag

