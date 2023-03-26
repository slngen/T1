'''
Author: CT
Date: 2023-03-18 12:35
LastEditors: CT
LastEditTime: 2023-03-23 08:18
'''
import mindspore.nn as nn
import mindspore.ops as ops

from Config import config

def conv_step(in_channels, out_channels, kernel_size=3, stride=1, pad_mode="same", padding=0, has_bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode, padding, has_bias=has_bias)

def make_encoder(in_channels, out_channels, conv_nums, selfconv_flag):
    cells = nn.CellList([conv_step(in_channels, out_channels), nn.ReLU()])
    if config.norm:
        cells.append(nn.BatchNorm2d(num_features=out_channels))
    for _ in range(conv_nums-1):
        cells.append(conv_step(out_channels, out_channels))
        cells.append(nn.ReLU())
        if config.norm:
            cells.append(nn.BatchNorm2d(num_features=out_channels))
    cells.append(nn.MaxPool2d(kernel_size=2, stride=2))
    # 1x1 Conv
    if selfconv_flag:
        cells.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1))
        cells.append(nn.ReLU())
    return cells

def make_decoder(in_channels):
    decoder = nn.SequentialCell([
        # nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, pad_mode="same", padding=0, has_bias=False),
        # nn.ReLU(),
        # nn.BatchNorm2d(num_features=in_channels),
        nn.Conv2d(in_channels=in_channels, out_channels=len(config.label_graph_mode)*config.class_nums, kernel_size=1, stride=1, pad_mode="same", padding=0, has_bias=False),
    ])
    return decoder

class Backbone(nn.Cell):
    def __init__(self, backbone_config, input_channels):
        super(Backbone, self).__init__()
        self.input_channels = input_channels
        self.image_size = config.image_size
        self.layer_nums = backbone_config.layer_nums
        self.conv_nums = backbone_config.conv_nums
        self.dims = backbone_config.dims
        self.dims.insert(0, self.input_channels)
        self.selfconv_flags = backbone_config.selfconv_flags
        self.class_nums = config.class_nums
        '''
        Encoder
        '''
        self.encoders = nn.CellList([])
        for layer_index in range(self.layer_nums):
            self.encoders.append(
                make_encoder(
                        in_channels = self.dims[layer_index], 
                        out_channels = self.dims[layer_index+1], 
                        conv_nums = self.conv_nums[layer_index],
                        selfconv_flag = self.selfconv_flags[layer_index]
                    )
                )

        '''
        Decoder
        '''
        self.decoders = nn.CellList([])
        for layer_index in range(self.layer_nums):
            self.decoders.append(
                make_decoder(self.dims[layer_index+1])
            )

        self.resize = nn.ResizeBilinear()

    def construct(self, x):
        # Encode
        feature_List = []
        for layer_index, encoder in enumerate(self.encoders):
            for cell in encoder:
                x = cell(x)
            feature_List.append(x)
        # Decode
        PL_List = []
        for layer_index in range(self.layer_nums):
            f = feature_List[layer_index]
            f = self.decoders[layer_index](f)
            f = self.resize(f, size=(self.image_size,self.image_size))
            f = ops.transpose(f, (0,2,3,1))
            PL_List.append([f[:,:,:,PLout_index*config.class_nums:(PLout_index+1)*config.class_nums] for PLout_index in range(int(f.shape[-1]/2))])

        return PL_List