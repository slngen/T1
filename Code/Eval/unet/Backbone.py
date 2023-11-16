'''
Author: CT
Date: 2023-10-29 16:06
LastEditors: CT
LastEditTime: 2023-11-16 15:23
'''
import torch
import torch.nn as nn
from torchinfo import summary
from Config import config

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, features):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for feature in features:
            self.layers.append(DoubleConv(in_channels, feature))
            self.layers.append(nn.MaxPool2d(2))
            in_channels = feature

    def forward(self, x):
        feature_maps = []
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                x = layer(x)
            else:
                x = layer(x)
                feature_maps.append(x)
        return feature_maps

class Decoder(nn.Module):
    def __init__(self, features, out_channels):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.layers.append(nn.ConvTranspose2d(features[i], features[i - 1], 2, stride=2))
            self.layers.append(DoubleConv(2*features[i - 1], features[i - 1]))
        self.conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, feature_maps):
        x = feature_maps[-1]
        for i in range(0, len(self.layers), 2):
            x = self.layers[i](x)
            skip_connection = feature_maps[-(i // 2 + 2)]
            x = torch.cat((x, skip_connection), dim=1)
            x = self.layers[i + 1](x)
        return self.conv(x)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=3):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class AttentionUnit(nn.Module):
    def __init__(self, out_channels):
        super(AttentionUnit, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(out_channels)
        self.map = DoubleConv(out_channels, out_channels//2)

    def forward(self, x):
        x = self.sa(x) * x
        x = self.ca(x) * x
        x = self.map(x)
        return x
    
class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.layers = nn.ModuleList([
            AttentionUnit(2*config.features[0]),
            AttentionUnit(2*config.features[1]),
            AttentionUnit(2*config.features[2]),
        ])

    def forward(self, features):
        att_features = []
        for feature_index in range(len(features)):
            feature = features[feature_index]
            att_feature = self.layers[feature_index](feature)
            att_features.append(att_feature)
        return att_features

class PositionalEmbedding(nn.Module):
    def __init__(self):
        super(PositionalEmbedding, self).__init__()
        self.embeddings = nn.ParameterDict()
        self.device = config.device
        # for eval
        eval_keys = [
            "pos0_1_ch64_s64", 
            "pos-1_0_ch64_s64", 
            "pos1_0_ch64_s64", 
            "pos0_-1_ch64_s64", 
            "pos0_1_ch128_s32", 
            "pos-1_0_ch128_s32", 
            "pos1_0_ch128_s32", 
            "pos0_-1_ch128_s32", 
            "pos0_1_ch256_s16", 
            "pos-1_0_ch256_s16", 
            "pos1_0_ch256_s16", 
            "pos0_-1_ch256_s16"
        ]
        for key in eval_keys:
            channel = int(key.split("ch")[1].split("_")[0])
            size = int(key.split("s")[-1])
            self.embeddings[key] = nn.Parameter(torch.randn(1, channel, size, size).to(self.device))

    def forward(self, feature_maps, pos):
        feature_outs = []
        for feature_index in range(len(feature_maps)):
            batch_size, channel, size, _ = feature_maps[feature_index].shape
            batch_embeddings = []
            for i in range(batch_size):
                key = f"pos{'_'.join(map(str, pos[i].int().tolist()))}_ch{channel}_s{size}"
                if key not in self.embeddings:
                    self.embeddings[key] = nn.Parameter(torch.randn(1, channel, size, size).to(self.device))
                batch_embeddings.append(self.embeddings[key])
            positional_encoding = torch.cat(batch_embeddings)
            feature_out = torch.einsum('bchw,bcwj->bchj', feature_maps[feature_index], positional_encoding)
            feature_outs.append(feature_out)
        return feature_outs
    
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.device = config.device
        self.encoder = Encoder(config.input_dim, config.features)
        self.decoder = Decoder(config.features, 2)
        self.embedding = PositionalEmbedding()
        self.attention = AttentionModule()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, directions = torch.randint(-1,1,(config.batch_size,2))):
        center_patch = x[:,0,:,:,:]
        guide_patchs = x[:,1:,:,:,:]

        score = 0
        center_features = self.encoder(center_patch)
        for patch_index in range(directions.shape[1]):
            guide_features = self.encoder(guide_patchs[:,patch_index,:,:,:])
            positional_guide_features = self.embedding(guide_features, directions[:,patch_index:,])
            features = [torch.cat((center_features[i], positional_guide_features[i]), dim=1) for i in range(len(config.features))]
            att_features = self.attention(features)
            output = self.decoder(att_features)
            output = self.softmax(output)
            score += output
        score = self.softmax(output)
        return score

        # center_patch = x[:,0,:,:,:]
        # center_features = self.encoder(center_patch)
        # output = self.decoder(center_features)
        # output = self.softmax(output)
        # return output

# 测试函数
if __name__ == "__main__":
    model = Backbone()
    summary(model, input_size=(2, config.batch_size, config.input_dim, config.image_size, config.image_size))
    print(model)