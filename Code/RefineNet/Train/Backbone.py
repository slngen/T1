'''
Author: CT
Date: 2023-11-26 09:43
LastEditors: CT
LastEditTime: 2023-11-26 09:59
'''
import torch.nn.functional as F
import torch.nn as nn
import torch

from Config import config
 
class BasicBlock(nn.Module):
    expansion: int = 4
    def __init__(self, inplanes, planes, stride = 1, downsample = None, groups = 1,
        base_width = 64, dilation = 1, norm_layer = None):
        
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, planes ,kernel_size=3, stride=stride, 
                               padding=dilation,groups=groups, bias=False,dilation=dilation)
        
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes ,kernel_size=3, stride=stride, 
                               padding=dilation,groups=groups, bias=False,dilation=dilation)
        
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample= None,
        groups = 1, base_width = 64, dilation = 1, norm_layer = None,):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, bias=False, padding=dilation, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
        return out
 
 
class ResNet(nn.Module):
    def __init__(
        self,block, layers,num_classes = 1000, zero_init_residual = False, groups = 1,
        width_per_group = 64, replace_stride_with_dilation = None, norm_layer = None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
            
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(6, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
 
    def _make_layer(
        self,
        block,
        planes,
        blocks,
        stride = 1,
        dilate = False,
    ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = stride
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,  planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion))
 
        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )
        return nn.Sequential(*layers)
 
    def _forward_impl(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)
        return outs
 
    def forward(self, x) :
        return self._forward_impl(x)
    def _resnet(block, layers, pretrained_path = None, **kwargs,):
        model = ResNet(block, layers, **kwargs)
        if pretrained_path is not None:
            model.load_state_dict(torch.load(pretrained_path),  strict=False)
        return model
    
    def resnet50(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 6, 3],pretrained_path,**kwargs)
    
    def resnet101(pretrained_path=None, **kwargs):
        return ResNet._resnet(Bottleneck, [3, 4, 23, 3],pretrained_path,**kwargs)
 
class ResidualConvUnit(nn.ModuleList):
    def __init__(self, in_channels):
        super(ResidualConvUnit, self).__init__()
        for i in range(4):
            self.append(
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(in_channels//(2**(3 - i)) , in_channels//(2**(3 - i)) , 3, padding=1, bias=False),
                    nn.BatchNorm2d(in_channels//(2**(3 - i)) ),
                    nn.ReLU(),
                    nn.Conv2d(in_channels//(2**(3 - i)) , in_channels//(2**(3 - i)) , 3, padding=1, bias=False),
                )
            )  
        
        
    def forward(self, x):
        outs = []   
        for index, module in enumerate(self):
            x1 = module(x[index])
            x1 = x[index] + x1
            outs.append(x1)
        return outs
    
def un_pool(input, scale):
    return F.interpolate(input, scale_factor=scale, mode='bilinear', align_corners=True)
    
class MultiResolutionFusion(nn.ModuleList):
    def __init__(self, in_channels, out_channels, scale_factors = [1,2,4,8]):
        super(MultiResolutionFusion, self).__init__()
        self.scale_factors = scale_factors
        
        for index, scale in enumerate(scale_factors):
            self.append(
                nn.Sequential(
                    nn.Conv2d(in_channels//2** (len(scale_factors)-index-1), out_channels, kernel_size=3, padding=1)
                    )
            )
 
    def forward(self, x):
        outputs = []
        for index, module in enumerate(self):
            xi = module(x[index])
            xi = un_pool(xi, scale=self.scale_factors[index])
            outputs.append(xi)
        return outputs[0] + outputs[1] + outputs[2] + outputs[3]    
 
    
class ChainedResidualPool(nn.ModuleList):
    def __init__(self, in_channels, blocks=4):
        super(ChainedResidualPool, self).__init__()
        self.in_channels = in_channels
        self.blocks = blocks
        self.relu = nn.ReLU()
        for i in range(blocks):
            self.append(
                nn.Sequential(
                    nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                    nn.Conv2d(in_channels, in_channels,kernel_size=3, padding=1, stride=1, bias=False),
                )
            )
        
    def forward(self, x):
        x = self.relu(x)
        path = x
        for index, CRP in enumerate(self):
            path = CRP(path)
            x = x + path  
        return x
        
class Backbone(nn.Module):
    def __init__(self, num_classes=2):
        super(Backbone, self).__init__()
        self.backbone = ResNet.resnet50()
        self.final = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=3, padding=1, bias=False),
        )
        self.ResidualConvUnit = ResidualConvUnit(2048)
        self.MultiResolutionFusion = MultiResolutionFusion(2048, 256)
        self.ChainedResidualPool = ChainedResidualPool(256)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.ResidualConvUnit(x)
        x = self.MultiResolutionFusion(x)
        x = self.ChainedResidualPool(x)
        x = un_pool(x, scale=4)
        x =  self.final(x)
        x = self.softmax(x)
        return x
    
if __name__ == "__main__":
    from torchinfo import summary
    
    model = Backbone()
    summary(model, input_size=(config.batch_size, config.input_dim, config.image_size, config.image_size))
    # print(model)