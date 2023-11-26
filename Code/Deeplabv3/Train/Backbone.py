import torch.nn as nn
from torchvision import models

from Config import config

class Backbone(nn.Module):
    def __init__(self, num_classes=2, in_channels=6):
        super(Backbone, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(weights=None, num_classes=num_classes)
        self.model.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)["out"]

if __name__ == "__main__":
    from torchinfo import summary

    model = Backbone()
    summary(model, input_size=(config.batch_size, config.input_dim, config.image_size, config.image_size))
    # print(model)