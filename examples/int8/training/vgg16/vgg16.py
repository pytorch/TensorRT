"""
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556) (ICLR 2015)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce


class VGG(nn.Module):
    def __init__(self, layer_spec, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()

        layers = []
        in_channels = 3
        for l in layer_spec:
            if l == "pool":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [
                    nn.Conv2d(in_channels, l, kernel_size=3, padding=1),
                    nn.BatchNorm2d(l),
                    nn.ReLU(),
                ]
                in_channels = l

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def vgg16(num_classes=1000, init_weights=False):
    vgg16_cfg = [
        64,
        64,
        "pool",
        128,
        128,
        "pool",
        256,
        256,
        256,
        "pool",
        512,
        512,
        512,
        "pool",
        512,
        512,
        512,
        "pool",
    ]
    return VGG(vgg16_cfg, num_classes, init_weights)
