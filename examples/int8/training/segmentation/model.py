import torch
import torch.nn as nn


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


def add_conv(in_channels, out_channels, nums=2):
    layers = []
    for i in range(nums):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        in_channels = out_channels

    return nn.Sequential(*layers)


# [64, 64, 128, 128, 256, 256, 256, 512, 512, 512]
class VGG16Unet(nn.Module):
    def __init__(
        self,
        pretrained=False,
        in_channels=3,
        out_channels=150,
        features=[(64, 2), (128, 2), (256, 3), (512, 3)],
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_connections = []
        self.features = features

        # For downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        for (feature, count) in features:
            self.downs.append(add_conv(in_channels, feature, count))
            in_channels = feature

        for (feature, count) in reversed(features):
            self.ups.append(up_conv(feature * 2, feature))
            self.ups.append(add_conv(feature * 2, feature, count))

        self.bottleneck = add_conv(features[-1][0], features[-1][0] * 2, 2)
        self.final_conv = nn.Conv2d(features[0][0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for mod in self.downs:
            x = mod(x)
            skip_connections.append(x)
            x = self.maxpool(x)

        x = self.bottleneck(x)

        # Reverse the skip connections ordering
        skip_connections = skip_connections[::-1]

        # Second part of the model architecture
        k = 0
        for idx, ups in enumerate(self.ups):
            x = ups(x)
            if idx == 0 or idx == 2 or idx == 4 or idx == 6:
                skip_conn = skip_connections[k]
                k += 1
                x = torch.cat((skip_conn, x), dim=1)

        x = self.final_conv(x)
        return x
