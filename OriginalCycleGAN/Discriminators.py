import torch.nn as nn
import torch


def _discriminator_layer(in_feat, out_feat, ksize=4, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_feat, out_channels=out_feat, kernel_size=ksize, stride=stride),
        nn.BatchNorm2d(out_feat),
        nn.LeakyReLU(0.2)
    )


def _generate_layer_list(n):
    conv_list = []
    for i in range(n):
        conv_list.append(_discriminator_layer(64 * 2 ** i, 64 * 2 ** (i + 1), 2))
    return nn.ModuleList(conv_list)


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2)
        self.leaky = nn.LeakyReLU(0.2)
        self.convList = _generate_layer_list(3)
        self.last_conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky(self.conv1(x))
        for conv in self.convList:
            x = conv(x)
        x = self.last_conv(x)
        x = self.sigmoid(x)
        #x = x.view(x.shape[0], -1)
        return x #torch.mean(x, 1)


class PixelDiscriminator(nn.Module):
    def __init__(self):
        super(PixelDiscriminator, self).__init__()
        layers = [_discriminator_layer(3, 64, ksize=1),
        _discriminator_layer(64, 128, ksize=1),
        _discriminator_layer(128, 1, ksize=1),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)