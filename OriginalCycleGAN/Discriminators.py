import torch.nn as nn
import torchvision


def _discriminator_layer(in_feat, out_feat, ksize=4, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_feat, out_channels=out_feat, kernel_size=ksize, stride=stride),
        nn.BatchNorm2d(out_feat),
        nn.LeakyReLU(0.2)
    )


def convolutionalEncoder(layers):
    conv_list = []
    for i in range(layers):
        conv_list.append(_discriminator_layer(64 * 2 ** i, 64 * 2 ** (i + 1), 2))
    return nn.Sequential(*conv_list)


class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2), nn.LeakyReLU(0.2))
        inbetween_conv = convolutionalEncoder(3)
        last_conv = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2)
        sigmoid = nn.Sigmoid()
        layers = [conv1, inbetween_conv, last_conv, sigmoid]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


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


class PerceptualDiscriminator(nn.Module):
    def __init__(self):
        super(PerceptualDiscriminator, self).__init__()
        
        layers = [_discriminator_layer(3, 64, ksize=1),
        _discriminator_layer(64, 128, ksize=1),
        _discriminator_layer(128, 1, ksize=1),
            nn.Sigmoid()
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)