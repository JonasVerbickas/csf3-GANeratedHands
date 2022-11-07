import torch.nn as nn
import torchvision


class Conv2dTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        resnet50 = torchvision.models.resnet50(pretrained=True)
        for param in resnet50.parameters():
            param.requires_grad = False
        self.res_blocks = nn.Sequential(*list(resnet50.children())[:6])
        self.res_blocks.requires_grad = False
        self.upsample_blocks = nn.Sequential(
            Conv2dTransposeBlock(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2dTransposeBlock(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            Conv2dTransposeBlock(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1))
        self.last = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.res_blocks(x)
        x = self.upsample_blocks(x)
        x = self.last(x)
        x = self.sigmoid(x)
        return x
