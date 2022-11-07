import torch.nn as nn
import torchvision
import functools

"""
My implementation
"""
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
        #for param in resnet50.parameters():
        #    param.requires_grad = False
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


"""
Other implementation
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, padding=0,
                 norm_layer=None, act_layer=None):
        super(ConvBlock, self).__init__()

        layer_list = [nn.Conv2d(in_channels, out_channels, kernel_size=ksize,
                                stride=stride, padding=padding)]

        if norm_layer:
            layer_list.append(norm_layer(out_channels))

        if act_layer:
            layer_list.append(act_layer)

        self.module = nn.Sequential(*layer_list)

    def forward(self, inp):
        """ Simple forward function """
        return self.module(inp)


class TransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, padding=0,
                 output_padding=0, norm_layer=None, act_layer=None):
        super(TransConvBlock, self).__init__()

        layer_list = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=ksize,
                                         stride=stride, padding=padding,
                                         output_padding=output_padding)]
        if norm_layer:
            layer_list.append(norm_layer(out_channels))

        if act_layer:
            layer_list.append(act_layer)

        self.module = nn.Sequential(*layer_list)

    def forward(self, inp):
        """ Simple forward function """
        return self.module(inp)


class ResidualBlock(nn.Module):
    """
                        Defines a ResnetBlock
    X ------------------------identity------------------------
    |-- Convolution -- Norm -- ReLU -- Convolution -- Norm --|

    Parameters:
        in_ch: Number of input channels
        norm_layer: Batch or Instance normalization
        use_dropout: If set to True, activations will be 0'ed with a probability of 0.5 by default
    """

    def __init__(self, in_ch, norm_layer=None, use_dropout=False):
        super(ResidualBlock, self).__init__()
        layer_list = [nn.ReflectionPad2d(1),
                      ConvBlock(in_ch, in_ch, ksize=3, stride=1, norm_layer=norm_layer,
                                act_layer=nn.ReLU(True))]

        if use_dropout:
            layer_list += [nn.Dropout(0.5)]

        layer_list += [nn.ReflectionPad2d(1),
                       ConvBlock(in_ch, in_ch, ksize=3, stride=1, norm_layer=norm_layer)]

        self.module = nn.Sequential(*layer_list)

    def forward(self, inp):
        """ Forward function with skip connections """
        return inp + self.module(inp)


class Resnet(nn.Module):
    """Resnet-based generator consisting of a downsampling step (7x7 conv to encode color features
    + two 3x3 2-strided convs for the downsampling), Resnet blocks in between, and the upsampling
    step (two 3x3 2-strided transposed convs + one 7x7 final conv)"""
    def __init__(self, deconvolution=True, n_blocks=6):
        """Construct a Resnet-based generator
        Parameters:
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
        """
        super(Resnet, self).__init__()

        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)

        class hparams:
            inp_ch = 3
            ngf = 64
            use_dropout = True


        # First 7x7 convolution to encode color information + 2 downsampling steps
        model = [nn.ReflectionPad2d(3),
                 ConvBlock(hparams.inp_ch, hparams.ngf * 1, ksize=7, stride=1, padding=0,
                           norm_layer=norm_layer, act_layer=nn.ReLU(True)),
                 ConvBlock(hparams.ngf * 1, hparams.ngf * 2, ksize=3, stride=2, padding=1,
                           norm_layer=norm_layer, act_layer=nn.ReLU(True)),
                 ConvBlock(hparams.ngf * 2, hparams.ngf * 4, ksize=3, stride=2, padding=1,
                           norm_layer=norm_layer, act_layer=nn.ReLU(True))]

        # ResNet blocks
        for _ in range(n_blocks):
            model += [ResidualBlock(hparams.ngf * 4, norm_layer=norm_layer,
                                    use_dropout=hparams.use_dropout)]

        # Upsampling phase
        if deconvolution:
            model += [TransConvBlock(hparams.ngf * 4, hparams.ngf * 2, ksize=3, stride=2, padding=1,
                                     output_padding=1, norm_layer=norm_layer,
                                     act_layer=nn.ReLU(True)),
                      TransConvBlock(hparams.ngf * 2, hparams.ngf * 1, ksize=3, stride=2, padding=1,
                                     output_padding=1, norm_layer=norm_layer,
                                     act_layer=nn.ReLU(True))]
        else:
            model += [nn.Upsample(scale_factor=2, mode='bilinear'),
                      nn.ReflectionPad2d(1),
                      ConvBlock(hparams.ngf * 4, hparams.ngf * 2, ksize=3, norm_layer=norm_layer,
                                act_layer=nn.ReLU(True))]
            model += [nn.Upsample(scale_factor=2, mode='bilinear'),
                      nn.ReflectionPad2d(1),
                      ConvBlock(hparams.ngf * 2, hparams.ngf * 1, ksize=3, norm_layer=norm_layer,
                                act_layer=nn.ReLU(True))]

        model += [nn.ReflectionPad2d(3),
                  ConvBlock(hparams.ngf, 3, ksize=7, padding=0, act_layer=nn.Tanh())]

        self.model = nn.Sequential(*model)

    def forward(self, inp):
        """ Standard forward """
        return self.model(inp)