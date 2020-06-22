import math

import torch
from torch import nn

"""
![alt text](https://i.ibb.co/JQ9JL2t/image.png)

![alt text](https://i.ibb.co/Tcmfjjn/image.png)
"""

class ResidualBlock(nn.Module):
    """
    At the core of our very deep generator network G, which
    is illustrated in Figure 4 are B residual blocks with identical
    layout. Inspired by Johnson et al. [33] we employ the block
    layout proposed by Gross and Wilber [24]. Specifically, we
    use two convolutional layers with small 3×3 kernels and 64
    feature maps followed by batch-normalization layers [32]
    and ParametricReLU [28] as the activation function.

    k3n64s1 =>
        kernel_size = 3,
        channels = 64,
        stride = 1
    """

    def __init__(self, channels=64, kernel_size=3, stride=1):
        super().__init__()
        padding = 3 // 2
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.prelu = nn.PReLU()

        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        # Element-wise sum
        return residual + x

class SPCNBlock(nn.Module):
    """
    SPCN - sub-pixel convolutional neural network
    We increase the resolution of the input image with two trained
    sub-pixel convolution layers as proposed by Shi et al. [48].

    https://arxiv.org/pdf/1609.05158.pdf

    k3n256s1 =>
        kernel_size = 3,
        channels = 256, (64 * (2 ** 2))
        stride = 1
    """

    def __init__(self, in_channels,
                 upscale_factor=2,
                 kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=in_channels * (upscale_factor ** 2),
                              kernel_size=kernel_size,
                              padding=kernel_size // 2)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=upscale_factor)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)

        return x


class ConvolutionalBlock(nn.Module):
    """
    Convolutional block has convolutional, Batch Normalization and activation layers.
    It supports 3 types of activation: PReLU, LeakyReLU and Tanh
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, batch_norm=False, activation=None):
        super(ConvolutionalBlock, self).__init__()

        if activation is not None:
            activation = activation.lower()
            assert activation in {'prelu', 'leakyrelu', 'tanh'}

        layers = list()

        layers.append(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2))

        if batch_norm is True:
            layers.append(nn.BatchNorm2d(num_features=out_channels))

        if activation == 'prelu':
            layers.append(nn.PReLU())
        elif activation == 'leakyrelu':
            layers.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            layers.append(nn.Tanh())

        self.conv_block = nn.Sequential(*layers)

    def forward(self, input):
        output = self.conv_block(input)

        return output


class SRResNet(nn.Module):
    """
    The SRResNet, as defined in the paper.
    """

    def __init__(self, large_kernel_size=9, small_kernel_size=3, n_channels=64, n_blocks=16, scaling_factor=4):
        """
        :param large_kernel_size: kernel size of the first and last convolutions which transform the inputs and outputs
        :param small_kernel_size: kernel size of all convolutions in-between the residual and subpixel convolutional blocks
        :param n_channels: number of channels in-between i.e, the input and output channels for the residual and subpixel convolutional blocks
        :param n_blocks: number of residual blocks
        :param scaling_factor: factor to scale input images by (along both dimensions) in the subpixel convolutional block
        """
        super(SRResNet, self).__init__()

        self.conv_block1 = ConvolutionalBlock(in_channels=3, out_channels=n_channels, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='PReLu')

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels=n_channels) for i in range(n_blocks)])

        self.conv_block2 = ConvolutionalBlock(in_channels=n_channels, out_channels=n_channels,
                                              kernel_size=small_kernel_size,
                                              batch_norm=True, activation=None)

        n_SPCNblocks = int(math.log2(scaling_factor))
        self.SPCNblocks = nn.Sequential(
            *[SPCNBlock(in_channels=n_channels, upscale_factor=2, kernel_size=small_kernel_size) for i
              in range(n_SPCNblocks)])

        self.conv_block3 = ConvolutionalBlock(in_channels=n_channels, out_channels=3, kernel_size=large_kernel_size,
                                              batch_norm=False, activation='Tanh')

    def forward(self, lr_imgs):
        output = self.conv_block1(lr_imgs)
        residual = output 
        output = self.residual_blocks(output)  
        output = self.conv_block2(output) 
        output = output + residual 
        output = self.SPCNblocks(output)  
        sr_imgs = self.conv_block3(output)  
        return sr_imgs
