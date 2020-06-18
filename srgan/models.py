import math

import torch
from torch import nn

"""
![alt text](https://i.ibb.co/JQ9JL2t/image.png)

![alt text](https://i.ibb.co/Tcmfjjn/image.png)
"""


class Generator(nn.Module):
    """
    """

    def __init__(self, upscale_factor=4, image_channels=3,
                 residual_block_channels=64,
                 num_residual_blocks=5):
        super().__init__()
        self.upscale_factor = upscale_factor
        self.image_channels = image_channels
        self.residual_block_channels = residual_block_channels
        self.num_residual_blocks = num_residual_blocks

        # k9n64s1
        initial_block_param = {
            "in_channels": image_channels,
            "kernel_size": 9,
            "out_channels": residual_block_channels,
            "stride": 1,
            "padding": 9 // 2
        }
        self.initial_block = nn.Sequential(
            nn.Conv2d(**initial_block_param),
            nn.PReLU()
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels=residual_block_channels)
              for _ in range(num_residual_blocks)]
        )

        self.skip_block = SkipBlock(channels=residual_block_channels)

        # two trained sub-pixel convolution layers
        num_spcn_blocks = int(math.log(upscale_factor, 2))
        self.spcn_blocks = nn.Sequential(
            *[SPCNBlock(in_channels=residual_block_channels,
                        upscale_factor=2)
              for _ in range(num_spcn_blocks)]
        )

        #  k9n3s1
        final_block_param = {
            "in_channels": residual_block_channels,
            "kernel_size": 9,
            "out_channels": image_channels,
            "stride": 1,
            "padding": 9 // 2
        }
        self.final_block = nn.Conv2d(**final_block_param)

    def forward(self, x):
        initial_out = self.initial_block(x)
        B_residual_out = self.residual_blocks(initial_out)
        skip_out = self.skip_block(B_residual_out,
                                   initial_out)
        spcn_out = self.spcn_blocks(skip_out)
        pixels = self.final_block(spcn_out)
        # tanh to squish => [-1, 1]
        out = torch.tanh(pixels)

        return out


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


class SkipBlock(nn.Module):
    """
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

    def forward(self, x, img):
        residual = self.conv1(x)
        residual = self.bn1(residual)

        # Element-wise sum
        return residual + img


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


class Discriminator(nn.Module):
    """
    We follow the architectural guidelines summarized by Radford et al. [44]
    and use LeakyReLU activation (α = 0.2) and avoid max-pooling throughout
    the discriminator network. It contains eight convolutional layers with
    an increasing number of 3 × 3 filter kernels, increasing by a factor of
    2 from 64 to 512 kernels as in the VGG network [49].
    Strided convolutions are used to reduce the image resolution each time
    the number of features is doubled. The resulting 512 feature maps are
    followed by two dense layers and a final sigmoid activation
    function to obtain a probability for sample classification.
    """

    def __init__(self, image_channels=3, num_middle_blocks=7):
        super().__init__()
        self.kernel_size = 3
        self.image_channels = image_channels
        self.num_middle_blocks = num_middle_blocks
        # k3n64s1
        initial_block_param = {
            "in_channels": image_channels,
            "kernel_size": self.kernel_size,
            "out_channels": 64,
            "stride": 1,
            "padding": self.kernel_size // 2
        }

        self.initial_block = nn.Sequential(
            nn.Conv2d(**initial_block_param),
            nn.LeakyReLU(0.2)
        )
        # 7 middle blocks from 64 channels to 512 channnels
        in_channels = 64
        out_channels = None

        middle_blocks = []
        for i in range(num_middle_blocks):
            if i % 2 == 0:
                stride = 2
                out_channels = in_channels
            else:
                stride = 1
                out_channels = in_channels * 2
            middle_blocks.append(
                MiddleBlock(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=self.kernel_size,
                            stride=stride))
            in_channels = out_channels

        self.middle_blocks = nn.Sequential(*middle_blocks)

        # final linear layers
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        self.linear1 = nn.Linear(out_channels * 6 * 6, 1024)
        self.leakyRelu = nn.LeakyReLU(0.2)
        self.linear2 = nn.Linear(1024, 1)

    def forward(self, x):
        batch_size = x.size(0)
        init_out = self.initial_block(x)
        mid_out = self.middle_blocks(init_out)
        out = self.adaptive_pool(mid_out)
        out = self.linear1(out.view(batch_size, -1))
        out = self.leakyRelu(out)
        out = self.linear2(out)
        return out


class MiddleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               padding=padding,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.leakyRelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.leakyRelu(output)
        return output
