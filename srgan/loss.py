import torchvision
from torch import nn


class VGG19Loss(nn.Module):
    """
    A truncated VGG19 network, such that its output is the
    'feature map obtained by the j-th convolution (after activation)
    before the i-th maxpooling layer within the VGG19 network',
    as defined in the paper.
    Used to calculate the MSE loss in this VGG feature-space,
    i.e. the VGG loss.
    """

    def __init__(self, i, j):
        """"""
        super().__init__()

        # pre-trained VGG19
        vgg19 = torchvision.models.vgg19(pretrained=True)
        count_i = 0
        count_j = 0
        layers = []
        idx = 0

        for idx, layer in enumerate(vgg19.features):
            if isinstance(layer, nn.MaxPool2d):
                count_i += 1
                count_j = 0
            elif isinstance(layer, nn.Conv2d):
                count_j += 1

            layers.append(layer)
            # before the i-th maxpooling layer
            if count_j == j and count_i == (i - 1):
                break
        # (after activation)
        layers.append(vgg19.features[idx + 1])
        self.vgg_layers = nn.Sequential(*layers)

    def forward(self, input):
        """
        Forward propagation
        :param input: high-resolution or super-resolution images,
        a tensor of size (N, 3, w * scaling factor, h * scaling factor)
        :return: the specified VGG19 feature map,
        a tensor of size (N, feature_map_channels, feature_map_w, feature_map_h)
        """
        output = self.vgg_layers(input)

        return output
