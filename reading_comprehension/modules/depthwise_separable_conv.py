import collections
import torch
from allennlp.nn.activations import Activation


class DepthwiseSeparableConv(torch.nn.Module):
    """
    Depthwise Separable Convolution described in
    `Xception: Deep learning with depthwise separable convolutions <http://arxiv.org/abs/1610.02357>`_ .

    This module Performs a depthwise convolution that acts separately on channels
    followed by a pointwise convolution that mixes channels.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 activation: str = 'relu',
                 dim: int = 1,
                 bias: bool = True) -> None:
        super().__init__()
        if dim == 1:
            padding_left = kernel_size // 2
            padding_right = padding_left if kernel_size % 2 != 0 else padding_left - 1
            self.depthwise_conv = torch.nn.Sequential(
                    torch.nn.ReflectionPad1d((padding_left, padding_right)),
                    torch.nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                    kernel_size=kernel_size, groups=in_channels, bias=bias))
            self.pointwise_conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=1, bias=bias)
        elif dim == 2:
            if isinstance(kernel_size, collections.Iterable):
                kernel_1, kernel_2 = kernel_size
                padding_1_left = kernel_1 // 2
                padding_1_right = padding_1_left if kernel_1 % 2 != 0 else padding_1_left - 1
                padding_2_left = kernel_2 // 2
                padding_2_right = padding_2_left if kernel_2 % 2 != 0 else padding_2_left - 1
                padding = (padding_1_left, padding_1_right, padding_2_left, padding_2_right)
            else:
                padding_left = kernel_size // 2
                padding_right = padding_left if kernel_size % 2 != 0 else padding_left - 1
                padding = (padding_left, padding_right, padding_left, padding_right)
            self.depthwise_conv = torch.nn.Sequential(
                    torch.nn.ReflectionPad2d(padding),
                    torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                    kernel_size=kernel_size, groups=in_channels, bias=bias))
            self.pointwise_conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=1, bias=bias)
        else:
            raise Exception(f"We currently only handle 1 and 2 dimensional convolutions here. You gave {dim}.")
        if activation is not None:
            self._activation = Activation.by_name(activation)()
        else:
            self._activation = Activation.by_name("linear")()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # pylint: disable=arguments-differ
        return self._activation(self.pointwise_conv(self.depthwise_conv(x)))
