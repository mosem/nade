import torch
from torch import nn

from src.utils import capture_init
from src.models.modules import ResnetBlock, PixelShuffle1D, WNConv1d
from src.models.seanet import Seanet

class FineTuneModule(nn.Module):

    def __init__(self, depth, residual_layers, in_channels, hidden_channels, out_channels):
        super().__init__()
        fine_tune_module_list = []

        fine_tune_wrapper_layer_1 = [
            nn.ReflectionPad1d(3),
            WNConv1d(in_channels, hidden_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        fine_tune_module_list += fine_tune_wrapper_layer_1

        for i in range(depth):
            conv_block = [
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3),
                WNConv1d(hidden_channels,
                         hidden_channels,
                         kernel_size=7),
            ]
            for j in range(residual_layers):
                conv_block += [ResnetBlock(hidden_channels, dilation=3 ** j)]
            fine_tune_module_list += conv_block

        if depth > 0:
            fine_tune_wrapper_layer_2 = [
                nn.LeakyReLU(0.2),
                nn.ReflectionPad1d(3),
                WNConv1d(hidden_channels, out_channels, kernel_size=7, padding=0)
            ]
            fine_tune_module_list += fine_tune_wrapper_layer_2

        self.fine_tune_module = nn.Sequential(*fine_tune_module_list)

    def forward(self, signal):
        return self.fine_tune_module(signal)


class SeanetFt(nn.Module):

    @capture_init
    def __init__(self, seanet_ft_args):
        super().__init__()
        self.seanet_module = Seanet(**seanet_ft_args.seanet)
        self.fine_tune_module = FineTuneModule(**seanet_ft_args.fine_tune)

    def forward(self, x, hr_len):
        out = self.seanet_module(x)
        out = self.fine_tune_module(out)
        return out