import torch
from torch import nn

from src.models.caunet import Caunet
from src.models.seanet import Seanet
from src.models.seanet_lstm import SeanetLSTM
from src.models.demucs import Demucs
from src.models.transformer import Transformer
from src.utils import capture_init
from src.models.modules import ResnetBlock, PixelShuffle1D, WNConv1d

import math
from treelib import Tree
import logging

logger = logging.getLogger(__name__)

class SpNode(nn.Module):

    def __init__(self, parent = None):
        super().__init__()
        self.parent = parent
        self.sub_pixel_layer = PixelShuffle1D(upscale_factor=2)

    def forward(self, signals):
        """
           :param low_pass_signal: low sample rate signal [B,C,T]
           :param high_pass_signal: low sample rate signal [B,C,T]
           :return: high sample rate signal [B,C,upscale_factor*T]
       """
        out = torch.cat(signals, dim=1)  # [B,upscale_factor*C,T]
        out = self.sub_pixel_layer(out)  # [B,C,upscale_factor*T]
        return out


def create_sub_tree(tree, parent, level):
    if level == 0:
        return
    left = tree.create_node(tag=f'{level}-left', data=SpNode(parent.data), parent=parent)
    right = tree.create_node(tag=f'{level}-right', data=SpNode(parent.data), parent=parent)
    create_sub_tree(tree, left, level-1)
    create_sub_tree(tree, right, level-1)

class SpCombTree(nn.Module):

    def __init__(self, upscale_factor):
        self.upscale_factor = upscale_factor
        self.tree = Tree()
        root = self.tree.create_node(tag='root', data=SpNode())

        depth = math.log2(upscale_factor)-1
        create_sub_tree(self.tree, root, depth)

        self.tree.show()



    def forward(self, signals):
        it = iter(signals)
        for left_input, right_input in zip(it,it):
            pass


class DividedSpCombModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.l_comb_child = PixelShuffle1D(upscale_factor=2)
        self.r_comb_child = PixelShuffle1D(upscale_factor=2)
        self.comb_root = PixelShuffle1D(upscale_factor=2)

    def forward(self, signals):
        """

        :param signals: a list of 4 tensors of [B,C,T]
        :return: high sample rate signal of [B,C,4*T]
        """
        left = torch.cat(signals[:2], dim=1)  # [B,2*C,T]
        right = torch.cat(signals[2:], dim=1)  # [B,2*C,T]
        left = self.l_comb_child(left)  # [B,C,2*T]
        right = self.r_comb_child(right)  # [B,C,2*T]
        out = torch.cat([left, right], dim=1)  # [B,2*C,2*T]
        out = self.comb_root(out)  # [B,C,4*T]
        return out


class SpCombModule(nn.Module):

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.sub_pixel_layer = PixelShuffle1D(upscale_factor=upscale_factor)

    def forward(self, signals):
        """

        :param low_pass_signal: low sample rate signal [B,C,T]
        :param high_pass_signal: low sample rate signal [B,C,T]
        :return: high sample rate signal [B,C,upscale_factor*T]
        """
        full_band_signal = torch.cat(signals, dim=1)  # [B,upscale_factor*C,T]
        full_band_signal = self.sub_pixel_layer(full_band_signal)  # [B,C,upscale_factor*T]
        return full_band_signal


class ConditionalBias(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, signal):
        """

        :param signal: [B,C,T]
        :return:
        """
        bias = signal.permute(0, 2, 1)
        bias = self.linear(bias)
        bias = bias.permute(0, 2, 1)
        return bias


class LowPassModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conditional_bias = ConditionalBias(in_channels, out_channels)

        low_pass_wrapper_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(out_channels, out_channels, kernel_size=7, padding=0)
        ]
        self.low_pass_module = nn.Sequential(*low_pass_wrapper_layer)

    def forward(self, signal):
        bias = self.conditional_bias(signal)
        return self.low_pass_module(signal + bias)


class HighPassModule(nn.Module):

    def __init__(self, in_channels, depth, residual_layers, hidden_channels, out_channels):
        super().__init__()
        high_pass_module_list = []

        high_pass_wrapper_layer_1 = [
            nn.ReflectionPad1d(3),
            WNConv1d(in_channels, hidden_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        high_pass_module_list += high_pass_wrapper_layer_1

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
            high_pass_module_list += conv_block

        high_pass_wrapper_layer_2 = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(hidden_channels, out_channels, kernel_size=7, padding=0)
        ]
        high_pass_module_list += high_pass_wrapper_layer_2

        self.high_pass_module = nn.Sequential(*high_pass_module_list)

    def forward(self, signal, hr_shape):
        return self.high_pass_module(signal)


def get_high_pass_module(args):
    if 'high_pass_module' in args.experiment.interponet and args.experiment.interponet.high_pass_module == 'seanet':
        high_pass_module = Seanet(**args.experiment.seanet)
    elif 'high_pass_module' in args.experiment.interponet and args.experiment.interponet.high_pass_module == 'seanet_lstm':
        high_pass_module = SeanetLSTM(**args.experiment.seanet_lstm)
    elif 'high_pass_module' in args.experiment.interponet and args.experiment.interponet.high_pass_module == 'caunet':
        high_pass_module = Caunet(**args.experiment.caunet)
    elif 'high_pass_module' in args.experiment.interponet and args.experiment.interponet.high_pass_module == 'demucs':
        high_pass_module = Demucs(**args.experiment.demucs)
    elif 'high_pass_module' in args.experiment.interponet and args.experiment.interponet.high_pass_module == 'transformer':
        high_pass_module = Transformer(**args.experiment.transformer)
    else:
        in_channels = 1 if args.experiment.interponet.parallel else args.experiment.interponet.out_channels
        high_pass_module = HighPassModule(in_channels,
                                          args.experiment.interponet.depth,
                                          args.experiment.interponet.residual_layers,
                                          args.experiment.interponet.hidden_channels,
                                          args.experiment.interponet.out_channels)
    logger.info(f'high pass module class name: {type(high_pass_module).__name__}')
    return high_pass_module

def get_high_pass_input(args,signal):
    if 'high_pass_input' in args.experiment.interponet and args.experiment.interponet.high_pass_input == 'noise':
        high_pass_input = torch.rand_like(signal)
    else:
        high_pass_input = signal
    return high_pass_input


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


class Interponet(nn.Module):

    @capture_init
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.out_channels = args.experiment.interponet.out_channels
        self.lr_sr = args.experiment.interponet.lr_sr
        self.hr_sr = args.experiment.interponet.hr_sr
        self.scale_factor = self.hr_sr/self.lr_sr
        self.parallel = args.experiment.interponet.parallel

        self.low_pass_module = LowPassModule(1, args.experiment.interponet.out_channels)

        self.high_pass_modules = nn.ModuleList()
        for i in range(1, int(self.scale_factor)):
            self.high_pass_modules.append(get_high_pass_module(args))

        if self.scale_factor == 4 and self.args.experiment.interponet.divide_comb:
            self.comb_module = DividedSpCombModule()
        else:
            self.comb_module = SpCombModule(upscale_factor=int(self.scale_factor))

        self.fine_tune_module = FineTuneModule(**args.experiment.interponet.fine_tune_module)

    def estimate_output_length(self, input_length):
        return input_length * self.scale_factor

    def forward(self, signal, hr_shape):
        low_pass_output = self.low_pass_module(signal)
        high_pass_input = get_high_pass_input(self.args, signal) if self.parallel else low_pass_output
        high_pass_outputs = []
        for i in range(len(self.high_pass_modules)):
            high_pass_output = self.high_pass_modules[i](high_pass_input, high_pass_input.shape[-1])
            high_pass_outputs.append(high_pass_output)
        full_band_output = self.comb_module([low_pass_output] + high_pass_outputs)
        full_band_output = self.fine_tune_module(full_band_output)

        return {'full_band': full_band_output,
                'low_pass': low_pass_output,
                'high_pass': high_pass_outputs}