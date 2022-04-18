import torch
from torch import nn
from torchaudio.functional import resample
import math
import logging

logger = logging.getLogger(__name__)

from src.utils import capture_init
from src.models.modules import ResnetBlock, PixelShuffle1D, WNConv1d
from src.models.seanet import Seanet
from src.models.modules import BLSTM

class SpCombModule(nn.Module):

    def __init__(self, upscale_factor:int=2):
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

    def __init__(self, in_channels, out_channels, lr_sr, hr_sr):
        super().__init__()
        self.conditional_bias = ConditionalBias(in_channels, out_channels)
        self.lr_sr = lr_sr
        self.hr_sr = hr_sr


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

    def __init__(self, in_channels, hidden_channels, out_channels, depth, residual_layers):
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

    def forward(self, signal, hr_len=None):
        return self.high_pass_module(signal)


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

class UpscaleBlock(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, hp_depth, hp_n_res_layers,
                 lr_sr, hr_sr, upsample,
                 hp_input, load_hp_path=None, hp_name=None, hp_args=None, freeze_params=None,
                 use_lp_module=True,
                 ft_name=None, ft_args=None):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.hp_depth = hp_depth
        self.hp_n_res_layers = hp_n_res_layers
        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.upsample = upsample
        self.hp_input = hp_input
        self.load_hp_path = load_hp_path
        self.hp_name = hp_name
        self.hp_args = hp_args
        self.freeze_params = freeze_params
        self.use_lp_module = use_lp_module
        self.ft_name = ft_name
        self.ft_args = ft_args
        self.hp_in_channels = in_channels if self.hp_input == 'parallel' else self.hidden_channels

        self.low_pass_module = self.get_low_pass_module()
        self.high_pass_module = self.get_high_pass_module()

        self.comb_module = SpCombModule()

        self.fine_tune_module = self.get_fine_tune_module()


    def wrap_seanet_module(self, seanet_args):
        high_pass_module_list = []

        hp_module = Seanet(**seanet_args)
        if self.load_hp_path:
            logger.info(f'loading Seanet from: {self.load_hp_path}')
            package = torch.load(self.load_hp_path, 'cpu')
            hp_module.load_state_dict(package['models']['generator']['state'])
        if self.freeze_params:
            logger.info(f'freezing Seanet params')
            for param in hp_module.parameters():
                param.requires_grad = False

        high_pass_module_list.append(hp_module)

        wrapper_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(seanet_args.out_channels, self.hidden_channels, kernel_size=7, padding=0)
        ]

        high_pass_module_list.append(nn.Sequential(*wrapper_layer))

        return nn.Sequential(*high_pass_module_list)


    def get_low_pass_module(self):
        if self.use_lp_module:
            low_pass_module = LowPassModule(self.in_channels, self.hidden_channels, self.lr_sr, self.hr_sr)
        else:
            low_pass_module = WNConv1d(self.in_channels, self.hidden_channels, kernel_size=1, padding=0)
        return low_pass_module


    def get_high_pass_module(self):
        if self.hp_name == 'seanet':
            logger.info(f'hp module: {self.hp_name}')
            high_pass_module = self.wrap_seanet_module(self.hp_args)
        else:
            logger.info(f'hp module: vanilla')
            high_pass_module = HighPassModule(self.hp_in_channels, self.hidden_channels, self.hidden_channels,
                                              self.hp_depth, self.hp_n_res_layers)
        return high_pass_module


    def get_fine_tune_module(self):
        logger.info(f'ft name: {self.ft_name}')
        if self.ft_name == 'lstm':
            ft_module = nn.Sequential(
                    BLSTM(**self.ft_args),
                    WNConv1d(self.hidden_channels, 1, kernel_size=1, padding=0)
                )

        else:
            ft_module = FineTuneModule(**self.ft_args)
        return ft_module


    def forward(self, signal, hr_len):
        if self.upsample:
            signal = resample(signal, self.lr_sr, self.hr_sr)
        lp_out = self.low_pass_module(signal)
        if self.hp_input == 'parallel':
            hp_out = self.high_pass_module(signal)
        else: # sequential
            hp_out = self.high_pass_module(lp_out)
        if hr_len and hr_len < hp_out.shape[-1]:
            hp_out = hp_out[..., :hr_len]
        # print(f'hr_len: {hr_len}')
        # print(f'hp shape: {hp_out.shape}')
        # print(f'lp shape: {lp_out.shape}')
        if self.upsample:
            full_band_out = self.comb_module([lp_out[..., ::2], hp_out[..., 1::2]])
        else:
            full_band_out = self.comb_module([lp_out, hp_out])
        full_band_out = self.fine_tune_module(full_band_out)

        return full_band_out


class Interponet_2(nn.Module):

    @capture_init
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_channels = args.experiment.interponet.hidden_channels
        self.hp_depth = args.experiment.interponet.hp_depth
        self.hp_n_res_layers = args.experiment.interponet.hp_n_res_layers

        self.lr_sr = args.experiment.interponet.lr_sr
        self.hr_sr = args.experiment.interponet.hr_sr
        self.scale_factor = int(self.hr_sr / self.lr_sr)

        self.n_upscale_blocks = int(math.log2(self.scale_factor))

        self.upscale_blocks = nn.ModuleList()

        self.upsample = args.experiment.interponet.upsample if 'upsample' in args.experiment.interponet else False
        self.load_hp = args.experiment.interponet.load_pretrained_hp if 'load_pretrained_hp' in args.experiment.interponet else False
        self.load_hp_path = args.experiment.interponet.load_hp_path if 'load_hp_path' in args.experiment.interponet else None
        self.hp_input = args.experiment.interponet.hp_input if 'hp_input' in args.experiment.interponet else None
        self.hp_module_name = args.experiment.interponet.high_pass_module if 'high_pass_module' in args.experiment.interponet else None
        self.hp_module_args = args.experiment.hp_module if 'hp_module' in args.experiment else None
        self.hp_freeze_params = args.experiment.interponet.hp_freeze_params if 'hp_freeze_params' in args.experiment.interponet else None
        self.use_lp_module = args.experiment.interponet.use_lp_module if 'use_lp_module' in args.experiment.interponet else True
        self.ft_module_name = args.experiment.interponet.fine_tune_module if 'fine_tune_module' in args.experiment.interponet else None
        self.ft_module_args = args.experiment.ft_module if 'ft_module' in args.experiment else None

        logger.info(f'hp_module_name: {self.hp_module_name}')


        for i in range(self.n_upscale_blocks):
            self.upscale_blocks.append(UpscaleBlock(1, self.hidden_channels, 1, self.hp_depth,
                                   self.hp_n_res_layers,
                                   self.lr_sr, self.hr_sr, self.upsample, self.hp_input,
                                   self.load_hp_path, self.hp_module_name, self.hp_module_args, self.hp_freeze_params,
                                   self.use_lp_module,
                                   self.ft_module_name, self.ft_module_args))

    def estimate_output_length(self, input_length):
        return input_length * self.scale_factor

    def forward(self, signal, hr_len):
        outs = []
        out = signal
        for i in range(self.n_upscale_blocks):
            out = self.upscale_blocks[i](out, hr_len)
            outs.append(out)

        return outs