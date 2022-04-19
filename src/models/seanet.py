import torch.nn as nn
import math
from src.models.modules import ResnetBlock, WNConv1d, WNConvTranspose1d, weights_init
from src.utils import capture_init
from torchaudio.functional import resample
from torch.nn import functional as F

import logging
logger = logging.getLogger(__name__)

class Seanet(nn.Module):

    @capture_init
    def __init__(self,
                 latent_space_size=128,
                 ngf=32, n_residual_layers=3,
                 resample=1,
                 normalize=True,
                 floor=1e-3,
                 ratios=[8, 8, 2, 2],
                 in_channels=1,
                 out_channels=1,
                 lr_sr=16000,
                 hr_sr=16000,
                 upsample=True):
        super().__init__()

        self.resample = resample
        self.normalize = normalize
        self.floor = floor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.scale_factor = int(self.hr_sr/self.lr_sr)
        self.upsample = upsample

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.ratios = ratios
        mult = int(2 ** len(ratios))

        decoder_wrapper_conv_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(latent_space_size, mult * ngf, kernel_size=7, padding=0),
        ]

        encoder_wrapper_conv_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(mult * ngf, latent_space_size, kernel_size=7, padding=0)
        ]

        self.encoder.insert(0, nn.Sequential(*encoder_wrapper_conv_layer))
        self.decoder.append(nn.Sequential(*decoder_wrapper_conv_layer))

        for i, r in enumerate(ratios):
            encoder_block = [
                nn.LeakyReLU(0.2),
                WNConv1d(mult * ngf // 2,
                         mult * ngf,
                         kernel_size=r * 2,
                         stride=r,
                         padding=r // 2 + r % 2,
                         ),
            ]

            decoder_block = [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers - 1, -1, -1):
                encoder_block = [ResnetBlock(mult * ngf // 2, dilation=3 ** j)] + encoder_block

            for j in range(n_residual_layers):
                decoder_block += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

            self.encoder.insert(0, nn.Sequential(*encoder_block))
            self.decoder.append(nn.Sequential(*decoder_block))

        encoder_wrapper_conv_layer = [
            nn.ReflectionPad1d(3),
            WNConv1d(self.in_channels, ngf, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.encoder.insert(0, nn.Sequential(*encoder_wrapper_conv_layer))

        decoder_wrapper_conv_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.decoder.append(nn.Sequential(*decoder_wrapper_conv_layer))

        self.apply(weights_init)

    def estimate_output_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        depth = len(self.ratios)
        for idx in range(depth - 1, -1, -1):
            stride = self.ratios[idx]
            kernel_size = 2 * stride
            padding = stride // 2 + stride % 2
            length = math.ceil((length - kernel_size + 2 * padding) / stride) + 1
            length = max(length, 1)
        for idx in range(depth):
            stride = self.ratios[idx]
            kernel_size = 2 * stride
            padding = stride // 2 + stride % 2
            output_padding = stride % 2
            length = (length - 1) * stride + kernel_size - 2 * padding + output_padding
        return int(length)

    def pad_to_valid_length(self, signal):
        valid_length = self.estimate_output_length(signal.shape[-1])
        padding_len = valid_length - signal.shape[-1]
        signal = F.pad(signal, (0, padding_len))
        return signal, padding_len

    def forward(self, signal, hr_len=None):
        """

        :param signal: [Batch-size, in_channels, Time]
                    in_channels: lr channel, hr_band_1,...,hr_band_n
        :param hr_len:
        :return:  [Batch-size, out_channels, Time]
                    out_channels: hr_band_1,...,hr_band_n
        """

        target_len = signal.shape[-1]
        if self.upsample:
            target_len *= self.scale_factor
        if self.normalize:
            mono = signal.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            signal = signal / (self.floor + std)
        else:
            std = 1
        x = signal
        # print(f'target_len: {target_len}')
        logger.info(f'beginning of seanet: {x.shape}')
        if self.upsample:
            x = resample(x,self.lr_sr, self.hr_sr)
        # print(f'after resample: {x.shape}')

        x, padding_len = self.pad_to_valid_length(x)
        # print(f'after padding: {x.shape}, padding: {padding_len}')
        skips = []
        for i, encode in enumerate(self.encoder):
            logger.info(f'encode {i}. x shape: {x.shape}')
            x = encode(x)
            skips.append(x)
        for j, decode in enumerate(self.decoder):
            skip = skips.pop(-1)
            logger.info(f'decode {j}. x shape: {x.shape}, skip shape: {skip.shape}')
            x = x + skip
            x = decode(x)
        # print(f'before trimming: {x.shape}')
        # trim_len = x.shape[-1] - hr_len
        if target_len < x.shape[-1]:
            x = x[...,:target_len]
        # print(f'after trimming: {x.shape}, trim: {trim_len}')
        # print(f'end of seanet: {x.shape}')
        x = std * x
        x = x.view(x.size(0), self.out_channels, x.size(-1))
        logger.info(f'end of seanet: {x.shape}')
        return x

