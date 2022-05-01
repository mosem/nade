# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

from torchaudio.functional import resample
from torch import nn

from src.utils import capture_init, center_trim
from torch.nn import functional as F

import logging

logger = logging.getLogger(__name__)


class BLSTM(nn.Module):
    def __init__(self, dim, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(bidirectional=True, num_layers=layers, hidden_size=dim, input_size=dim)
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class DemucsSourceSep(nn.Module):
    @capture_init
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 channels=64,
                 depth=6,
                 rewrite=True,
                 glu=True,
                 rescale=0.1,
                 resample=False,
                 kernel_size=8,
                 stride=4,
                 growth=2.,
                 lstm_layers=2,
                 context=3,
                 normalize=False,
                 lr_sr=16000,
                 hr_sr=16000,
                 upsample=False,
                 cumulative=False):
        """
        Args:
            sources (list[str]): list of source names
            audio_channels (int): stereo or mono
            channels (int): first convolution channels
            depth (int): number of encoder/decoder layers
            rewrite (bool): add 1x1 convolution to each encoder layer
                and a convolution to each decoder layer.
                For the decoder layer, `context` gives the kernel size.
            glu (bool): use glu instead of ReLU
            resample_input (bool): upsample x2 the input and downsample /2 the output.
            rescale (int): rescale initial weights of convolutions
                to get their standard deviation closer to `rescale`
            kernel_size (int): kernel size for convolutions
            stride (int): stride for convolutions
            growth (float): multiply (resp divide) number of channels by that
                for each layer of the encoder (resp decoder)
            lstm_layers (int): number of lstm layers, 0 = no lstm
            context (int): kernel size of the convolution in the
                decoder before the transposed convolution. If > 1,
                will provide some context from neighboring time
                steps.
            samplerate (int): stored as meta information for easing
                future evaluations of the model.
            segment_length (int): stored as meta information for easing
                future evaluations of the model. Length of the segments on which
                the model was trained.
        """

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.context = context
        self.stride = stride
        self.depth = depth
        self.resample = resample
        self.channels = channels
        self.normalize = normalize

        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.scale_factor = self.hr_sr / self.lr_sr

        self.cumulative = cumulative
        self.upsample = upsample

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        if glu:
            activation = nn.GLU(dim=1)
            ch_scale = 2
        else:
            activation = nn.ReLU()
            ch_scale = 1
        for index in range(depth):
            encode = []
            encode += [nn.Conv1d(in_channels, channels, kernel_size, stride), nn.ReLU()]
            if rewrite:
                encode += [nn.Conv1d(channels, ch_scale * channels, 1), activation]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = self.out_channels
            if rewrite:
                decode += [nn.Conv1d(channels, ch_scale * channels, context), activation]
            decode += [nn.ConvTranspose1d(channels, out_channels, kernel_size, stride)]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            in_channels = channels
            channels = int(growth * channels)

        channels = in_channels

        if lstm_layers:
            self.lstm = BLSTM(channels, lstm_layers)
        else:
            self.lstm = None

        if rescale:
            rescale_module(self, reference=rescale)

    def estimate_output_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length when context = 1. If context > 1,
        the two signals can be center trimmed to match.

        For training, extracts should have a valid length.For evaluation
        on full tracks we recommend passing `pad = True` to :method:`forward`.
        """
        # if self.resample:
        #     length *= 2
        for i in range(self.depth):
            # logger.info(f'{i}: {length}')
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1
        for j in range(self.depth):
            # logger.info(f'{j}: {length}')
            length = (length - 1) * self.stride + self.kernel_size

        # if self.resample:
        #     length = math.ceil(length / 2)
        # logger.info(f'final: {length}')
        return int(length)

    def pad_to_valid_length(self, signal):
        valid_length = self.estimate_output_length(signal.shape[-1])
        padding_len = valid_length - signal.shape[-1]
        signal = F.pad(signal, (0, padding_len))
        return signal, padding_len

    def forward(self, mix, hr_len=None):
        x = mix
        # logger.info(f'beginning of demucs: {x.shape}')

        target_len = x.shape[-1]
        if self.upsample:
            target_len *= self.scale_factor

        if self.resample:
            target_len *= 2

        if self.normalize:
            mono = mix.mean(dim=1, keepdim=True)
            mean = mono.mean(dim=-1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
        else:
            mean = 0
            std = 1

        x = (x - mean) / (1e-5 + std)
        if self.resample:
            x = resample(x, self.hr_sr, self.hr_sr*2)
            # logger.info(f'after resample: {x.shape}')


        x, padding_len = self.pad_to_valid_length(x)

        # logger.info(f'after padding: {x.shape}, padding: {padding_len}')

        saved = []
        for encode in self.encoder:
            # logger.info(f'encode: {x.shape[-1]}')
            x = encode(x)
            saved.append(x)
        if self.lstm:
            x = self.lstm(x)
        for decode in self.decoder:
            # logger.info(f'decode: {x.shape[-1]}')
            skip = center_trim(saved.pop(-1), x)
            x = x + skip
            x = decode(x)

        # logger.info(f'end of demucs: {x.shape}')
        if target_len < x.shape[-1]:
            x = x[..., :target_len]

        # logger.info(f'before resample: {x.shape}')
        if self.resample:
            x = resample(x, self.hr_sr * 2, self.hr_sr)
            # logger.info(f'after resample: {x.shape}')

        x = x * std + mean
        x = x.view(x.size(0), self.out_channels, x.size(-1))
        # logger.info(f'end end of demucs: {x.shape}')
        return x