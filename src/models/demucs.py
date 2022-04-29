# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import math

from torch import nn
from torch.nn import functional as F

from src.models.modules import BLSTM
from torchaudio.functional import resample
from src.utils import capture_init

import logging

logger = logging.getLogger(__name__)


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)




class Demucs(nn.Module):
    """
    Demucs speech enhancement model.
    Args:
        - chin (int): number of input channels.
        - chout (int): number of output channels.
        - hidden (int): number of initial hidden channels.
        - depth (int): number of layers.
        - kernel_size (int): kernel size for each layer.
        - stride (int): stride for each layer.
        - causal (bool): if false, uses BiLSTM instead of LSTM.
        - resample (int): amount of resampling to apply to the input/output.
            Can be one of 1, 2 or 4.
        - growth (float): number of channels is multiplied by this for every layer.
        - max_hidden (int): maximum number of channels. Can be useful to
            control the size/speed of the model.
        - normalize (bool): if true, normalize the input.
        - glu (bool): if true uses GLU instead of ReLU in 1x1 convolutions.
        - rescale (float): controls custom weight initialization.
            See https://arxiv.org/abs/1911.13254.
        - floor (float): stability flooring when normalizing.

    """
    @capture_init
    def __init__(self, chin: int = 1,
                 chout: int = 1,
                 hidden: int = 48,
                 max_hidden: int = 10000,
                 causal: bool = True,
                 floor: float = 1e-3,
                 glu: bool = True,
                 depth: int = 5,
                 kernel_size: int = 8,
                 stride_sizes = [2, 2, 8, 8],
                 normalize: bool = True,
                 resample: int = 1,
                 growth: int = 2,
                 rescale: float = 0.1,
                 lr_sr=16000,
                 hr_sr=16000):

        super().__init__()
        if resample not in [1, 2, 4]:
            raise ValueError("Resample should be 1, 2 or 4.")

        self.chin = chin
        self.chout = chout
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.stride_sizes = stride_sizes
        self.depth = depth
        self.causal = causal
        self.floor = floor
        self.resample = resample
        self.normalize = normalize
        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.scale_factor = self.hr_sr / self.lr_sr

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        activation = nn.GLU(1) if glu else nn.ReLU()
        ch_scale = 2 if glu else 1
        chin, hidden, chout = chin, hidden, chout

        for index, stride_size in enumerate(self.stride_sizes):
            encode = []
            encode += [
                nn.Conv1d(chin, hidden, self.kernel_size, stride_size),
                nn.ReLU(),
                nn.Conv1d(hidden, hidden * ch_scale, 1), activation,
            ]
            self.encoder.append(nn.Sequential(*encode))

            decode = []
            decode += [
                nn.Conv1d(hidden, ch_scale * hidden, 1), activation,
                nn.ConvTranspose1d(hidden, chout, self.kernel_size, stride_size),
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            chout = hidden
            chin = hidden
            hidden = min(int(growth * hidden), max_hidden)

        self.lstm = BLSTM(chin, bi=not causal)
        if rescale:
            rescale_module(self, reference=rescale)

    def estimate_output_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.resample)
        for stride_size in self.stride_sizes:
            length = math.ceil((length - self.kernel_size)/stride_size + 1)
            length = max(length, 1)
        for stride_size in self.stride_sizes[::-1]:
            length = (length - 1) * stride_size + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    def pad_to_valid_length(self, signal):
        valid_length = self.estimate_output_length(signal.shape[-1])
        padding_len = valid_length - signal.shape[-1]
        signal = F.pad(signal, (0, padding_len))
        return signal, padding_len

    def forward(self, signal, hr_len=None):

        if signal.dim() == 2:
            signal = signal.unsqueeze(1)


        if self.normalize:
            mono = signal.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            signal = signal / (self.floor + std)
        else:
            std = 1
        x = signal

        x = resample(x, self.lr_sr, self.hr_sr)

        x, padding_len = self.pad_to_valid_length(x)

        if self.resample > 1:
            x = resample(x, self.hr_sr, self.resample*self.hr_sr)

        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        x = self.lstm(x)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip[..., :x.shape[-1]]
            x = decode(x)

        if self.resample > 1:
            x = resample(x, self.resample * self.hr_sr, self.hr_sr)

        if hr_len and hr_len < x.shape[-1]:
            x = x[..., :hr_len]

        return std * x
