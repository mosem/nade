import torch
from torch import nn

import math
import numpy as np

from src.utils import capture_init

import logging
logger = logging.getLogger(__name__)


def get_upsample_filter(size):
    """Make a 1D linear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size]
    filter = (1 - abs(og - center) / factor)
    return torch.from_numpy(filter).float()

class ConvBlock(nn.Module):

    def __init__(self, channels=64, conv_kernel_size=3):
        super().__init__()

        self.channels = channels
        self.conv_kernel_size = conv_kernel_size

        self.block = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=conv_kernel_size, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=conv_kernel_size, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=conv_kernel_size, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=conv_kernel_size, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=conv_kernel_size, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=conv_kernel_size, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=conv_kernel_size, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=conv_kernel_size, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=conv_kernel_size, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=conv_kernel_size, stride=1, padding='same', bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose1d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, signal):
        output = self.block(signal)
        return output


class LapSrn(nn.Module):

    @capture_init
    def __init__(self, channels, lr_sr, hr_sr):
        super().__init__()

        self.channels = channels
        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.scale_factor = self.hr_sr / self.lr_sr


        self.conv_input = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.pyramid_layers = nn.ModuleList()

        for i in range(int(self.scale_factor)-1):
            convt_I1 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False)
            convt_R1 = nn.Conv1d(in_channels=channels, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
            convt_F1 = ConvBlock(channels=channels)

            self.pyramid_layers.append(nn.ModuleDict({'convt_I': convt_I1, 'convt_R': convt_R1, 'convt_F': convt_F1}))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, signal, hr_len=None):
        lr = signal
        out = self.relu(self.conv_input(lr))

        hr_signals = []

        for i in range(int(self.scale_factor)-1):
            out = self.pyramid_layers[i]['convt_F'](out)
            convt_I = self.pyramid_layers[i]['convt_I'](lr)
            convt_R = self.pyramid_layers[i]['convt_R'](out)
            hr_signals.append(convt_I + convt_R)
            lr = convt_I + convt_R

        # for i,sig in enumerate(hr_signals):
        #     logger.info(f'{i}: length = {sig.shape}')
        return hr_signals
