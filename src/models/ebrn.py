import torch
from torch import nn
import logging

from src.models.modules import ResnetBlock, WNConv1d, WNConvTranspose1d, weights_init
from src.utils import capture_init

logger = logging.getLogger(__name__)

class ConvBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)

class BRM(nn.Module):

    @capture_init
    def __init__(self, channels, upscale_factor):
        super().__init__()
        # logger.info(f'upscale factor: {upscale_factor}')
        upscale_factor = int(upscale_factor)
        # super resolution flow
        self.upscale_layer = nn.Sequential(
            nn.LeakyReLU(0.2),
            WNConvTranspose1d(
                channels,
                channels,
                kernel_size=upscale_factor * 2,
                stride=upscale_factor,
                padding=upscale_factor // 2 + upscale_factor % 2,
                output_padding=upscale_factor % 2,
            ))

        sr_flow = []

        for j in range(3):
            sr_flow += [ConvBlock(channels, dilation=3 ** j)]

        self.sr_flow = nn.Sequential(*sr_flow)

        # back projection flow
        self.downscale_layer = nn.Sequential(
            nn.LeakyReLU(0.2),
            WNConv1d(channels,
                     channels,
                     kernel_size=upscale_factor * 2,
                     stride=upscale_factor,
                     padding=upscale_factor // 2 + upscale_factor % 2,
                     ),
        )

        bp_flow = []

        for j in range(3):
            bp_flow += [ConvBlock(channels, dilation=3 ** j)]

        self.bp_flow = nn.Sequential(*bp_flow)


    def forward(self, x):
        sr_x  = self.upscale_layer(x)
        bp_x = self.downscale_layer(sr_x)

        bp_res = bp_x - x
        sr_out = self.sr_flow(sr_x)
        bp_out = self.bp_flow(bp_res) + bp_res

        return sr_out, bp_out






class EBRN(nn.Module):

    @capture_init
    def __init__(self, channels, n_brm, lr_sr, hr_sr):
        super().__init__()
        self.channels = channels
        self.n_brm = n_brm
        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.scale_factor = self.hr_sr / self.lr_sr

        self.feature_extract_stack = nn.ModuleList()

        self.feature_extract_stack.append(nn.Sequential(
                                            nn.ReflectionPad1d(3),
                                            WNConv1d(1, channels, kernel_size=7, padding=0),
                                            nn.Tanh(),
                                        ))

        for i in range(1,3):
            self.feature_extract_stack.append(nn.Sequential(
                nn.ReflectionPad1d(3),
                WNConv1d(channels, channels, kernel_size=7, padding=0),
                nn.Tanh(),
            ))

        self.brm_stack = nn.ModuleList()

        for i in range(self.n_brm):
            self.brm_stack.append(BRM(self.channels, self.scale_factor))

        self.brm_wrappers = nn.ModuleList()



        for i in range(self.n_brm-1):
            self.brm_wrappers.append(nn.Sequential(nn.LeakyReLU(0.2),
                                                   nn.ReflectionPad1d(3),
                                                   WNConv1d(self.channels, self.channels, kernel_size=7, padding=0)))

        self.reconstruct_layer = nn.Sequential(nn.LeakyReLU(0.2),
                                               nn.ReflectionPad1d(3),
                                               WNConv1d(self.channels*self.n_brm, 1, kernel_size=7, padding=0))



    def forward(self, x, hr_len=None):

        sr_stack = []
        bp_stack = []

        # logger.info(f'x shape: {x.shape}')
        # logger.info(f'hr len: {hr_len}')

        for i, feature_extract in enumerate(self.feature_extract_stack):
            x = feature_extract(x)

        # logger.info(f'x shape: {x.shape}')

        bp = x

        # logger.info(f'bp shape: {bp.shape}')

        for i, brm in enumerate(self.brm_stack):
            sr, bp = brm(bp)
            sr_stack.append(sr)
            bp_stack.append(bp)

            # logger.info(f'{i}: bp shape: {bp.shape}')
            # logger.info(f'{i}: sr shape: {sr.shape}')

        for i in range(self.n_brm-1, 0, -1):
            sr_stack[i-1] = self.brm_wrappers[i-1](sr_stack[i] + sr_stack[i - 1])
            # logger.info(f' sr_stack[{i-1}]: {sr_stack[i - 1].shape}')

        out = torch.cat(sr_stack, dim=1)

        # logger.info(f'out: {out.shape}')

        out = self.reconstruct_layer(out)

        # logger.info(f'out: {out.shape}')

        return out