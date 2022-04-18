import math
import torch
import torch.nn as nn
from torchaudio.functional import resample
from src.models.modules import TorchSignalToFrames, TorchOLA, DualTransformer
from src.utils import capture_init

import logging

logger = logging.getLogger(__name__)


class Dsconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), stride=1, padding=0, dilation=(1, 1)):
        super(Dsconv2d, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding_size, r=1):
        # upconvolution only along second dimension of image
        # Upsampling using sub pixel layers
        super(SPConvTranspose2d, self).__init__()
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, padding=(0, padding_size))
        self.r = r

    def forward(self, x):
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out


class DenseBlock(nn.Module):
    def __init__(self, input_size, depth=5, in_channels=64):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    Dsconv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                             dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.LayerNorm(input_size))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out


class DecodeLayer(nn.Module):
    def __init__(self, dense_block, decode_block):
        super(DecodeLayer, self).__init__()
        self.dense_block = dense_block
        self.decode_block = decode_block

    def forward(self, x, skip):
        dense_out = self.dense_block(x)
        cat_out = torch.cat([dense_out, skip], dim=1)
        out = self.decode_block(cat_out)
        return out


class Caunet(nn.Module):

    @capture_init
    def __init__(self, frame_size=512, hidden=64, depth=5, dense_block_depth=3, kernel_size=3,
                 stride_size=2,
                 in_channels=1,
                 out_channels=1,
                 lr_sr=16000,
                 hr_sr=16000):
        super(Caunet, self).__init__()
        self.depth = depth
        self.dense_block_depth = dense_block_depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = (1, kernel_size)
        self.stride_size = stride_size
        self.padding_size = kernel_size // 2
        self.hidden = hidden
        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.scale_factor = self.hr_sr / self.lr_sr
        self.frame_size = self._estimate_valid_frame_size(frame_size)
        self.frame_shift = self.frame_size // 2

        self.signalPreProcessor = TorchSignalToFrames(frame_size=self.frame_size,
                                                      frame_shift=self.frame_shift)
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        encoder_input_size = self.frame_size
        input_layer = [nn.Conv2d(in_channels=self.in_channels, out_channels=self.hidden, kernel_size=(1, 1)),
                       nn.LayerNorm(encoder_input_size),
                       nn.PReLU(self.hidden)]
        self.encoder.append(nn.Sequential(*input_layer))
        for i in range(self.depth):
            encoder_output_size = (encoder_input_size + 2 * self.padding_size - self.kernel[1]) // self.stride_size + 1
            encode_layer = [DenseBlock(encoder_input_size, self.dense_block_depth, self.hidden),
                            nn.Conv2d(in_channels=self.hidden, out_channels=self.hidden, kernel_size=self.kernel,
                                      stride=(1, self.stride_size), padding=(0, self.padding_size)),
                            nn.LayerNorm(encoder_output_size),
                            nn.PReLU(self.hidden)]

            dense_block = DenseBlock(encoder_output_size, self.dense_block_depth, self.hidden)
            decode_block = nn.Sequential(SPConvTranspose2d(in_channels=self.hidden * 2, out_channels=self.hidden,
                                                           kernel_size=self.kernel, padding_size=self.padding_size,
                                                           r=self.stride_size),
                                         nn.LayerNorm(encoder_input_size),
                                         nn.PReLU(self.hidden))
            decode_layer = DecodeLayer(dense_block, decode_block)

            self.encoder.append(nn.Sequential(*encode_layer))
            self.decoder.insert(0, decode_layer)

            encoder_input_size = encoder_output_size

        self.dual_transformer = DualTransformer(self.hidden, self.hidden, nhead=4, num_layers=6)

        self.out_conv = nn.Conv2d(in_channels=self.hidden, out_channels=self.out_channels, kernel_size=(1, 1))
        self.ola = TorchOLA(self.frame_shift)

    def _estimate_valid_frame_size(self, frame_size):
        valid_frame_size = frame_size
        for i in range(self.depth):
            valid_frame_size = (valid_frame_size + 2 * self.padding_size - self.kernel[1]) // self.stride_size + 1
            valid_frame_size = max(valid_frame_size, 1)
        for i in range(self.depth):
            valid_frame_size = (valid_frame_size + 2 * self.padding_size - self.kernel[1] + 1) * self.stride_size
        return valid_frame_size

    def estimate_output_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        length = math.ceil(length * self.scale_factor)
        n_frames = math.ceil((length - self.frame_size) / self.frame_shift + 1)
        length = (n_frames - 1) * self.frame_shift + self.frame_size
        return int(length)

    def forward(self, x, hr_len=None):
        # x = resample(x, self.lr_sr, self.hr_sr)

        skips = []
        # logger.info(f'1. {x.shape}')
        out = self.signalPreProcessor(x)

        # logger.info(f'2. {out.shape}')
        for i, encode in enumerate(self.encoder):
            out = encode(out)
            # logger.info(f'3-{i}. {out.shape}')
            skips.append(out)

        out = self.dual_transformer(out)

        # logger.info(f'4. {out.shape}')

        for i, decode in enumerate(self.decoder):
            skip = skips.pop(-1)
            out = decode(out, skip)
            # logger.info(f'5-{i}. {out.shape}')

        out = self.out_conv(out)
        # logger.info(f'6. {out.shape}')
        out = self.ola(out)
        # logger.info(f'7. {out.shape}')

        out = out[..., :hr_len]
        # logger.info(f'8. {out.shape}')

        return out