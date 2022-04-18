import math
import torch
import torch.nn as nn
from torchaudio.functional import resample
from src.models.modules import TorchSignalToFrames, TorchOLA, DualTransformer
from src.utils import capture_init

class Transformer(nn.Module):

    @capture_init
    def __init__(self, frame_size=512, hidden=64,
                 in_channels=1,
                 out_channels=1,
                 lr_sr=16000,
                 hr_sr=16000):
        super(Transformer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden = hidden
        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.scale_factor = self.hr_sr / self.lr_sr
        self.frame_size = frame_size
        self.frame_shift = self.frame_size // 2

        self.signalPreProcessor = TorchSignalToFrames(frame_size=self.frame_size,
                                                      frame_shift=self.frame_shift)

        self.dual_transformer = DualTransformer(self.hidden, self.hidden, nhead=4, num_layers=6)

        self.out_conv = nn.Conv2d(in_channels=self.hidden, out_channels=self.out_channels, kernel_size=(1, 1))
        self.ola = TorchOLA(self.frame_shift)

    def forward(self, x, hr_len=None):
        # x = resample(x, self.lr_sr, self.hr_sr)

        skips = []
        out = self.signalPreProcessor(x)

        out = self.dual_transformer(out)

        out = self.out_conv(out)
        out = self.ola(out)

        out = out[..., :hr_len]

        return out