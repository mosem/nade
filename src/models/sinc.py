import torch.nn as nn
from torchaudio.transforms import Resample
from src.utils import capture_init


class Sinc(nn.Module):

    @capture_init
    def __init__(self, lr_sr, hr_sr):
        super().__init__()
        self.resample_transform = Resample(lr_sr, hr_sr)

    def forward(self, x, hr_len):
        return self.resample_transform(x)