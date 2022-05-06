# import augment
import torch
from torch import nn
from torchaudio import sox_effects
import julius
import logging

logger = logging.getLogger(__name__)

class Augment(nn.Module):

    def __init__(self, sr, n_bands):
        super().__init__()
        self.sr = sr
        self.n_bands = n_bands

        # eps = 1e-7

        self.band_width = (sr/2) / n_bands
        self.bands = [[i * self.band_width, (i + 1) * self.band_width] for i in range(n_bands)]
        # self.bands[-1][-1] -= eps

        # self.band_reject_chains = []

        self.band_passes = nn.ModuleList()

        for band_start,band_end in self.bands:
            filter = julius.BandPassFilter(band_start / sr, band_end / sr)

            self.band_passes.append(filter)

            # self.band_passes.append([['sinc', '-a', '120', f'{band_start}-{band_end}']])
            # self.band_reject_chains.append(augment.EffectChain().sinc('-a', '120', f'{band_end}-{band_start}'))

    def apply_bandpass(self,x, band_idx):
        return self.band_passes[band_idx](x)
        # return sox_effects.apply_effects_tensor(x, self.sr, self.band_passes[band_idx], channels_first=True)[0]

    def forward(self, x):
        """

        :param x: [Batch,1,Time]
        :return: [Batch, n_bands, Time]
        """
        bands = []
        for band_idx in range(self.n_bands):
            bands.append(self.band_passes[band_idx](x))
            # bands.append(sox_effects.apply_effects_tensor(x, self.sr, self.band_passes[band_idx], channels_first=True)[0])
            # bands.append(self.band_reject_chains[band_idx].apply(x, src_info={'rate': self.sr}))
        return torch.cat(bands, dim=0)