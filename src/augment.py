import augment
import torch
from torch import nn
from torchaudio import sox_effects


class Augment(nn.Module):

    def __init__(self, sr, n_bands):
        super().__init__()
        self.sr = sr
        self.n_bands = n_bands

        eps = 1e-7

        self.band_width = (sr/2) / n_bands - eps
        self.bands = [(i * self.band_width, (i + 1) * self.band_width) for i in range(n_bands)]

        # self.band_reject_chains = []

        self.effects = []

        for band_start,band_end in self.bands:
            self.effects.append([['sinc', '-a', '120', f'{band_end}-{band_start}']])
            # self.band_reject_chains.append(augment.EffectChain().sinc('-a', '120', f'{band_end}-{band_start}'))

    def forward(self, x):
        """

        :param x: [Batch,1,Time]
        :return: [Batch, n_bands, Time]
        """
        bands = []
        for band_idx in range(self.n_bands):
            bands.append(sox_effects.apply_effects_tensor(x, self.sr, self.effects[band_idx], channels_first=True)[0])
            # bands.append(self.band_reject_chains[band_idx].apply(x, src_info={'rate': self.sr}))
        return torch.cat(bands, dim=0)