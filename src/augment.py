import augment
from torch import nn

class Augment(nn.Module):

    def __init__(self, sr, n_bands):
        self.sr = sr
        self.n_bands = n_bands

        self.band_width = sr / n_bands
        self.bands = [(i * self.band_width, (i + 1) * self.band_width) for i in range(n_bands)]

        self.band_reject_chains = []

        for band_start,band_end in self.bands:
            self.band_reject_chains.append(augment.EffectChain().sinc('-a', '120', f'{band_end}-{band_start}'))

    def forward(self, x, band_idx):
        return self.band_reject_chains[band_idx].apply(x, src_info={'rate': self.sr})