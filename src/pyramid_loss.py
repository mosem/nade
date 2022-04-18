import math

import torch
from torchaudio import transforms

import torch.nn.functional as F

from src.sisnr_loss import SisnrLoss
from src.stft_loss import SpectralConvergengeLoss, LogSTFTMagnitudeLoss, stft


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", factor_sc=0.1,
                 factor_mag=0.1):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.x_spectogram_transform = transforms.Spectrogram(n_fft=self.fft_size//2)
        self.y_spectogram_transform = transforms.Spectrogram(n_fft=self.fft_size)
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y, y_pass: str):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
            y_pass: low/high
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = torch.clamp(self.x_spectogram_transform(x), min=1e-7).transpose(-1, -2)
        y_mag = torch.clamp(self.y_spectogram_transform(y), min=1e-7).transpose(-1, -2)
        # print(f'before {y_pass}-pass. y_mag: {y_mag.shape}, x_mag: {x_mag.shape}')
        y_n_fft = y_mag.shape[-1]
        x_n_fft = x_mag.shape[-1]
        assert math.ceil(y_n_fft/2) == x_n_fft
        if y_pass == 'high':
            y_mag = y_mag[...,:x_n_fft]
        else:
            y_mag = y_mag[..., x_n_fft-1:]

        # print(f'after {y_pass}-pass. y_mag: {y_mag.shape}, x_mag: {x_mag.shape}')

        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return self.factor_sc*sc_loss + self.factor_mag*mag_loss


class PyramidLoss(torch.nn.Module):

    def __init__(self, args):
        super(PyramidLoss, self).__init__()
        self.downsample_fn = transforms.Resample(args.experiment.hr_sr, args.experiment.lr_sr).to(args.device)
        self.upsample_fn = transforms.Resample(args.experiment.lr_sr, args.experiment.hr_sr).to(args.device)

        self.sisnrloss = SisnrLoss()

        self.args = args

        self.low_loss_lambda = args.experiment.low_loss_lambda
        self.high_loss_labmda = args.experiment.high_loss_lambda

        if self.args.experiment.pyramid_loss == 'stft':
            self.stft_loss = STFTLoss().to(args.device)



    def _get_loss(self, hr, pr):
        return F.l1_loss(hr, pr)
        # loss = 0
        # if self.args.loss == 'l1':
        #     loss = F.l1_loss(hr, pr)
        # elif self.args.loss == 'l2':
        #     loss = F.mse_loss(hr, pr)
        # elif self.args.loss == 'sisnr':
        #     loss = self.sisnrloss(hr.squeeze(dim=1), pr.squeeze(dim=1))
        # return loss

    def forward(self, pr_signals, hr):
        hr = hr.repeat((1,self.args.experiment.interponet.out_channels,1))
        if self.args.experiment.pyramid_loss == 'sinc':
            pr_low_pass = self.upsample_fn(pr_signals['low_pass'])
            pr_high_pass = self.upsample_fn(pr_signals['high_pass'])

            hr_low_pass = self.upsample_fn(self.downsample_fn(hr))
            hr_residual = hr - hr_low_pass

            low_loss = self._get_loss(hr_low_pass, pr_low_pass)
            high_loss = self._get_loss(pr_high_pass, hr_residual)
        elif self.args.experiment.pyramid_loss == 'subsample':
            band_bank = [pr_signals['low_pass']] + pr_signals['high_pass']
            hop_size = len(band_bank)
            pyramid_losses = {}
            for i in range(hop_size):
                sub_samples_i = hr[...,i::hop_size]
                pyramid_loss = self._get_loss(sub_samples_i, band_bank[i])
                pyramid_losses.update({f'pyramid_loss_{i}': pyramid_loss})

            return pyramid_losses
            # pr_low_pass = pr_signals['low_pass']
            # pr_high_pass = pr_signals['high_pass']
            # hr_sub_sample_0 = hr[..., ::2]
            # hr_sub_sample_1 = hr[..., 1::2]
            #
            # low_loss = self._get_loss(hr_sub_sample_0, pr_low_pass)
            # high_loss = self._get_loss(hr_sub_sample_1, pr_high_pass)
        elif self.args.experiment.pyramid_loss == 'stft':
            pr_low_pass = pr_signals['low_pass']
            pr_high_pass = pr_signals['high_pass']

            low_loss = self.stft_loss(pr_low_pass, hr, 'low')
            high_loss = self.stft_loss(pr_high_pass, hr, 'high')


        low_loss *= self.low_loss_lambda
        high_loss *= self.high_loss_labmda

        return {'low': low_loss, 'high': high_loss}


