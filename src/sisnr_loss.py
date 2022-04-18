import torch
import numpy as np

class SisnrLoss(torch.nn.Module):

    def __init__(self):
        super(SisnrLoss, self).__init__()

    def forward(self, ref_sig, out_sig, eps=1e-8):
        """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
        Args:
            ref_sig: numpy.ndarray, [B, T]
            out_sig: numpy.ndarray, [B, T]
        Returns:
            SISNR
        """

        assert len(ref_sig) == len(out_sig)
        B, T = ref_sig.shape
        ref_sig = ref_sig - torch.mean(ref_sig, dim=1).reshape(B, 1)
        out_sig = out_sig - torch.mean(out_sig, dim=1).reshape(B, 1)
        ref_energy = (torch.sum(ref_sig ** 2, dim=1) + eps).reshape(B, 1)
        proj = (torch.sum(ref_sig * out_sig, dim=1).reshape(B, 1)) * \
               ref_sig / ref_energy
        noise = out_sig - proj
        ratio = torch.sum(proj ** 2, dim=1) / (torch.sum(noise ** 2, dim=1) + eps)
        sisnr = 10 * torch.log(ratio + eps) / np.log(10.0)
        return 1/sisnr.mean()