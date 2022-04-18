import torch
from torch import nn

class Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self, eps=1e-6):
        super(Charbonnier_loss, self).__init__()
        self.eps = eps

    def forward(self, ref_sig, out_sig):

        diff = out_sig - ref_sig
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.sum(error)

        return loss