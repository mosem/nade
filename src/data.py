# This code is based on FAIR's Demucs denoiser: https://github.com/facebookresearch/denoiser

import json
import logging
import os
from tqdm import tqdm
import torch
import torchaudio
import math
import sys

from torch.nn import functional as F
from torch.utils.data import Dataset
from src.audio import Audioset
from torchaudio import transforms
from torchaudio.functional import resample

from src.augment import Augment

logger = logging.getLogger(__name__)


def match_files(lr, hr):
    """match_files.
    Sort files to match lr and hr filenames.
    :param lr: list of the low-resolution filenames
    :param hr: list of the high-resolution filenames
    """
    lr.sort()
    hr.sort()

def assert_sets(lr_set, hr_set):
    transform = transforms.Resample(lr_set.sample_rate, hr_set.sample_rate)
    n_samples = len(lr_set)
    for i in tqdm(range(n_samples)):
        upsampled_lr_i = transform(lr_set[i])
        hr_i = match_hr_signal(hr_set[i], hr_set.sample_rate / lr_set.sample_rate)
        assert upsampled_lr_i.shape == hr_i.shape


def match_hr_signal(hr_sig, lr_sig, scale):
    hr_len = hr_sig.shape[-1]
    lr_len = lr_sig.shape[-1]
    new_lr_len = int(lr_len*scale)
    if hr_len < new_lr_len:
        hr_sig = F.pad(hr_sig, (0, new_lr_len - hr_sig.shape[-1]))
    elif hr_len > new_lr_len:
        hr_sig = hr_sig[...,:new_lr_len]
    return hr_sig

def match_signal(signal, ref_len):
    sig_len = signal.shape[-1]
    if sig_len < ref_len:
        signal = F.pad(signal, (0, ref_len - sig_len))
    elif sig_len > ref_len:
        signal = signal[..., :ref_len]
    return signal

class PrHrSet(Dataset):
    def __init__(self, samples_dir, filenames=None):
        self.samples_dir = samples_dir
        if filenames is not None:
            files = [i for i in os.listdir(samples_dir) if any(i for j in filenames if j in i)]
        else:
            files = os.listdir(samples_dir)

        self.hr_filenames = list(sorted(filter(lambda x: x.endswith('_hr.wav'), files)))
        self.lr_filenames = list(sorted(filter(lambda x: x.endswith('_lr.wav'), files)))
        self.pr_filenames = list(sorted(filter(lambda x: x.endswith('_pr.wav'), files)))

    def __len__(self):
        return len(self.hr_filenames)

    def __getitem__(self, i):
        lr_i, lr_sr = torchaudio.load(os.path.join(self.samples_dir, self.lr_filenames[i]))
        hr_i, hr_sr = torchaudio.load(os.path.join(self.samples_dir, self.hr_filenames[i]))
        pr_i, pr_sr = torchaudio.load(os.path.join(self.samples_dir, self.pr_filenames[i]))
        pr_i = match_signal(pr_i, hr_i.shape[-1])
        assert hr_i.shape == pr_i.shape
        lr_filename = self.lr_filenames[i]
        lr_filename = lr_filename[:lr_filename.index('_lr.wav')]
        hr_filename = self.hr_filenames[i]
        hr_filename = hr_filename[:hr_filename.index('_hr.wav')]
        pr_filename = self.pr_filenames[i]
        pr_filename = pr_filename[:pr_filename.index('_pr.wav')]
        assert lr_filename == hr_filename == pr_filename
        return lr_i, hr_i, pr_i, lr_filename


class LrHrSet(Dataset):
    def __init__(self, json_dir, lr_sr, hr_sr, stride = None, segment = None,
                 pad=True, with_path=False, n_bands=2):
        """__init__.
        :param json_dir: directory containing both hr.json and lr.json
        :param length: maximum sequence length
        :param stride: the stride used for splitting audio sequences in seconds
        :param segment: the segment length used for splitting audio sequences in seconds
        :param pad: pad the end of the sequence with zeros
        :param sample_rate: the signals sampling rate
        :param with_path: whether to return tensors with filepath
        """

        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.with_path = with_path
        self.n_bands = n_bands
        self.hr_augment = Augment(self.hr_sr, self.n_bands)
        lr_json = os.path.join(json_dir, 'lr.json')
        hr_json = os.path.join(json_dir, 'hr.json')
        with open(lr_json, 'r') as f:
            lr = json.load(f)
        with open(hr_json, 'r') as f:
            hr = json.load(f)

        lr_stride = stride * lr_sr if stride else None
        hr_stride = stride * hr_sr if stride else None
        lr_length = segment * lr_sr if segment else None
        hr_length = segment * hr_sr if segment else None

        match_files(lr, hr)
        self.lr_set = Audioset(lr, sample_rate=lr_sr, length=lr_length, stride=lr_stride, pad=pad, channels=1,
                               with_path=with_path)
        self.hr_set = Audioset(hr, sample_rate=hr_sr, length=hr_length, stride=hr_stride, pad=pad, channels=1,
                               with_path=with_path)
        assert len(self.hr_set) == len(self.lr_set)


    def __getitem__(self, index):
        if self.with_path:
            hr_sig, hr_path = self.hr_set[index]
            lr_sig, lr_path = self.lr_set[index]
        else:
            hr_sig = self.hr_set[index]
            lr_sig = self.lr_set[index]
        hr_sig = match_hr_signal(hr_sig, lr_sig, self.hr_sr / self.lr_sr)

        lr_sig = resample(lr_sig, self.lr_sr, self.hr_sr)

        hr_sig = self.hr_augment(hr_sig)

        # masks = torch.zeros_like(hr_sig)
        # lr_sig = torch.cat([lr_sig, hr_sig, masks], dim=0)
        lr_sig = torch.cat([lr_sig, hr_sig], dim=0)

        if self.with_path:
            return (lr_sig, lr_path), (hr_sig, hr_path)
        else:
            return lr_sig, hr_sig

    def __len__(self):
        return len(self.lr_set)


if __name__ == "__main__":
    json_dir = '../egs/vctk/16-24/val'
    lr_sr = 16000
    hr_sr = 24000
    pad = True
    stride_sec = 2
    segment_sec = 2

    data_set = LrHrSet(json_dir, lr_sr, hr_sr, stride_sec, segment_sec)
    assert_sets(data_set.lr_set, data_set.hr_set)
    print(f'done asserting dataset from {json_dir}')
