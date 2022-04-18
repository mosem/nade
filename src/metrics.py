import os
import subprocess
import logging

import numpy as np
import sox
import torch
from pesq import pesq
from pystoi import stoi

from src.stft_loss import stft

logger = logging.getLogger(__name__)

VISQOL_PATH = "/cs/labs/adiyoss/moshemandel/visqol-master; "

def run_metrics(clean, estimate, args, filename):
    hr_sr = args.experiment.hr_sr if 'experiment' in args else args.hr_sr
    pesq, stoi, snr, lsd, sisnr, visqol = get_metrics(clean, estimate, hr_sr, filename)
    return pesq, stoi, snr, lsd, sisnr, visqol, estimate


def get_metrics(clean, estimate, sr, filename):
    clean = clean.squeeze(dim=1)
    estimate = estimate.squeeze(dim=1)
    estimate_numpy = estimate.numpy()
    clean_numpy = clean.numpy()
    pesq = get_pesq(clean_numpy, estimate_numpy, sr=sr)
    stoi = get_stoi(clean_numpy, estimate_numpy, sr=sr)
    snr = get_snr(clean_numpy, estimate_numpy)
    lsd = get_lsd(clean, estimate).item()
    sisnr = get_sisnr(clean_numpy, estimate_numpy)
    visqol = get_visqol(clean_numpy, estimate_numpy, filename, sr)
    return pesq, stoi, snr, lsd, sisnr, visqol


def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    pesq_val = 0
    if sr not in [8000, 16000]:
        return pesq_val
    for i in range(len(ref_sig)):
        mode = 'wb' if sr == 16000 else 'nb'
        tmp = pesq(sr, ref_sig[i], out_sig[i], mode)  # from pesq
        pesq_val += tmp
    return pesq_val


def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    for i in range(len(ref_sig)):
        stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=True)
    return stoi_val


# get_snr and get_lsd are taken from: https://github.com/nanahou/metric/blob/master/measure_SNR_LSD.py
def get_snr(ref_sig, out_sig):
    """
       Compute SNR (signal to noise ratio)
       Arguments:
           out_sig: vector (torch.Tensor), enhanced signal [B,T]
           ref_sig: vector (torch.Tensor), reference signal(ground truth) [B,T]
    """

    ref = np.power(ref_sig, 2)
    diff = np.power(out_sig - ref_sig, 2)

    ratio = np.sum(ref, axis=-1) / np.sum(diff, axis=-1)
    value = 10 * np.log10(ratio)

    return value


def lsd(pred, target):
    sp = torch.log10(self.stft(pred).square().clamp(1e-8))
    st = torch.log10(self.stft(target).square().clamp(1e-8))
    return (sp - st).square().mean(dim=1).sqrt().mean()

def get_lsd(ref_sig, out_sig):
    """
       Compute LSD (log spectral distance)
       Arguments:
           out_sig: vector (torch.Tensor), enhanced signal [B,T]
           ref_sig: vector (torch.Tensor), reference signal(ground truth) [B,T]
    """

    fft_size = 1024
    shift_size = 120
    win_length = 600
    window = torch.hann_window(win_length)

    X = torch.log(torch.pow(torch.abs(stft(out_sig, fft_size, shift_size, win_length, window)),2))
    Y = torch.log(torch.pow(torch.abs(stft(ref_sig, fft_size, shift_size, win_length, window)), 2))


    diff = torch.pow(X - Y, 2)
    sum_freq = torch.sqrt(torch.sum(diff, dim=-1) / diff.size(-1))
    value = torch.sum(sum_freq, dim=-1) / sum_freq.size(-1)

    return value


def get_sisnr(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    B, T = ref_sig.shape
    ref_sig = ref_sig - np.mean(ref_sig, axis=1).reshape(B, 1)
    out_sig = out_sig - np.mean(out_sig, axis=1).reshape(B, 1)
    ref_energy = (np.sum(ref_sig ** 2, axis=1) + eps).reshape(B, 1)
    proj = (np.sum(ref_sig * out_sig, axis=1).reshape(B, 1)) * \
           ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2, axis=1) / (np.sum(noise ** 2, axis=1) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr.mean()


# from: https://github.com/eagomez2/upf-smc-speech-enhancement-thesis/blob/main/src/utils/evaluation_process.py
def get_visqol(ref_sig, out_sig, filename, sr):
    tmp_reference = f"{filename}_ref.wav"
    tmp_estimation = f"{filename}_est.wav"

    reference_abs_path = os.path.abspath(tmp_reference)
    estimation_abs_path = os.path.abspath(tmp_estimation)

    tfm = sox.Transformer()
    tfm.convert(bitdepth=16)
    ref_sig = np.transpose(ref_sig)
    out_sig = np.transpose(out_sig)
    try:
        tfm.build_file(input_array=ref_sig, sample_rate_in=sr, output_filepath=reference_abs_path)
        tfm.build_file(input_array=out_sig, sample_rate_in=sr, output_filepath=estimation_abs_path)

        visqol_cmd = ("cd " + VISQOL_PATH +
                      "./bazel-bin/visqol "
                      f"--reference_file {reference_abs_path} "
                      f"--degraded_file {estimation_abs_path} "
                      f"--use_speech_mode")

        visqol = subprocess.run(visqol_cmd, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # parse stdout to get the current float value
        visqol = visqol.stdout.decode("utf-8").split("\t")[-1].replace("\n", "")
        visqol = float(visqol)

    except Exception as e:
        logger.info(f'failed to get visqol of {filename}')
        logger.info(str(e))
        visqol = 0

    else:
        # remove files to avoid filling space storage
        os.remove(reference_abs_path)
        os.remove(estimation_abs_path)

    return visqol