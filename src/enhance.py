import os
import logging
import argparse
import sys

import torch
import torchaudio
from torch.utils.data import DataLoader
from concurrent.futures import ProcessPoolExecutor

from src.utils import LogProgress
from src.data import LrHrSet
from src.models.sinc import Sinc

from torchaudio.functional import resample



logger = logging.getLogger(__name__)


def get_random_mask(n_bands, device, t, T):
    p_max = 1.0
    p_min = 0.0
    alpha = 0.95
    p = max(p_min, p_max - (t - 1) * (p_max - p_min) / (alpha * (T - 1)))
    masks = torch.rand(size=(1, n_bands, 1), dtype=torch.float,
                          device=device)
    return (masks>p).int()


def gibbs_inference(model, lr_sig, hr_sig, args):
    """

    :param model:
    :param lr_sig:  [Batch-size, in_channels, Time]
                    in_channels: lr channel, hr_band_1,...,hr_band_n, masks_sentinels
    :param hr_sig:  [Batch-size, out_channels, Time]
                    out_channels: hr_band_1,...,hr_band_n
    :param args:
    :return:
    """
    n_steps = args.experiment.n_gibbs_steps
    n_bands = args.experiment.n_bands
    lr_n_bands = args.experiment.lr_n_bands
    with torch.no_grad():
        lr_channel = lr_sig[:,0:lr_n_bands,:]
        if args.experiment.cumulative:
            out, out_cumulative = model(lr_sig, hr_sig.shape[-1])
        else:
            out = model(lr_sig, hr_sig.shape[-1])
        prev_out = out

        for i in range(1, n_steps):
            masks = get_random_mask(n_bands,args.device,i,n_steps).expand_as(prev_out)
            flipped_masks = torch.zeros_like(masks)
            flipped_masks[masks==0] = 1

            tmp_masks = masks.detach().clone() # tmp masks is propagated via network and should not be used later
            next_input = torch.cat([lr_channel, tmp_masks * prev_out, tmp_masks], dim=1)
            if args.experiment.cumulative:
                tmp_out, tmp_out_cumulative = model(next_input, hr_sig.shape[-1])
            else:
                tmp_out = model(next_input, hr_sig.shape[-1])

            logger.info(f'{i}) masks: {masks[0, :, 0]}, binary?: {torch.all(sum(masks == 1, masks == 0)).item()}')

            # combine channels from previous output and current output
            prev_out = masks * prev_out + flipped_masks * tmp_out


    # return tmp_out

    if args.experiment.save_cumulative:
        return (tmp_out, tmp_out_cumulative)
    else:
        return tmp_out

    # return prev_out



def get_estimate(model, lr_sig, hr_sig, args):
    torch.set_num_threads(1)
    with torch.no_grad():
        estimate = gibbs_inference(model, lr_sig, hr_sig, args)
    return estimate


def write(wav, filename, sr):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def save_wavs_with_cumulative(pr_channels_sigs, pr_cumulative_sigs, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr, lr_n_bands):
    # Write result
    for lr, hr, pr_channels, pr, filename in zip(lr_sigs, hr_sigs, pr_channels_sigs, pr_cumulative_sigs, filenames):
        lr = lr[0:lr_n_bands, :]
        # logger.info(f'pr_channels shape: {pr_channels.shape}, pr shape: {pr.shape}, lr_shape: {lr.shape}')
        for j in range(lr.shape[0]):
            lr_j = resample(lr[j:j + 1, :], hr_sr, lr_sr)
            # logger.info(f'lr_j shape: {lr_j.shape}')
            write(lr_j, filename + f'_lr_{j}.wav', sr=lr_sr)

        for i in range(hr.shape[0]):
            write(hr[i:i + 1, :], filename + f'_hr_{i}.wav', sr=hr_sr)
            write(pr_channels[i:i + 1, :], filename + f'_pr_{i}.wav', sr=hr_sr)

        hr = torch.sum(hr, keepdim=True, dim=0)
        pr_sum = torch.sum(pr_channels, keepdim=True, dim=0)
        write(pr_sum, filename + "_pr_sum.wav", sr=hr_sr)

        lr = torch.sum(lr, keepdim=True, dim=0)
        lr = resample(lr, hr_sr, lr_sr)
        write(lr, filename + "_lr.wav", sr=lr_sr)
        write(hr, filename + "_hr.wav", sr=hr_sr)
        write(pr, filename + "_pr.wav", sr=hr_sr)



def save_wavs(processed_sigs, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr, lr_n_bands):
    # Write result
    for lr, hr, pr, filename in zip(lr_sigs, hr_sigs, processed_sigs, filenames):
        lr = lr[0:lr_n_bands, :]
        for j in range(lr.shape[0]):
            lr_j = resample(lr[j:j + 1, :], hr_sr, lr_sr)
            write(lr_j, filename + f'_lr_{j}.wav', sr=lr_sr)

        for i in range(hr.shape[0]):
            write(hr[i:i+1, :], filename + f'_hr_{i}.wav', sr=hr_sr)
            write(pr[i:i + 1,:], filename + f'_pr_{i}.wav', sr=hr_sr)

        hr = torch.sum(hr, keepdim=True, dim=0)
        pr = torch.sum(pr, keepdim=True, dim=0)

        lr = torch.sum(lr, keepdim=True, dim=0)
        lr = resample(lr, hr_sr, lr_sr)
        write(lr, filename + "_lr.wav", sr=lr_sr)
        write(hr, filename + "_hr.wav", sr=hr_sr)
        write(pr, filename + "_pr.wav", sr=hr_sr)


def _estimate_and_save(model, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr, args, lr_n_bands):
    estimate = get_estimate(model, lr_sigs, hr_sigs, args)
    if args.experiment.save_cumulative:
        estimate_channels, estimate_cumulative = estimate
        save_wavs_with_cumulative(estimate_channels, estimate_cumulative, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr,
                                  lr_n_bands)
    else:
        save_wavs(estimate, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr, lr_n_bands)


def enhance(dataloader, model, args):
    model.eval()

    os.makedirs(args.samples_dir, exist_ok=True)
    lr_sr = args.experiment.lr_sr if 'experiment' in args else args.lr_sr
    hr_sr = args.experiment.hr_sr if 'experiment' in args else args.hr_sr

    lr_n_bands = args.experiment.lr_n_bands

    total_filenames = []

    with ProcessPoolExecutor(args.num_workers) as pool:
        iterator = LogProgress(logger, dataloader, name="Generate enhanced files")
        pendings = []
        for i, data in enumerate(iterator):
            # Get batch data
            (lr_sigs, lr_paths), (hr_sigs, hr_paths) = data
            lr_sigs = lr_sigs.to(args.device)
            hr_sigs = hr_sigs.to(args.device)
            filenames = [os.path.join(args.samples_dir, os.path.basename(path).rsplit(".", 1)[0]) for path in lr_paths]
            total_filenames += [os.path.basename(path).rsplit(".", 1)[0] for path in lr_paths]
            if args.device == 'cpu' and args.num_workers > 1:
                pendings.append(
                    pool.submit(_estimate_and_save,
                                model, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr, args, lr_n_bands))
            else:
                # Forward
                estimate = get_estimate(model, lr_sigs, hr_sigs, args)
                if args.experiment.save_cumulative:
                    estimate_channels, estimate_cumulative = estimate
                    save_wavs_with_cumulative(estimate_channels, estimate_cumulative, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr, lr_n_bands)
                else:
                    save_wavs(estimate, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr, lr_n_bands)

            if i == args.enhance_samples_limit:
                break

        if pendings:
            print('Waiting for pending jobs...')
            for pending in LogProgress(logger, pendings, updates=5, name="Generate enhanced files"):
                pending.result()

    return total_filenames


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_dir', type=str)
    parser.add_argument('samples_dir', type=str)
    parser.add_argument('--model', nargs="?", default='sinc', type=str, choices=['sinc'])
    parser.add_argument('--device', nargs="?", default='cpu', type=str, choices=['cpu', 'cuda'])
    parser.add_argument('--lr_sr', nargs="?", default=8000, type=int)
    parser.add_argument('--hr_sr', nargs="?", default=16000, type=int)
    parser.add_argument('--stride', nargs="?", default=-1, type=float)
    parser.add_argument('--segment', nargs="?", default=-1, type=float)
    parser.add_argument('--num_workers', nargs="?", default=1, type=int)
    parser.add_argument('--batch_size', nargs="?", default=16, type=int)
    parser.add_argument('--enhance_samples_limit', nargs="?", default=-1, type=int)

    return parser


def get_model(args):
    if args.model == "sinc":
        return Sinc(args.lr_sr, args.hr_sr)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    print(args)
    json_dir = args.json_dir

    model = get_model(args)

    stride = args.stride if args.stride != -1 else None
    segment =  args.segment if args.segment != -1 else None


    data_set = LrHrSet(json_dir, args.lr_sr, args.hr_sr, stride, segment, with_path=True)
    dataloader = DataLoader(data_set, batch_size=1, shuffle=False)
    enhance(dataloader, model, args)

    print('done enhancing.')

