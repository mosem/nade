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



logger = logging.getLogger(__name__)

def get_estimate(model, lr_sig, hr_sig, args):
    torch.set_num_threads(1)
    with torch.no_grad():
        out = model(lr_sig, hr_sig.shape[-1])
        if 'experiment' in args and args.experiment.model == 'interponet':
            estimate = out['full_band']
        elif 'experiment' in args and args.experiment.model == 'interponet_2':
            estimate = out[-1]
        elif args.experiment.model == 'lapsrn':
            estimate = out[-1]
        else:
            estimate = out
    return estimate


def write(wav, filename, sr):
    # Normalize audio if it prevents clipping
    wav = wav / max(wav.abs().max().item(), 1)
    torchaudio.save(filename, wav.cpu(), sr)


def save_wavs(processed_sigs, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr):
    # Write result
    for lr, hr, pr, filename in zip(lr_sigs, hr_sigs, processed_sigs, filenames):
        write(lr, filename + "_lr.wav", sr=lr_sr)
        write(hr, filename + "_hr.wav", sr=hr_sr)
        write(pr, filename + "_pr.wav", sr=hr_sr)


def _estimate_and_save(model, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr, args):
    estimate = get_estimate(model, lr_sigs, hr_sigs, args)
    save_wavs(estimate, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr)


def enhance(dataloader, model, args):
    model.eval()

    os.makedirs(args.samples_dir, exist_ok=True)
    lr_sr = args.experiment.lr_sr if 'experiment' in args else args.lr_sr
    hr_sr = args.experiment.hr_sr if 'experiment' in args else args.hr_sr

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
                                model, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr, args))
            else:
                # Forward
                estimate = get_estimate(model, lr_sigs, hr_sigs, args)
                save_wavs(estimate, lr_sigs, hr_sigs, filenames, lr_sr, hr_sr)

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

