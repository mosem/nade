import argparse
import os
import logging

import torch
import wandb


from torchaudio.transforms import Spectrogram
from torch.utils.data import DataLoader

from src import distrib
from src.data import PrHrSet
from src.log_results import log_results
from src.metrics import run_metrics
from src.utils import LogProgress, bold, convert_spectrogram_to_heatmap

logger = logging.getLogger(__name__)

WANDB_PROJECT_NAME = 'Bandwidth Extension'
WANDB_ENTITY = 'huji-dl-audio-lab'


def evaluate(args, data_loader, epoch):
    total_pesq = 0
    total_stoi = 0
    total_lsd = 0
    total_sisnr = 0
    total_visqol = 0
    total_cnt = 0

    files_to_log = []
    wandb_n_files_to_log = args.wandb.n_files_to_log if 'wandb' in args else args.wandb_n_files_to_log
    hr_sr = args.experiment.hr_sr if 'experiment' in args else args.hr_sr

    with torch.no_grad():
        iterator = LogProgress(logger, data_loader, name="Eval estimates")
        for i, data in enumerate(iterator):
            # Get batch data
            lr, hr, pr, filename = data
            filename = filename[0]
            logger.info(f'evaluating on {filename}')
            if wandb_n_files_to_log == -1 or len(files_to_log) < wandb_n_files_to_log:
                files_to_log.append(filename)

            if args.device != 'cpu':
                hr = hr.cpu()
                pr = pr.cpu()

            pesq_i, stoi_i, snr_i, lsd_i, sisnr_i, visqol_i, estimate_i = run_metrics(hr, pr, args, filename)
            if filename in files_to_log:
                log_to_wandb(estimate_i, pesq_i, stoi_i, snr_i, lsd_i, sisnr_i, visqol_i,
                             filename, epoch, hr_sr)
            total_pesq += pesq_i
            total_stoi += stoi_i
            total_lsd += lsd_i
            total_sisnr += sisnr_i
            total_visqol += visqol_i

            total_cnt += 1

    metrics = [total_pesq, total_stoi, total_lsd, total_sisnr, total_visqol]
    avg_pesq, avg_stoi, avg_lsd, avg_sisnr, avg_visqol = distrib.average([m / total_cnt for m in metrics], total_cnt)
    logger.info(bold(
        f'{args.experiment.name}, {args.experiment.lr_sr}->{args.experiment.hr_sr}. Test set performance:PESQ={avg_pesq}, STOI={avg_stoi}, LSD={avg_lsd}, SISNR={avg_sisnr} ,VISQOL={avg_visqol}.'))
    return avg_pesq, avg_stoi, avg_lsd, avg_sisnr, avg_visqol


def log_to_wandb(signal, pesq, stoi, snr, lsd, sisnr, visqol, filename, epoch, sr):
    spectrogram_transform = Spectrogram()
    enhanced_spectrogram = spectrogram_transform(signal).log2()[0, :, :].numpy()
    enhanced_spectrogram_wandb_image = wandb.Image(convert_spectrogram_to_heatmap(enhanced_spectrogram),
                                                   caption=filename)
    enhanced_wandb_audio = wandb.Audio(signal.squeeze().numpy(), sample_rate=sr, caption=filename)
    wandb.log({f'test samples/{filename}/pesq': pesq,
               f'test samples/{filename}/stoi': stoi,
               f'test samples/{filename}/snr': snr,
               f'test samples/{filename}/lsd': lsd,
               f'test samples/{filename}/sisnr': sisnr,
               f'test samples/{filename}/visqol': visqol,
               f'test samples/{filename}/spectrogram': enhanced_spectrogram_wandb_image,
               f'test samples/{filename}/audio': enhanced_wandb_audio},
              step=epoch)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('samples_dir', type=str)
    parser.add_argument('--device', nargs="?", default='cpu', type=str, choices=['cpu', 'cuda'])
    parser.add_argument('--lr_sr', nargs="?", default=8000, type=int)
    parser.add_argument('--hr_sr', nargs="?", default=16000, type=int)
    parser.add_argument('--num_workers', nargs="?", default=1, type=int)
    parser.add_argument('--wandb_mode', nargs="?", default='online', type=str)
    parser.add_argument('--wandb_n_files_to_log', nargs="?", default=10, type=int)
    parser.add_argument('--n_bins', nargs="?", default=5, type=int)
    parser.add_argument('--log_results', action='store_false')


    return parser

def update_args(args):
    d = vars(args)
    experiment = argparse.Namespace()
    experiment.name = 'nuwave'
    experiment.lr_sr = args.lr_sr
    experiment.hr_sr = args.hr_sr
    d['experiment'] = experiment


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    samples_dir = args.samples_dir
    print(args)
    update_args(args)
    print(args)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)


    wandb_mode = os.environ['WANDB_MODE'] if 'WANDB_MODE' in os.environ.keys() else args.wandb_mode
    wandb.init(mode=wandb_mode, project=WANDB_PROJECT_NAME, entity=WANDB_ENTITY, config=args, group='nuwave')

    data_set = PrHrSet(samples_dir)
    dataloader = DataLoader(data_set, batch_size=1, shuffle=False)
    avg_pesq, avg_stoi, avg_lsd, avg_sisnr, avg_visqol = evaluate(args, dataloader, epoch=0)

    log_results(args, dataloader, epoch=0)

    print(f'pesq: {avg_pesq}, stoi: {avg_stoi}, lsd: {avg_lsd}, sisnr: {avg_sisnr}, visqol: {avg_visqol}')
    print('done evaluating.')