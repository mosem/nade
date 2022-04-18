# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import json
import logging
from pathlib import Path
import os
import time

import torchaudio.transforms
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import distrib, google_sheets_logger
from src.data import PrHrSet
from src.enhance import enhance
from src.evaluate import evaluate
from src.log_results import log_results
from src.metrics import get_sisnr
from src.models.modules import MultiPeriodDiscriminator, MultiScaleDiscriminator, discriminator_loss, feature_loss, \
    generator_loss
from src.pyramid_loss import PyramidLoss
from src.sisnr_loss import SisnrLoss
from src.charbonnier_loss import Charbonnier_loss
from src.stft_loss import MultiResolutionSTFTLoss
from src.utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress
from src.augment import Augment

from torchaudio.functional import resample

logger = logging.getLogger(__name__)

SERIALIZE_KEY_MODELS = 'models'
SERIALIZE_KEY_OPTIMIZERS = 'optimizers'
SERIALIZE_KEY_HISTORY = 'history'
SERIALIZE_KEY_STATE = 'state'
SERIALIZE_KEY_BEST_STATES = 'best_states'
SERIALIZE_KEY_ARGS = 'args'

GENERATOR_KEY = 'generator'
GENERATOR_OPTIMIZER_KEY = 'generator_optimizer'

METRICS_KEY_EVALUATION_LOSS = 'evaluation_loss'
METRICS_KEY_BEST_LOSS = 'best_loss'

METRICS_KEY_PESQ = 'Average pesq'
METRICS_KEY_STOI = 'Average stoi'
METRICS_KEY_LSD = 'Average lsd'
METRICS_KEY_SISNR = 'Average sisnr'
METRICS_KEY_VISQOL = 'Average visqol'


class Solver(object):
    def __init__(self, data, models, optimizers, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']

        self.adversarial_mode = 'adversarial' in args.experiment and args.experiment.adversarial

        self.models = models
        self.dmodels = {k: distrib.wrap(model) for k, model in models.items()}
        self.model = self.models['generator']
        self.dmodel = self.dmodels['generator']

        lr_lambda_fn = lambda epoch: args.scheduler_factor ** epoch


        self.optimizers = optimizers
        self.optimizer = optimizers['optimizer']
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda_fn)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_factor)
        if 'disc_optimizer' in optimizers:
            self.disc_optimizer = optimizers['disc_optimizer']
            # disc_lr_lambda_fn = lambda epoch: args.scheduler_factor ** epoch
            # self.disc_scheduler = torch.optim.lr_scheduler.LambdaLR(self.disc_optimizer, lr_lambda=disc_lr_lambda_fn)
            # self.disc_scheduler = torch.optim.lr_scheduler.StepLR(self.disc_optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_factor)
        else:
            self.disc_optimizer = None


        # Training config
        self.device = args.device
        logger.info(f'device: {self.device}')
        self.epochs = args.epochs

        # Checkpoints
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        self.checkpoint = args.checkpoint
        if self.checkpoint:
            self.checkpoint_file = Path(args.checkpoint_file)
            self.best_file = Path(args.best_file)
            logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.history_file = args.history_file

        self.best_states = None
        self.restart = args.restart
        self.history = []  # Keep track of loss
        self.samples_dir = args.samples_dir  # Where to save samples
        self.num_prints = args.num_prints  # Number of times to log per epoch
        self.args = args
        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                  factor_mag=args.stft_mag_factor).to(self.device)
        self.sisnrloss = SisnrLoss()
        self.charbonnier_loss = Charbonnier_loss()
        if 'pyramid_loss' in self.args.experiment and self.args.experiment.pyramid_loss and self.args.experiment.model == 'interponet':
            self.pyramid_loss = PyramidLoss(self.args)
        if 'discriminator_model' in self.args.experiment and self.args.experiment.discriminator_model == 'hifi':
            self.melspec_transform = torchaudio.transforms.MelSpectrogram(
                                            self.args.experiment.hr_sr,
                                            n_fft=self.args.experiment.mel_loss_n_fft,
                                            n_mels=self.args.experiment.mel_loss_n_mels,
                                            hop_length=self.args.experiment.mel_loss_hop_length,
                                            win_length=self.args.experiment.mel_loss_win_length).to(self.device)

        # scale_factor = int(args.experiment.hr_sr / args.experiment.lr_sr)
        self.hr_augment = Augment(args.experiment.hr_sr, args.experiment.n_bands).to(self.device)
        # self.lr_augment = Augment(args.experiment.lr_sr, int(args.experiment.n_bands/scale_factor))

        self._reset()

    def _copy_models_states(self):
        states = {}
        for name, model in self.models.items():
            states[name] = copy_state(model.state_dict())
        return states

    def _serialize_models(self):
        serialized_models = {}
        for name, model in self.models.items():
            serialized_models[name] = serialize_model(model)
        return serialized_models

    def _serialize_optimizers(self):
        serialized_optimizers = {}
        for name, optimizer in self.optimizers.items():
            serialized_optimizers[name] = optimizer.state_dict()
        return serialized_optimizers

    def _serialize(self):
        package = {}
        package[SERIALIZE_KEY_MODELS] = self._serialize_models()
        package[SERIALIZE_KEY_OPTIMIZERS] = self._serialize_optimizers()
        package[SERIALIZE_KEY_HISTORY] = self.history
        package[SERIALIZE_KEY_BEST_STATES] = self.best_states
        package[SERIALIZE_KEY_ARGS] = self.args
        tmp_path = str(self.checkpoint_file) + ".tmp"
        torch.save(package, tmp_path)
        # renaming is sort of atomic on UNIX (not really true on NFS)
        # but still less chances of leaving a half written checkpoint behind.
        os.rename(tmp_path, self.checkpoint_file)

        # Saving only the latest best model.
        models = package[SERIALIZE_KEY_MODELS]
        for model_name, best_state in package[SERIALIZE_KEY_BEST_STATES].items():
            models[model_name][SERIALIZE_KEY_STATE] = best_state
            model_filename = model_name + '_' + self.best_file.name
            tmp_path = os.path.join(self.best_file.parent, model_filename) + ".tmp"
            torch.save(models[model_name], tmp_path)
            model_path = Path(self.best_file.parent / model_filename)
            os.rename(tmp_path, model_path)


    def _load(self, package, load_best=False):
        if load_best:
            for name, model_package in package['best_states']['models'].items():
                self.models[name].load_state_dict(model_package['state'])
        else:
            for name, model_package in package['models'].items():
                self.models[name].load_state_dict(model_package['state'])
            for name, opt_package in package['optimizers'].items():
                self.optimizers[name].load_state_dict(opt_package)

    def _reset(self):
        """_reset."""
        load_from = None
        load_best = False
        keep_history = True
        # Reset
        if self.checkpoint and self.checkpoint_file.exists() and not self.restart:
            load_from = self.checkpoint_file
        elif self.continue_from:
            load_from = self.continue_from
            load_best = self.args.continue_best
            keep_history = self.args.keep_history

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            self._load(package, load_best)
            if keep_history:
                self.history = package[SERIALIZE_KEY_HISTORY]
            self.best_states = package[SERIALIZE_KEY_BEST_STATES]


    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")
        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k.capitalize()}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch + 1}: {info}")

        logger.info('-' * 70)
        logger.info("Trainable Params:")
        for name, model in self.models.items():
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mb = n_params * 4 / 2 ** 20
            logger.info(f"{name}: parameters: {n_params}, size: {mb} MB")

        for epoch in range(len(self.history), self.epochs):
            # Train one epoch
            self.model.train()
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            losses = self._run_one_epoch(epoch)
            logger_msg = f'Train Summary | End of Epoch {epoch + 1} | Time {time.time() - start:.2f}s | ' \
                         + ' | '.join([f'{k} Loss {v:.5f}' for k, v in losses.items()])
            logger.info(bold(logger_msg))
            losses = {k + '_loss': v for k, v in losses.items()}

            if self.cv_loader:
                # Cross validation
                logger.info('-' * 70)
                logger.info('Cross validation...')
                self.model.eval()
                with torch.no_grad():
                    valid_losses = self._run_one_epoch(epoch, cross_valid=True)
                evaluation_loss = valid_losses['generator']
                logger_msg = f'Validation Summary | End of Epoch {epoch + 1} | Time {time.time() - start:.2f}s | ' \
                             + ' | '.join([f'{k} Valid Loss {v:.5f}' for k, v in valid_losses.items()])
                logger.info(bold(logger_msg))
                valid_losses = {'valid_' + k + '_loss': v for k, v in valid_losses.items()}
            else:
                valid_losses = {}
                evaluation_loss = 0


            best_loss = min(pull_metric(self.history, 'valid') + [evaluation_loss])
            metrics = {**losses, **valid_losses, METRICS_KEY_EVALUATION_LOSS: evaluation_loss,
                       METRICS_KEY_BEST_LOSS: best_loss}
            # Save the best model
            if evaluation_loss == best_loss:
                logger.info(bold('New best valid loss %.4f'), evaluation_loss)
                self.best_states = self._copy_models_states()

            # evaluate and enhance samples every 'eval_every' argument number of epochs
            # also evaluate on last epoch
            if ((epoch + 1) % self.eval_every == 0 or epoch == self.epochs - 1) and self.tt_loader:

                # Evaluate on the testset
                logger.info('-' * 70)
                logger.info('Evaluating on the test set...')
                # We switch to the best known model for testing
                with swap_state(self.model, self.best_states[GENERATOR_KEY]):
                    # enhance some samples
                    logger.info('Enhance and save samples...')
                    enhanced_filenames = enhance(self.tt_loader, self.model, self.args)
                    enhanced_dataset = PrHrSet(self.args.samples_dir, enhanced_filenames)
                    enhanced_dataloader = DataLoader(enhanced_dataset, batch_size=1, shuffle=False)

                    pesq, stoi, lsd, sisnr, visqol = evaluate(self.args, enhanced_dataloader, epoch)

                    if epoch == self.epochs - 1 and self.args.log_results:
                        # log results at last epoch
                        google_sheets_logger.log(self.args.experiment.name, {'pesq': pesq, 'stoi': stoi, 'lsd': lsd,
                                                                             'sisnr': sisnr, 'visqol': visqol},
                                                 self.args.experiment.lr_sr, self.args.experiment.hr_sr)
                        log_results(self.args, enhanced_dataloader, epoch)


                metrics.update({METRICS_KEY_PESQ: pesq, METRICS_KEY_STOI: stoi, METRICS_KEY_LSD: lsd,
                                METRICS_KEY_SISNR: sisnr, METRICS_KEY_VISQOL: visqol})



            wandb.log(metrics, step=epoch)
            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    self._serialize()
                    logger.debug("Checkpoint saved to %s", self.checkpoint_file.resolve())


    def _run_one_epoch(self, epoch, cross_valid=False):

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # get a different order for distributed training, otherwise this will get ignored
        data_loader.epoch = epoch

        label = ["Train", "Valid"][cross_valid]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        for i, data in enumerate(logprog):
            lr, hr = [x.to(self.device) for x in data]
            bands = self.hr_augment(hr)
            masks = torch.zeros_like(bands)
            input = torch.cat([lr,bands,masks],dim=1)
            pr = self.dmodel(input, hr.shape[-1])

            if self.adversarial_mode:
                if self.args.experiment.discriminator_model == 'hifi':
                    loss, discriminator_loss = self._get_hifi_adversarial_loss(hr, pr)
                else:
                    loss, discriminator_loss = self._get_melgan_adversarial_loss(hr, pr)
            else:
                loss = self._get_loss(hr, pr)

            # optimize model in training mode
            if not cross_valid:
                self._optimize(loss, pyramid_losses=None)
                if self.disc_optimizer is not None:
                    self._optimize_adversarial(discriminator_loss)

            logprog.update(loss=format(loss / (i + 1), ".5f"))
            # Just in case, clear some memory
            del pr, bands, masks

        avg_losses = {'generator': loss.item() / (i + 1)}
        if self.adversarial_mode:
            avg_losses.update({'discriminator': discriminator_loss.item() / (i + 1)})
        return avg_losses


    def _get_loss(self, hr, pr):
        loss = 0
        with torch.autograd.set_detect_anomaly(True):
            if self.args.loss == '':
                pass
            elif self.args.loss == 'l1':
                loss = F.l1_loss(hr, pr)
            elif self.args.loss == 'l2':
                loss = F.mse_loss(hr, pr)
            elif self.args.loss == 'sisnr':
                loss = self.sisnrloss(hr.squeeze(dim=1), pr.squeeze(dim=1))
            elif self.args.loss == 'stft_only':
                pass
            elif self.args.loss == 'charbonnier':
                loss = self.charbonnier_loss(hr, pr)
            else:
                raise ValueError(f'Invalid loss {self.args.loss}')
            # MultiResolution STFT loss
            if self.args.stft_loss:
                sc_loss, mag_loss = self.mrstftloss(pr.squeeze(1), hr.squeeze(1))
                loss += sc_loss + mag_loss


        return loss

    def _get_melgan_adversarial_loss(self, hr, pr):

        discriminator = self.dmodels['melgan']


        discriminator_fake_detached = discriminator(pr.detach())
        discriminator_real = discriminator(hr)
        discriminator_fake = discriminator(pr)

        total_loss_discriminator = self._get_melgan_discriminator_loss(discriminator_fake_detached, discriminator_real)
        generator_loss = self._get_melgan_generator_loss(discriminator_fake, discriminator_real)

        return generator_loss, total_loss_discriminator


    def _get_melgan_discriminator_loss(self, discriminator_fake, discriminator_real):
        discriminator_loss = 0
        for scale in discriminator_fake:
            discriminator_loss += F.relu(1 + scale[-1]).mean()

        for scale in discriminator_real:
            discriminator_loss += F.relu(1 - scale[-1]).mean()
        return discriminator_loss

    def _get_melgan_generator_loss(self, discriminator_fake, discriminator_real):
        generator_loss = 0
        for scale in discriminator_fake:
            generator_loss += F.relu(1 - scale[-1]).mean()
            # generator_loss += -scale[-1].mean()

        features_loss = 0
        features_weights = 4.0 / (self.args.experiment.discriminator.n_layers + 1)
        discriminator_weights = 1.0 / self.args.experiment.discriminator.num_D
        weights = discriminator_weights * features_weights

        for i in range(self.args.experiment.discriminator.num_D):
            for j in range(len(discriminator_fake[i]) - 1):
                features_loss += weights * F.l1_loss(discriminator_fake[i][j], discriminator_real[i][j].detach())

        return generator_loss + self.args.experiment.features_loss_lambda * features_loss


    def _get_hifi_adversarial_loss(self, hr, pr):
        mpd = self.dmodels['mpd']
        msd = self.dmodels['msd']

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(hr, pr.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MSD
        y_ds_hat_r, y_ds_hat_g, _, _ = msd(hr, pr.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        total_loss_discriminator = loss_disc_s + loss_disc_f

        # L1 Mel-Spectrogram Loss
        pr_mel = self.melspec_transform(pr)
        hr_mel = self.melspec_transform(hr)
        loss_mel = F.l1_loss(hr_mel, pr_mel) * 45

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(hr, pr)
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(hr, pr)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
        total_loss_generator = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        return total_loss_generator, total_loss_discriminator


    def _optimize(self, loss, pyramid_losses = None):
        self.optimizer.zero_grad()
        if pyramid_losses is not None:
            for p_loss in pyramid_losses.values():
                loss += p_loss
                # p_loss.backward(retain_graph=True)
        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()

    def _optimize_adversarial(self, discriminator_loss):
        self.disc_optimizer.zero_grad()
        discriminator_loss.backward()
        self.disc_optimizer.step()
        # self.disc_scheduler.step()