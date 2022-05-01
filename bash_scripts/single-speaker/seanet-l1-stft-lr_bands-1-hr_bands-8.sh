#!/bin/bash

. /cs/labs/adiyoss/moshemandel/speech-venv/bin/activate

python train.py \
  dset=single-8-16 \
  experiment=seanet \
  experiment.adversarial=False \
  dummy=nade-l1-stft-pyr-lr_n_bands-2-annealing- \
  experiment.lr_sr=8000 \
  experiment.hr_sr=16000 \
  wandb.tags=['nade','l1-stft','pyr'] \
  epochs=170 \
  restart=true \
  experiment.n_bands=8 \
  experiment.lr_n_bands=1 \
  experiment.seanet.in_channels=17 \
  experiment.seanet.out_channels=8 \

