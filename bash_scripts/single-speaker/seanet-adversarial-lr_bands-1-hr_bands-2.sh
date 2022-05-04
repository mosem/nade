#!/bin/bash

. /cs/labs/adiyoss/moshemandel/speech-venv/bin/activate

python train.py \
  dset=single-8-16 \
  experiment=seanet \
  dummy=nade-adversarial-annealing-grouping- \
  experiment.lr_sr=8000 \
  experiment.hr_sr=16000 \
  wandb.tags=['nade','l1-stft','pyr'] \
  epochs=170 \
  experiment.adversarial=True \
  experiment.n_bands=4 \
  experiment.lr_n_bands=2 \
  experiment.seanet.in_channels=10 \
  experiment.seanet.out_channels=4 \

