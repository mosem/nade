#!/bin/bash

. /cs/labs/adiyoss/moshemandel/speech-venv/bin/activate

python train.py \
  dset=single-8-16 \
  experiment=seanet \
  loss='' \
  dummy=nade-adv-not-normalized-masks- \
  experiment.lr_sr=8000 \
  experiment.hr_sr=16000 \
  wandb.tags=['nade','adv'] \
  epochs=170 \
