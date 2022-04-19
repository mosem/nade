#!/bin/bash

. /cs/labs/adiyoss/moshemandel/speech-venv/bin/activate

python train.py \
  dset=single-8-16 \
  experiment=seanet \
  experiment.adversarial=False \
  dummy=nade-l1-stft-not-normalized- \
  experiment.lr_sr=8000 \
  experiment.hr_sr=16000 \
  wandb.tags=['nade','l1-stft'] \
  epochs=170 \
  restart=true \
