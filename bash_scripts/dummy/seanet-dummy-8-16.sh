#!/bin/bash

. /cs/labs/adiyoss/moshemandel/speech-venv/bin/activate

python train.py \
  dset=dummy-8-16 \
  experiment=seanet \
  loss='' \
  wandb.mode=disabled \
  restart=true \
  epochs=1 \
