#!/bin/bash

# usage example: bash prep_egs_files.sh 16 24

. /cs/labs/adiyoss/moshemandel/speech-venv/bin/activate
cd /cs/labs/adiyoss/moshemandel/nade

lr=$1
hr=$2
mode=${3:-'single-speaker'}
n_samples_limit=${4:--1}
out_dir=$lr-$hr

if [[ $n_samples_limit -gt 0 ]]
then
  out_dir+="(${n_samples_limit})"
fi

echo "saving to $mode/${out_dir}"

tr_out=egs/vctk/$mode/$out_dir/tr
val_out=egs/vctk/$mode/$out_dir/val

lr_train_files=../data/vctk/$mode/train-files-$lr.txt
hr_train_files=../data/vctk/$mode/train-files-$hr.txt

lr_val_files=../data/vctk/$mode/val-files-$lr.txt
hr_val_files=../data/vctk/$mode/val-files-$hr.txt

mkdir -p $tr_out
mkdir -p $val_out

python -m src.prep_egs_files $lr_train_files $n_samples_limit > $tr_out/lr.json
python -m src.prep_egs_files $hr_train_files $n_samples_limit > $tr_out/hr.json

python -m src.prep_egs_files $lr_val_files $n_samples_limit > $val_out/lr.json
python -m src.prep_egs_files $hr_val_files $n_samples_limit > $val_out/hr.json