#!/bin/bash
# router336: 3 seeds per arm, 3 concurrent runs of 4 threads; arm natural first, then balanced.
R=$LOCAL/experiments/RV11/router336; cd $R
for arm in natural balanced; do
  for s in 0 1 2; do
    python code/rr336_train.py --pull pull --e0train $LOCAL/experiments/e0/data/e0_train.npz --ids p01_unit_ids.txt --out runs/${arm}_s${s} --arm $arm --seed $s --threads 4 > logs/train_${arm}_s${s}.log 2>&1 &
  done
  wait
done
date -u +%Y-%m-%dT%H:%M:%SZ > logs/run_all.done
