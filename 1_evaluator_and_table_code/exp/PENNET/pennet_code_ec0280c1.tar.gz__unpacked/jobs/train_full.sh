#!/bin/bash
# PENNET full run: 100k steps (batch 512, 1 GPU), auto-resume from $E/run/ckpt_latest.pt (gpuidle REQUEUE-safe), then
# greedy decoding of val1000 (subset seed 0) and test/gold x rendered/dataset for every milestone. Idempotent.
# Submitted by smoke_driver.sh with: sbatch --export=ALL,E=...,C=... (partition/GPU/time/exclude on the sbatch line).
set -u -o pipefail
U=$U; SIF=$U/images/torch27.sif; VENV=$U/venvs/e0; D=$U/data/ouhdl_e0_v3
R=$E/run; EV=$E/eval; mkdir -p $R $EV/preds
# GPU identity: never choose device ids ourselves; use what Slurm allocated (some nodes, e.g. <node>, do not cgroup-
# constrain GPUs). PCI bus order so CUDA index == nvidia-smi index == Slurm's numbering.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  [ -n "${SLURM_JOB_GPUS:-}" ] || { echo "no Slurm GPU allocation visible" >> $R/job_history.txt; exit 3; }
  export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
fi
GPU=$(nvidia-smi -i ${CUDA_VISIBLE_DEVICES%%,*} --query-gpu=index,name,compute_cap,pci.bus_id,uuid --format=csv,noheader | head -1)
echo "$(date -Is) job=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-} gpu=[$GPU] phase=start" >> $R/job_history.txt
RUN="apptainer exec --nv -B $SCRATCH_ROOT $SIF env PYTHONPATH=$C OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $VENV/bin/python"
$RUN -c "import torch,sys; c=torch.cuda.get_device_capability(0); print(torch.cuda.get_device_name(0), c); sys.exit(0 if c[0]>=8 else 3)" || { echo "GPU capability < 8.0" >> $R/job_history.txt; exit 3; }
if [ ! -f $R/training_complete.json ]; then
  $RUN $C/pennet_train.py --train-npz $D/e0_train.npz --run-dir $R --condition rendered --steps 100000 --batch 512 \
      --lr 1e-3 --clip 0.2 --ckpt-every 1000 --milestone-every 10000 --log-every 100 --seed 20261016
  rc=$?
  echo "$(date -Is) job=${SLURM_JOB_ID:-none} train_rc=$rc" >> $R/job_history.txt
  [ -f $R/training_complete.json ] || { echo "training not complete (rc $rc)"; exit 1; }
fi
for W in $R/milestones/ms_*.pt; do
  k=$((10#$(basename $W .pt | sed 's/ms_//')))
  P=$EV/preds/val1000_rendered_ms${k}_preds.npz
  if [ ! -f $P ]; then
    $RUN $C/pennet_decode.py --weights $W --npz $D/e0_val.npz --condition rendered --subset 1000 --subset-seed 0 --out $P.tmp.npz \
      && mv $P.tmp.npz $P || exit 4
  fi
  for S in test gold; do for COND in rendered dataset; do
    P=$EV/preds/${S}_${COND}_ms${k}_preds.npz
    [ -f $P ] && continue
    $RUN $C/pennet_decode.py --weights $W --npz $D/e0_${S}.npz --condition $COND --out $P.tmp.npz && mv $P.tmp.npz $P || exit 4
  done; done
done
n=$(ls $EV/preds/*_preds.npz | wc -l)
echo "$(date -Is) job=${SLURM_JOB_ID:-none} decode_files=$n (expected 50)" >> $R/job_history.txt
[ "$n" -eq 50 ] || exit 5
date -Is > $R/decode_complete.txt
echo "$(date -Is) job=${SLURM_JOB_ID:-none} phase=done" >> $R/job_history.txt
