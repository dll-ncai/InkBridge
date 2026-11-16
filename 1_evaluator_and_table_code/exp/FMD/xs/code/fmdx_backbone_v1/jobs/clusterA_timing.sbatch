#!/bin/bash
#SBATCH -J fmdx_bb_timing
#SBATCH -p gpuidle
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx[001-005],gpu[004-005,013-014,017-023,119-148]
#SBATCH --cpus-per-task=8
#SBATCH --mem=46G
#SBATCH --time=01:30:00
#SBATCH -o $U/exp/FMD/xs/logs/build/timing_%j.out
# Step time of the real arm layout (A6: 512 sequences / step; layout rule of arm.sh) on whatever cc>=8 GPU this lands on:
# 25 steps no-FM warm and 25 steps REPA (dummy D=1024 targets) on the first 3,000 LUO units (smoke render prefix).
set -u
source $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh
SM=${1:?smoke dir with luo_first3000_render224.npy + dummy targets}; T=$X/runs/timing/$SLURM_JOB_ID; mkdir -p $T
MEMG=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' '); MEMG=$((MEMG / 1024))
if [ $MEMG -ge 40 ]; then PERGPU=128; else PERGPU=32; fi; ACC=$((512 / PERGPU))
echo "gpu $(gpu_names) mem $MEMG layout $PERGPU x $ACC"
for k in nofm repa; do
  EX=(); [ $k = repa ] && EX=(--fm dinov3-vitl16 --repa-targets $SM/dummy_targets_dinov3.npy --repa-unit-ids $SM/dummy_targets_unit_ids.npy)
  $PY $BB/bb_train.py --arm-id timing_$k --run-dir $T/$k --regime warm --init-ckpt $LUO_CKPT --train-npz $LUO_NPZ --render-npy $SM/luo_first3000_render224.npy \
     --max-units 3000 --batch-size $PERGPU --grad-accum $ACC --train-steps 25 --save-every 1000 --resume-every 1000 --val-every 0 --log-every 5 --num-workers 6 "${EX[@]}" > $T/$k.log 2>&1
  python3 -c "
import json; r=[json.loads(l) for l in open('$T/$k/train_log.jsonl') if '\"train\"' in l]; s=[x['sec_per_step'] for x in r if x['step']>=10]
print('TIMING $k', '$(gpu_names)', 'layout $PERGPU x $ACC', 'sec_per_step', round(sum(s)/len(s),3) if s else None, 'max_alloc_gb', r[-1].get('cuda_max_alloc_gb'))" | tee -a $T/timing.txt
done
