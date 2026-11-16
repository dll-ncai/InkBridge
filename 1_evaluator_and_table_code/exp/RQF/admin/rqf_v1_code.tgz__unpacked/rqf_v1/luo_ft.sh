#!/bin/bash
# RQF LUO-recipe fine-tune (Route A LUO-FT recipe = COMP-p01 recipe; frame identity, no replay, 10k steps, ckpt 1k, val1000 selection) from INIT, then sampling with ctrl_sample.py
# (same sampler as the COMP/LUO comparator rows), DDIM-50 eta 0 EMA, batch 256, seeds 0-2. p01: test_ouhdl + 6 non-Urdu sets; p10/p30: test_ouhdl.
# Usage: luo_ft.sh TAG INIT_CKPT FRAC SEED
set -uo pipefail; exec 2>&1
source $U/exp/RQF/code/rqf_v1/rqf_env.sh
TAG=$1; INIT=$2; FRAC=$3; SEED=$4
REPO=$R/repos/nagamatsu; IMGX=$R/images/torch27.sif; VE=$R/venvs/e0
case "$INIT" in @*) RN=${INIT#@}; BM=$(cat $QR/$RN/eval/best_milestone.txt 2>/dev/null) || { echo "SRC_RUN_NOT_SELECTED $RN"; exit 5; }; INIT=$QR/$RN/checkpoints/checkpoint_$(printf %06d $BM).pt;; esac
[ -s "$INIT" ] || { echo "INIT_MISSING $INIT"; exit 5; }
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
awk -v c="$cc" 'BEGIN{exit !(c+0 >= 8.0)}' || { echo "GPU_CC_LT_8 $cc"; exit 4; }
MEMG=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' '); MEMG=$((MEMG / 1024))
if [ $MEMG -ge 40 ]; then export BATCH=128 ACCUM=4; else export BATCH=32 ACCUM=16; fi   # 512 sequences per optimiser step either way
export PYG="apptainer exec --nv -B $SCRATCH_ROOT $IMGX env CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$FCODE/lib:$REPO/src:$FCODE OMP_NUM_THREADS=4 PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled $VE/bin/python"
export PYC="apptainer exec -B $SCRATCH_ROOT $IMGX env PYTHONPATH=$FCODE/lib:$REPO/src:$FCODE OMP_NUM_THREADS=1 PYTHONHASHSEED=0 $VE/bin/python"
export CODE=$FCODE STAGE=$R RUNROOT=$QR ARM=$TAG FRAC=$FRAC FRAME=identity REPLAY=0 GATE_STEP="" TRAIN_SEED=$SEED SEEDS="0 1 2" PROCS=${SLURM_CPUS_PER_TASK:-8} INIT_OVERRIDE=$INIT
bash $QC/frame_arm_rqf.sh
RD=$QR/$TAG; EV=$RD/eval; BEST=$(cat $EV/best_milestone.txt 2>/dev/null); [[ "$BEST" =~ ^[0-9]+$ ]] || { echo "$(utc) NO_BEST_MILESTONE" >> $RD/job_history.txt; exit 7; }
SETS="test_ouhdl"; { [ "$FRAC" = p01 ] || [ "$FRAC" = p10 ]; } && SETS="test_ouhdl $SIX"
GPUN=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$(utc) final_sampling_start ms$BEST gpu=[$GPUN] host=$(hostname) slurm=${SLURM_JOB_ID:-x}" >> $RD/job_history.txt
$PYG $CCODE/ctrl_sample.py --label ${TAG}_ms${BEST} --run-dir exp/RQF/runs/$TAG --milestone $BEST --stage $R --out-dir $QP/$TAG --sets $SETS --frames identity --seeds 0 1 2 --batch-size 256 2>&1 | grep -av "sampling loop" | tail -3
rc=${PIPESTATUS[0]}; echo "$(utc) final_sampling_end rc=$rc gpu=[$GPUN]" >> $RD/job_history.txt; exit $rc
