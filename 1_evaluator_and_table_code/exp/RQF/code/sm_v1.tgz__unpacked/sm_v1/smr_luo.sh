#!/bin/bash
# SM-R on plain LUO ms6: copy of rqf_v1/luo_ft.sh with REPLAY=1 QU=0.5 (frame_code_a44e78b0 e0_train.py --replay-npz m5_luo_train.npz
# --urdu-share 0.5: two-source batches, 50 % target units / 50 % replay units, seeded by the training seed) and all 7 test sets at every fraction.
# Usage: smr_luo.sh TAG FRAC SEED
set -uo pipefail; exec 2>&1
source $U/exp/RQF/code/sm_v1/sm_env.sh
TAG=$1; FRAC=$2; SEED=$3; INIT=$LUO_CKPT
REPO=$R/repos/nagamatsu; IMGX=$R/images/torch27.sif; VE=$R/venvs/e0
cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
awk -v c="$cc" 'BEGIN{exit !(c+0 >= 8.0)}' || { echo "GPU_CC_LT_8 $cc"; exit 4; }
MEMG=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' '); MEMG=$((MEMG / 1024))
if [ $MEMG -ge 40 ]; then export BATCH=128 ACCUM=4; else export BATCH=32 ACCUM=16; fi
export PYG="apptainer exec --nv -B $SCRATCH_ROOT $IMGX env CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$FCODE/lib:$REPO/src:$FCODE OMP_NUM_THREADS=4 PYTHONHASHSEED=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled $VE/bin/python"
export PYC="apptainer exec -B $SCRATCH_ROOT $IMGX env PYTHONPATH=$FCODE/lib:$REPO/src:$FCODE OMP_NUM_THREADS=1 PYTHONHASHSEED=0 $VE/bin/python"
export CODE=$FCODE STAGE=$R RUNROOT=$QR ARM=$TAG FRAC=$FRAC FRAME=identity REPLAY=1 QU=0.5 QU_END="" CUR_START="" GATE_STEP="" TRAIN_SEED=$SEED SEEDS="0 1 2" PROCS=${SLURM_CPUS_PER_TASK:-8} INIT_OVERRIDE=$INIT
[ -n "${SM_SMOKE_STEPS:-}" ] && export STEPS=$SM_SMOKE_STEPS SAVE=$SM_SMOKE_STEPS
bash $QC/frame_arm_rqf.sh || exit $?
RD=$QR/$TAG; BEST=$(cat $RD/eval/best_milestone.txt 2>/dev/null); [[ "$BEST" =~ ^[0-9]+$ ]] || { echo "$(utc) NO_BEST_MILESTONE" >> $RD/job_history.txt; exit 7; }
[ -n "${SM_SMOKE_STEPS:-}" ] && { echo "smoke: selection done ms$BEST"; exit 0; }
GPUN=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
echo "$(utc) final_sampling_start ms$BEST gpu=[$GPUN] host=$(hostname) slurm=${SLURM_JOB_ID:-x}" >> $RD/job_history.txt
$PYG $CCODE/ctrl_sample.py --label ${TAG}_ms${BEST} --run-dir exp/RQF/runs/$TAG --milestone $BEST --stage $R --out-dir $QP/$TAG --sets $SETS7 --frames identity --seeds 0 1 2 --batch-size 256 2>&1 | grep -av "sampling loop" | tail -3
rc=${PIPESTATUS[0]}; echo "$(utc) final_sampling_end rc=$rc gpu=[$GPUN]" >> $RD/job_history.txt; [ $rc = 0 ] && touch $RD/SAMPLING_DONE; exit $rc
