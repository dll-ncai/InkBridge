#!/bin/bash
#SBATCH --partition=gpuidle
#SBATCH --gres=gpu:a4000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/RQF/R3/logs/samp_%x_%j.out
# RQF R3 (PREREG s3): COMP-p01 non-Urdu sampling seeds 1,2 (seed 0 exists on A4000 in comparators_a4000). Derived from comp_resample.sbatch (FMDX monitor): comparator re-sampling on the CLUSTER-A gpuidle RTX A4000 (prereg A8.1). Same sampler as the 4090 comparator rows: ctrl_sample.py -> frame_sample.py (frame_code_a44e78b0),
# identity frame, DDIM-50 eta 0, EMA, batch 256. Usage: sbatch -J fxm_cmp_<JOB> [--nice N] comp_resample.sbatch <JOB>   JOB in LUO C34 C35 C36
set -uo pipefail; exec 2>&1
JOB=$1
B=$U; IMG=$B/images/torch27.sif; VENV=$B/venvs/e0; REPO=$B/repos/nagamatsu
CODE=$B/exp/CAND/code/frame_code_a44e78b0; CC=$B/exp/CAND/controls/code; MR=$B/exp/FMD/xs/runs/fmdx_monitor; OUT=$U/exp/RQF/R3/comp_a4000/$JOB
SIX="test_casia_zh test_casia_en test_hpl_tamil test_iam500 german_unipen_de_test german_onhw_all"
CP=$B/exp/CAND/method_round/COMP/runs
case $JOB in
  LUO) LABEL=LUO_ms6; RD=exp/M5/runs/luo; MS=6; EXP=ab6b178d032d90e5; S1="$SIX"; SD1="0 1 2"; S2=""; SD2="";;
  C34) LABEL=comp_p01_identity_s20261034_vps_ms3; RD=exp/FMD/xs/runs/fmdx_monitor/comparators_a4000/c34_run; MS=3; EXP=149396eda43623a4; S1="$SIX"; SD1="1 2"; S2=""; SD2="";;
  C35) LABEL=comp_p01_identity_s20261035_ms3; RD=exp/CAND/method_round/COMP/runs/comp_p01_identity_s20261035; MS=3; EXP=b5a0ac4bc48e3bb3; S1="$SIX"; SD1="1 2"; S2=""; SD2="";;
  C36) LABEL=comp_p01_identity_s20261036_ms4; RD=exp/CAND/method_round/COMP/runs/comp_p01_identity_s20261036; MS=4; EXP=2a7e476cd475b257; S1="$SIX"; SD1="1 2"; S2=""; SD2="";;
  *) echo bad job; exit 2;;
esac
mkdir -p $OUT; H=$OUT/job_history.txt; utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
CK=$B/$RD/checkpoints/checkpoint_$(printf %06d $MS).pt
SHA=$(sha256sum $CK | cut -c1-16); [ "$SHA" = "$EXP" ] || { echo "$(utc) CKPT_SHA_MISMATCH $SHA != $EXP" >> $H; exit 3; }
echo "$(utc) start job=$JOB slurm=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0} host=$(hostname) gpu=[$(nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader | tr '\n' ';')] ckpt=$CK sha16=$SHA frame_sample_sha=$(sha256sum $CODE/frame_sample.py | cut -c1-16) ctrl_sample_sha=$(sha256sum $CC/ctrl_sample.py | cut -c1-16)" >> $H
PYG="apptainer exec --nv -B $SCRATCH_ROOT $IMG env PYTHONPATH=$CODE/lib:$REPO/src:$CODE OMP_NUM_THREADS=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled $VENV/bin/python"
$PYG $CC/ctrl_sample.py --label $LABEL --run-dir $RD --milestone $MS --stage $B --out-dir $OUT --sets $S1 --frames identity --seeds $SD1 --batch-size 256 2>&1 | grep -av "sampling loop" | tail -5
rc=${PIPESTATUS[0]}; echo "$(utc) phase1 rc=$rc" >> $H; [ $rc -eq 0 ] || exit 11
if [ -n "$S2" ]; then
$PYG $CC/ctrl_sample.py --label $LABEL --run-dir $RD --milestone $MS --stage $B --out-dir $OUT --sets $S2 --frames identity --seeds $SD2 --batch-size 256 2>&1 | grep -av "sampling loop" | tail -5
rc=${PIPESTATUS[0]}; echo "$(utc) phase2 rc=$rc" >> $H; [ $rc -eq 0 ] || exit 12
fi
echo "$(utc) end job=$JOB" >> $H
