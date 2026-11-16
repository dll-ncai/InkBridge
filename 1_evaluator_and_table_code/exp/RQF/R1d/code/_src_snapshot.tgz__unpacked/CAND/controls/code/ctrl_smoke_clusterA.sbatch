#!/bin/bash
#SBATCH --job-name=cand_ctrl_smoke
#SBATCH --partition=epyc-128
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=$U/exp/CAND/controls/logs/%x_%j.out
# CPU smoke of both control configurations (smoke numbers are not results): 20 steps, eff. batch 8, 8 units per set.
set -uo pipefail; exec 2>&1
B=$U; S=$B/exp/CAND/controls/smoke
export CTRL_OUT=$S RUNROOT=$S/runs CANDLOG=$S/cand_log_smoke.md SMOKE_UNITS=8 STEPS=20 SAVE=10 BATCH=4 ACCUM=2 WORKERS=0 AMP=--no-amp PROCS=8
X="test_iam500 test_casia_zh test_casia_en test_hpl_tamil german_onhw_all"
rc_all=0
ARM=smoke_luoft_p01_s1 FRAC=p01 FRAME=identity REPLAY=0 TRAIN_SEED=20261017 SEEDS="0" EXTRA_SETS="$X" bash $B/exp/CAND/controls/code/ctrl_arm.sh || rc_all=$?
ARM=smoke_luo_cont_nourdu FRAC=p01 FRAME=identity REPLAY=1 QU=0 TRAIN_SEED=20261016 SEEDS="0" EXTRA_SETS="$X" bash $B/exp/CAND/controls/code/ctrl_arm.sh || rc_all=$?
python3 - <<'PY'
import json
rc = json.load(open("$U/exp/CAND/controls/smoke/runs/smoke_luo_cont_nourdu/replay_config.json"))
print("replay_config sampler:", json.dumps(rc["sampler"])[:600])
PY
grep -h "urdu_draws\|script_draws\|Replay loader" $S/runs/smoke_luo_cont_nourdu/train_*.log | tail -3 || true
ls -la $S/preds | head -40
exit $rc_all
