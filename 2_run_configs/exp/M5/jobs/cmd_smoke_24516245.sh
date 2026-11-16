# M5 / X6 single card. Step A: GPU smoke on b200idle (2 GPUs) with the exact M5 scripts and tiny budgets.
# Step B: automatic smoke check. Step C (only if B passes): sbatch the full jobs on b200idle:
#   joint (2xB200, 100k steps E0 recipe + selection + eval), luo (2xB200, same), ft (2xB200, afterok:luo; 4 fine-tunes x 10k steps).
set -uo pipefail
TAG=e33511dd; B=$U; M=$B/exp/M5; MC=$M/code_$TAG; R=$M/smoke/runs
# guard (in case the dependency directive is not honoured): wait for prep Slurm 24516200 and its data provenance
while squeue -h -j 24516200 2>/dev/null | grep -q .; do sleep 60; done
[ -s $B/data/m5_multiscript_v1/PROVENANCE_m5_multiscript_v1.json ] || { echo "prep output missing: abort"; exit 1; }
mkdir -p $R $M/jobs; cp ./cmd.sh $M/jobs/cmd_smoke_${SLURM_JOB_ID:-x}.sh 2>/dev/null || true
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv | tee gpu.txt
export TAG RUNROOT=$R
t0=$(date +%s)
ARM=joint STEPS=200 SAVE=100 VALIDATION_EVERY=100 EVAL_MAX=64 bash $MC/m5_arm.sh || { echo "SMOKE joint run1 FAILED"; exit 1; }
t1=$(date +%s); echo "joint run1 wall $((t1-t0)) s"
[ -f $R/joint/resume_test_done ] || { rm -f $R/joint/training_complete.json; touch $R/joint/resume_test_done; }
ARM=joint STEPS=400 SAVE=100 VALIDATION_EVERY=100 EVAL_MAX=64 bash $MC/m5_arm.sh || { echo "SMOKE joint run2 (resume) FAILED"; exit 1; }
t2=$(date +%s); echo "joint run2 wall $((t2-t1)) s"
ARM=luo STEPS=100 SAVE=100 VALIDATION_EVERY=100 EVAL_MAX=32 bash $MC/m5_arm.sh || { echo "SMOKE luo FAILED"; exit 1; }
t3=$(date +%s); echo "luo wall $((t3-t2)) s"
FT_STEPS=200 FT_SAVE=100 ARMS="p01 p10" EVAL_MAX=32 VAL_SUBSET=64 INIT_RUN=$R/luo bash $MC/m5_ft.sh || { echo "SMOKE ft FAILED"; exit 1; }
t4=$(date +%s); echo "ft wall $((t4-t3)) s"
# checkpoint timing (s per 100 steps, includes the 25-sample snapshot at each save)
for rd in $R/joint $R/ft/p01 $R/ft/p10; do echo "$rd"; ls -l --time-style=+%s $rd/checkpoints | awk 'NR>1{print $6, $7}'; done | tee ckpt_times.txt
grep -h "number of train_data\|Resumed\|Initialised" $R/joint/train_*.log $R/luo/train_*.log $R/ft/*/train_*.log | tee train_markers.txt || true
mkdir -p out
cp gpu.txt ckpt_times.txt train_markers.txt out/ || true
for rd in $R/joint $R/luo $R/ft/p01 $R/ft/p10; do n=$(echo ${rd#$R/} | tr / _); cp $rd/job_history.txt out/job_history_$n.txt; cp $rd/eval/selection.json out/selection_$n.json 2>/dev/null || true; cp $rd/train_*_progress.txt out/ 2>/dev/null || true; [ -f $rd/init_from.json ] && cp $rd/init_from.json out/init_from_$n.json; done
cp $R/joint/eval/summary_M5_joint.csv $R/luo/eval/summary_M5_luo.csv $R/ft/summary_M5_ft.csv $M/PROVENANCE_M5.json out/ 2>/dev/null || true
cp $R/ft/job_history.txt out/job_history_ft_root.txt || true
ls -lh out
# ---------------- Step B: smoke check
python3 - <<PY > out/smoke_check.txt 2>&1 || { cat out/smoke_check.txt; echo "SMOKE CHECK FAILED: full jobs NOT submitted"; exit 1; }
import json, os, glob, csv
R = "$R"; bad = []
for rd in ["joint", "luo", "ft/p01", "ft/p10"]:
    if not os.path.exists(f"{R}/{rd}/training_complete.json"): bad.append(f"{rd}: no training_complete")
    if not os.path.exists(f"{R}/{rd}/eval/selection.json"): bad.append(f"{rd}: no selection")
if not any("Resumed from milestone 2" in open(f, errors="ignore").read() for f in glob.glob(f"{R}/joint/train_*.log")): bad.append("joint: resume not observed")
for a in ["p01", "p10"]:
    j = json.load(open(f"{R}/ft/{a}/init_from.json")) if os.path.exists(f"{R}/ft/{a}/init_from.json") else {}
    if j.get("source_step") != 100: bad.append(f"ft {a}: init_from {j}")
for s in [f"{R}/joint/eval/summary_M5_joint.csv", f"{R}/luo/eval/summary_M5_luo.csv", f"{R}/ft/summary_M5_ft.csv"]:
    if not os.path.exists(s): bad.append(f"missing {s}"); continue
    rows = list(csv.DictReader(open(s)))
    m5 = [r for r in rows if r["label"].startswith("m5_")]
    if not m5: bad.append(f"{s}: no m5 rows")
    if any(r["status"] != "ok" for r in m5): bad.append(f"{s}: missing m5 rows files")
    if any(r.get("n_errors") not in ("0", "") for r in m5): bad.append(f"{s}: scoring errors")
print("SMOKE_CHECK", "PASS" if not bad else "FAIL", bad)
raise SystemExit(1 if bad else 0)
PY
cat out/smoke_check.txt
# ---------------- Step C: full jobs (b200idle, requeue, auto-resume)
F=$M/runs; L=$M/logs; mkdir -p $F $L
for ARM in joint luo; do cat > $M/jobs/full_${ARM}.sbatch <<SB
#!/bin/bash
#SBATCH --job-name=m5_${ARM}
#SBATCH --partition=b200idle
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=5-00:00:00
#SBATCH --requeue
#SBATCH --output=$L/%x_%j.out
export TAG=$TAG RUNROOT=$F ARM=$ARM STEPS=100000 SAVE=10000 VALIDATION_EVERY=1000 EVAL_MAX=0 NGPU=2
bash $MC/m5_arm.sh
SB
done
cat > $M/jobs/full_ft.sbatch <<SB
#!/bin/bash
#SBATCH --job-name=m5_ft
#SBATCH --partition=b200idle
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=4-00:00:00
#SBATCH --requeue
#SBATCH --output=$L/%x_%j.out
export TAG=$TAG RUNROOT=$F FT_STEPS=10000 FT_SAVE=1000 EVAL_MAX=0 VAL_SUBSET=1000 ARMS="p01 p10 p30 p100"
bash $MC/m5_ft.sh
SB
if [ ! -s $M/jobs/full_slurm_ids.txt ]; then
  JJ=$(sbatch --parsable $M/jobs/full_joint.sbatch); JL=$(sbatch --parsable $M/jobs/full_luo.sbatch)
  JF=$(sbatch --parsable --dependency=afterok:$JL $M/jobs/full_ft.sbatch)
  printf 'joint %s\nluo %s\nft %s (afterok:%s)\nsubmitted_utc %s by orchestrator slurm %s\n' $JJ $JL $JF $JL "$(date -u +%FT%TZ)" ${SLURM_JOB_ID:-x} > $M/jobs/full_slurm_ids.txt
fi
cat $M/jobs/full_slurm_ids.txt | tee out/full_slurm_ids.txt
cp $M/jobs/full_*.sbatch out/ || true
squeue -u <user> -o '%i %P %j %T %r %b' | tee out/squeue_after_submit.txt || true
