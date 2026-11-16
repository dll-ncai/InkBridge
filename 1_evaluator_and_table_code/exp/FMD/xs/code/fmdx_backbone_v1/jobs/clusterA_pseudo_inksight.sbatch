#!/bin/bash
#SBATCH -J fmdx_pseudo_ink
#SBATCH -p cpuidle
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=10-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/FMD/xs/logs/pseudo/%x_%A_%a.out
# InkSight Small-p on the pseudo pool (same image tf220.sif, pylib/tf, weights hf/InkSight-Small-p rev 8514f237 and runner as the
# FMD val1000 screen run exp/FMD/code/inksight_val1000.sbatch; runner copied byte-identical, sha256 logged). One shard per task:
# manifest[k::NSHARD]. Resumes from out_<k>.jsonl (requeue-safe). Gate-aware (FAIL -> exit 0).
# Usage: sbatch --array=0-(NSHARD-1)%CONC clusterA_pseudo_inksight.sbatch NSHARD
set -u
S=$U; X=$S/exp/FMD/xs; PD=$X/data/pseudo; O=$PD/inksight; mkdir -p $O; cd $O
python3 -c "import json,sys; d=json.load(open('$X/verdicts/verdict_inksight_small_p.json')); sys.exit(1 if str(d.get('status','')).upper()=='FAIL' else 0)" 2>/dev/null
[ $? -eq 1 ] && { echo "InkSight verdict FAIL -> skipped"; exit 0; }
NS=$1; K=$SLURM_ARRAY_TASK_ID; k=$(printf %04d $K)
SRC=$S/exp/FMD/runs/inksight_val1000/inksight_runner.py
[ -f inksight_runner.py ] || cp $SRC .
sha256sum inksight_runner.py $SRC > runner_sha256_$k.txt
export PYTHONPATH=$S/pylib/tf INKSIGHT_REPO=$S/repos/inksight INKSIGHT_MODEL=$S/hf/InkSight-Small-p TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=""
apptainer exec -B $SCRATCH_ROOT $S/images/tf220.sif python -c "
import json; m=json.load(open('$PD/manifest.json')); json.dump(m[$K::$NS], open('shard_$k.json','w'))"
echo "$(date -u +%FT%TZ) start task=$K node=$(hostname) restart=${SLURM_RESTART_COUNT:-0}" >> job_history.txt
TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=4 apptainer exec -B $SCRATCH_ROOT $S/images/tf220.sif python inksight_runner.py shard_$k.json out_$k.jsonl default >> log_$k.txt 2>&1
rc=$?; echo "$(date -u +%FT%TZ) end task=$K rc=$rc lines=$(wc -l < out_$k.jsonl)" >> job_history.txt; exit $rc
