#!/bin/bash
set -eu
U=$U; Q=$U/exp/RQF; R=$Q/R3; C=$Q/code/rq3_v1; L=$Q/queue/rqf_log.md
IMG=$U/images/torch27.sif; PY=$U/venvs/uhwr_eval02/bin/python; UE=$U/shared/uhwr_eval-0.3.1; E0C=$U/e0/code_4cd40f57
SB="env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin /usr/bin/sbatch --parsable"
AX="apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$UE:$C ORDER_CODE=$C"
log(){ echo "$(date -u +%FT%TZ) [rqf-s3, <agent>] $*" | tee -a $L $R/logs/submit.txt; }
g=C34
c1=$($SB -p cpuidle --requeue -c 4 --mem=24G -t 04:00:00 -J r3_apply_C34 -o $R/logs/apply_C34_%j.out --wrap "RQ3_TRAIN=0 $AX $PY $C/rq3_apply.py C34")
c2=$($SB -p cpuidle --requeue -c 16 --mem=48G -t 10:00:00 -J r3_score_C34 --array=1-30%30 -d afterok:$c1 -o $R/logs/score_C34_%A_%a.out --wrap "\
 ONLY=\$(sed -n \${SLURM_ARRAY_TASK_ID}p $R/score_in_C34/tasks.txt); [ -z \"\$ONLY\" ] && exit 0; export TMPDIR=/tmp/R3_\${SLURM_JOB_ID}_\$SLURM_ARRAY_TASK_ID; mkdir -p \$TMPDIR; \
 echo \"\$(date -u +%FT%TZ) start g=C34 task=\$SLURM_ARRAY_TASK_ID only=\$ONLY node=\$(hostname) code_sha256=\$(sha256sum $C/lane_score.py|cut -c1-64)\" >> $R/score_out/job_history.txt; \
 apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=\$TMPDIR MPLCONFIGDIR=\$TMPDIR CUDA_VISIBLE_DEVICES= PYTHONPATH=$UE $PY $C/lane_score.py --preds-dir $R/score_in_C34 --out $R/score_out --e0code $E0C --procs 16 --only \"\$ONLY\"; rc=\$?; \
 echo \"\$(date -u +%FT%TZ) end g=C34 task=\$SLURM_ARRAY_TASK_ID rc=\$rc\" >> $R/score_out/job_history.txt; rm -rf \$TMPDIR; exit \$rc")
mkdir -p $R/read_C3 $R/read_all3
c3=$($SB -p cpuidle --requeue -c 4 --mem=24G -t 02:00:00 -J r3_read_C3 -d afterok:$c2:24964060 -o $R/logs/read_C3_%j.out --wrap "$AX $PY $C/rq3_read.py $R/read_C3 COMP_p01")
log "R3 COMP-p01 s20261034 (pulled from VPS 4090, sha-verified) apply=$c1 score_array=$c2 (30 tasks) read_C3=$c3 (all 3 training seeds; afterok $c2 and 24964060)"
f=$($SB -p cpuidle --requeue -c 2 --mem=8G -t 00:30:00 -J r3_summ3 -d afterok:24964056:$c3 -o $R/logs/summ3_%j.out --wrap "rm -rf $R/read_all3; mkdir -p $R/read_all3; cp $R/read_A/* $R/read_C3/* $R/read_all3/; $AX $PY $C/rq3_summarize.py $R/read_all3 $R/R3_summary_3seed")
log "R3 summary (final, COMP over 3 training seeds) job=$f afterok 24964056:$c3; old summary 24964062 (2-seed COMP) cancelled - own job, replaced"
env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin /usr/bin/scancel 24964062
log "scancel 24964062 (own summary job, superseded by $f)"
