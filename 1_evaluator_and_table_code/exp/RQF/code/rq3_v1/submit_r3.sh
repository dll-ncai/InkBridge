#!/bin/bash
# RQF R3 chain submitter (login node). Frozen code: exp/RQF/code/rq3_v1 (SHA256SUMS). Logs every id to exp/RQF/queue/rqf_log.md
set -eu
U=$U; Q=$U/exp/RQF; R=$Q/R3; C=$Q/code/rq3_v1; L=$Q/queue/rqf_log.md
IMG=$U/images/torch27.sif; PY=$U/venvs/uhwr_eval02/bin/python; UE=$U/shared/uhwr_eval-0.3.1; E0C=$U/e0/code_4cd40f57
SB="env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin /usr/bin/sbatch --parsable"
AX="apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=$UE:$C ORDER_CODE=$C"
mkdir -p $R/logs $R/score_out/rows $R/read_A $R/read_C
log(){ echo "$(date -u +%FT%TZ) [rqf-s3, <agent>] $*" | tee -a $L $R/logs/submit.txt; }
score(){ # $1=group $2=n_array $3=dep
 $SB -p cpuidle --requeue -c 16 --mem=48G -t 10:00:00 -J r3_score_$1 --array=1-$2%30 -d afterok:$3 -o $R/logs/score_$1_%A_%a.out --wrap "\
 ONLY=\$(sed -n \${SLURM_ARRAY_TASK_ID}p $R/score_in_$1/tasks.txt); [ -z \"\$ONLY\" ] && exit 0; export TMPDIR=/tmp/R3_\${SLURM_JOB_ID}_\$SLURM_ARRAY_TASK_ID; mkdir -p \$TMPDIR; \
 echo \"\$(date -u +%FT%TZ) start g=$1 task=\$SLURM_ARRAY_TASK_ID only=\$ONLY node=\$(hostname) code_sha256=\$(sha256sum $C/lane_score.py|cut -c1-64)\" >> $R/score_out/job_history.txt; \
 apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=\$TMPDIR MPLCONFIGDIR=\$TMPDIR CUDA_VISIBLE_DEVICES= PYTHONPATH=$UE $PY $C/lane_score.py --preds-dir $R/score_in_$1 --out $R/score_out --e0code $E0C --procs 16 --only \"\$ONLY\"; rc=\$?; \
 echo \"\$(date -u +%FT%TZ) end g=$1 task=\$SLURM_ARRAY_TASK_ID rc=\$rc\" >> $R/score_out/job_history.txt; rm -rf \$TMPDIR; exit \$rc"
}
# ---- group A (all bases except COMP): sampling already exists
a1=$($SB -p cpuidle --requeue -c 4 --mem=24G -t 08:00:00 -J r3_apply_A -o $R/logs/apply_A_%j.out --wrap "$AX $PY $C/rq3_apply.py A")
a2=$(score A 46 $a1)
a3=$($SB -p cpuidle --requeue -c 4 --mem=24G -t 02:00:00 -J r3_read_A -d afterok:$a2 -o $R/logs/read_A_%j.out --wrap "$AX $PY $C/rq3_read.py $R/read_A E0,M5,LUOft_p10,NOFM_W_FT,PEINIT_W_FT,LUO_ms6,NOFM_W_comp")
log "R3 group A (E0,M5,LUOft,NOFM_FT,PEINIT_FT,LUO_ms6,NOFM_comp; no sampling needed) apply=$a1 score_array=$a2 read=$a3 (afterok chain, cpuidle)"
# ---- group C (COMP-p01 C35,C36): A4000 sampling seeds 1,2 on 6 non-Urdu sets -> apply -> score -> read
s35=$($SB -p gpuidle --gres=gpu:a4000:1 --cpus-per-task=8 --mem=16G --requeue -J r3_samp_C35 $C/comp_rq3_sample.sbatch C35)
s36=$($SB -p gpuidle --gres=gpu:a4000:1 --cpus-per-task=8 --mem=16G --requeue -J r3_samp_C36 $C/comp_rq3_sample.sbatch C36)
log "R3 COMP-p01 A4000 sampling seeds1,2 x 6 non-Urdu sets: C35=$s35 C36=$s36 (gpuidle a4000; C34 ckpt only on VPS -> VPS lane)"
c1=$($SB -p cpuidle --requeue -c 4 --mem=24G -t 04:00:00 -J r3_apply_C -d afterok:$s35:$s36 --kill-on-invalid-dep=yes -o $R/logs/apply_C_%j.out --wrap "RQ3_TRAIN=1,2 $AX $PY $C/rq3_apply.py C")
c2=$(score C 60 $c1)
c3=$($SB -p cpuidle --requeue -c 4 --mem=24G -t 02:00:00 -J r3_read_C -d afterok:$c2 -o $R/logs/read_C_%j.out --wrap "$AX $PY $C/rq3_read.py $R/read_C COMP_p01")
log "R3 COMP-p01 (train seeds 20261035/36) apply=$c1 score_array=$c2 read=$c3 (afterok chain)"
f=$($SB -p cpuidle --requeue -c 2 --mem=8G -t 00:30:00 -J r3_summ -d afterok:$a3:$c3 -o $R/logs/summ_%j.out --wrap "mkdir -p $R/read_all; cp $R/read_A/* $R/read_C/* $R/read_all/ 2>/dev/null; $AX $PY $C/rq3_summarize.py $R/read_all $R/R3_summary")
log "R3 summary job=$f (afterok read_A=$a3, read_C=$c3)"
