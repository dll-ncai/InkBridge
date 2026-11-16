# RQF R2d (second held-out script: HPL Tamil) shared paths/helpers. Sourced by stage.sh. Same $SCRATCH_ROOT paths on both lanes
# (CLUSTER-B: $U_B bind-mounted at $U inside the Pyxis container, R2D_SITE=clusterB).
R=$U; RD=$R/exp/RQF/R2d; RC=$RD/code/r2d_v1; DD=$RD/data
X=$R/exp/FMD/xs; BB=$X/code/fmdx_backbone_v1; HC=$R/exp/FMD/code/fmd_harness_v1; C0=$R/e0/code_4cd40f57
UE=$R/shared/uhwr_eval-0.3.1; REPO=$R/repos/nagamatsu; MC=$R/exp/M5/code_e33511dd; FC=$R/exp/CAND/code/frame_code_a44e78b0
IMG=$R/images/torch27.sif; VENV=$R/venvs/e0; EVENV=$R/venvs/uhwr_eval02
D=$R/data/m5_multiscript_v1; OU=$R/data/ouhdl_e0_v3; DB=$X/data/backbone
SITE=${R2D_SITE:-clusterA}
RUNB=${RUNB:-$RD/runs}; PREDB=${PREDB:-$RD/preds}; SCOREB=${SCOREB:-$RD/score_out}
ENVB="NCCL_P2P_DISABLE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
P_M5=$REPO/src:$MC:$C0                                            # M5 recipe (m5_train/m5_build/e0_sample/e0_score as M5)
P_BB=$REPO/src:$BB:$BB/vendor:$HC:$C0:$UE:$X/code/pydeps_timm     # FMDX backbone code (as fmdx_backbone_v1/jobs/common.sh)
P_FC=$FC/lib:$REPO/src:$FC                                        # frame code (COMP-p01 recipe)
if [ "$SITE" = clusterA ]; then W="apptainer exec --nv -B $SCRATCH_ROOT -B /tmp $IMG env $ENVB"; WC="apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env $ENVB"
else W="env $ENVB"; WC="env $ENVB"; fi
py()  { local pp=$1; shift; $W PYTHONPATH=$pp OMP_NUM_THREADS=${OMPN:-4} $VENV/bin/python "$@"; }
pyc() { local pp=$1; shift; $WC PYTHONPATH=$pp OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= $VENV/bin/python "$@"; }
PY3=${PY3:-python3}
# v1.1 (8 Nov): optional node-local copies of the venvs (BeeGFS import stalls on CPU nodes; same remedy as the Metric track)
if [ "${R2D_SHM:-0}" = 1 ] && [ "$SITE" = clusterA ]; then SH=/dev/shm/r2d_${SLURM_JOB_ID:-$$}
  if [ ! -x $SH/e0/bin/python ]; then mkdir -p $SH && cp -a $VENV $SH/e0 && cp -a $EVENV $SH/uhwr_eval02 && mkdir -p $SH/numba; fi
  VENV=$SH/e0; EVENV=$SH/uhwr_eval02; ENVB="$ENVB NUMBA_CACHE_DIR=$SH/numba"
  W="apptainer exec --nv -B $SCRATCH_ROOT -B /tmp -B /dev/shm $IMG env $ENVB"; WC="apptainer exec -B $SCRATCH_ROOT -B /tmp -B /dev/shm $IMG env $ENVB"; fi
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
gpus() { nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader 2>/dev/null | tr '\n' ';'; }
hist() { mkdir -p "$1"; echo "$(utc) $2 site=$SITE host=$(hostname) slurm=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0} part=${SLURM_JOB_PARTITION:-none} gpus=[$(gpus)]" >> "$1/job_history.txt"; }
gpu_guard() { [ "${SMOKE:-0}" = 1 ] && return 0; local cc; cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  awk -v c="$cc" 'BEGIN{exit !(c+0 >= 8.0)}' || { echo "GPU_CC_LT_8 $cc"; exit 4; }; }
check_code() { (cd $RC && sha256sum -c --quiet SHA256SUMS.txt) || { echo "r2d_v1 differs from SHA256SUMS.txt"; exit 3; }; }
memg() { local m; m=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' '); echo $((m / 1024)); }
# start_watch KIND LOG RUN MIN_STEP PGID : in-job GATE_FAIL watcher (non-finite loss / divergence), milestone gate lines in RUN/gate_log.txt
start_watch() { $PY3 $RC/gate_watch.py --kind $1 --log "$2" --run-dir $3 --min-step $4 --max-loss ${GATE_MAX_LOSS:-0.10} --pgid $5 --every ${GATE_EVERY:-60} >> $3/gate_watch.out 2>&1 & WATCH_PID=$!; }
stop_watch() { kill $WATCH_PID 2>/dev/null; wait $WATCH_PID 2>/dev/null; true; }
