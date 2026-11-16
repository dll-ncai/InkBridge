# FMDX backbone build: shared paths and helpers (sourced). Same paths on both clusters: on CLUSTER-B the Pyxis container
# bind-mounts $U_B at $U (FMDX_SITE=clusterB); on CLUSTER-A commands run in torch27.sif.
R=$U; X=$R/exp/FMD/xs; BB=$X/code/fmdx_backbone_v1
HC=$R/exp/FMD/code/fmd_harness_v1; C0=$R/e0/code_4cd40f57; UE=$R/shared/uhwr_eval-0.3.1; REPO=$R/repos/nagamatsu
IMG=$R/images/torch27.sif; VENV=$R/venvs/e0; FMDVENV=$R/exp/FMD/venv; TIMMDEPS=$X/code/pydeps_timm
D=$R/data/m5_multiscript_v1; LUO_NPZ=$D/m5_luo_train.npz; P01_NPZ=$D/m5_ft_ouhdl_p01.npz
LUO_CKPT=$R/exp/M5/runs/luo/checkpoints/checkpoint_000006.pt          # sha256 ab6b178d...
DB=$X/data/backbone
LUO_R224=$DB/luo_train_render224.npy; LUO_UIDS=$DB/luo_train_render224__unit_ids.npy
P01_R224=$DB/p01train_render224.npy
VALSETS="casia_zh casia_en hpl_tamil iam"
SITE=${FMDX_SITE:-clusterA}
ENVB="NCCL_P2P_DISABLE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
PYP=$REPO/src:$BB:$BB/vendor:$HC:$C0:$UE:$TIMMDEPS
if [ "$SITE" = clusterA ]; then
  WRAP="apptainer exec --nv -B $SCRATCH_ROOT -B /tmp $IMG env $ENVB PYTHONPATH=$PYP OMP_NUM_THREADS=4"
  WRAPFM="apptainer exec --nv -B $SCRATCH_ROOT -B /tmp $IMG env $ENVB OMP_NUM_THREADS=8"
else
  WRAP="env $ENVB PYTHONPATH=$PYP OMP_NUM_THREADS=4"
  WRAPFM="env $ENVB OMP_NUM_THREADS=8"
fi
PY="$WRAP $VENV/bin/python"
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
gpu_names() { nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | tr '\n' ';'; }
hist() {  # hist RUNDIR WHAT
  mkdir -p "$1"; echo "$(utc) $2 site=$SITE host=$(hostname) slurm=${SLURM_JOB_ID:-none} restart=${SLURM_RESTART_COUNT:-0} part=${SLURM_JOB_PARTITION:-none} gpus=[$(gpu_names)]" >> "$1/job_history.txt"
}
check_code() { (cd $BB && sha256sum -c --quiet SHA256SUMS.txt) || { echo "fmdx_backbone_v1 differs from SHA256SUMS.txt"; return 3; }; }
# arm table lookup: arm_field ARM COLUMN
ARMS_TSV=$X/admin/arms.tsv     # arm table (data file, sha256 logged in arm_log.md at every change)
arm_field() { awk -F'\t' -v a="$1" -v c="$2" 'NR==1{for(i=1;i<=NF;i++) h[$i]=i; next} $h["arm_id"]==a {print $h[c]; exit}' $ARMS_TSV; }
# ---- race lock on the VPS hub (atomic mkdir on Windows; owner file = lane + job id). Echoes WIN | LOSE <owner> | ERR
VPS="Administrator@<ip>"; VPSKEY=$HOME/.ssh/uhwr_staging_vps_ed25519
SSHV="ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -i $VPSKEY $VPS"
race_lock() {  # race_lock ARM LANE. v1.0.1-novps (<author> decision 4 Nov: VPS off): atomic mkdir on THIS cluster's filesystem
  local arm=$1 lane=$2 d=$X/race_local/$1 out
  mkdir -p $X/race_local
  if mkdir $d 2>/dev/null; then echo "$lane ${SLURM_JOB_ID:-none} $(utc)" > $d/owner.txt; echo WIN; return 0; fi
  out=$(head -1 $d/owner.txt 2>/dev/null)
  if [ -z "$out" ]; then sleep 5; out=$(head -1 $d/owner.txt 2>/dev/null); fi
  if [ -z "$out" ]; then echo ERR; return 0; fi
  case "$out" in "$lane "*) echo WIN;; *) echo "LOSE $out";; esac
}
