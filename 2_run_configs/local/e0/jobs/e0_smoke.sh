#SBATCH -p gpuidle
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
# E0 smoke job: env setup (once), data staging (once), 1,000 training steps on 1 A30,
# resume check (+200 steps), DDIM-50 sampling of 512 val units, uhwr_eval scoring.
set -u
B=$U
IMG=$B/images/torch27.sif
VENV=$B/venvs/e0
D=$B/data/ouhdl_e0_v3
CODE_TAG=2be9e593
C=$B/e0/code_$CODE_TAG
REPO=$B/repos/nagamatsu
mkdir -p $D $B/e0 $B/runs/e0
echo "host $(hostname) job ${SLURM_JOB_ID:-none} start $(date -Is)" | tee job_info.txt
sha256sum e0_code.tar.gz e0_*.npz | tee staged_sha256.txt
for f in e0_train.npz e0_val.npz e0_test.npz e0_gold.npz; do [ -f $D/$f ] || cp ./$f $D/$f; done
(cd $D && sha256sum e0_*.npz) | tee data_sha256.txt
mkdir -p $C && tar -xzf e0_code.tar.gz -C $C --strip-components=1
git -C $REPO rev-parse HEAD | tee repo_commit.txt || true
git -C $REPO status --porcelain | tee repo_status.txt || true
sha256sum $IMG | tee image_sha256.txt || true
if [ ! -x $VENV/bin/python ]; then
  apptainer exec -B $SCRATCH_ROOT $IMG python -m venv --system-site-packages $VENV
  apptainer exec -B $SCRATCH_ROOT $IMG $VENV/bin/pip install -q "torch==2.7.1" accelerate==1.12.0 einops==0.8.1 \
      ema-pytorch==0.7.7 lmdb==1.7.5 opencv-python-headless==4.12.0.88 tslearn==0.7.0 wandb==0.23.1 \
      numba pandas scipy matplotlib scikit-image 2>&1 | tail -5
fi
apptainer exec -B $SCRATCH_ROOT $IMG $VENV/bin/pip freeze > pip_freeze.txt
RUN="apptainer exec --nv -B $SCRATCH_ROOT $IMG env PYTHONPATH=$REPO/src:$C OMP_NUM_THREADS=1 $VENV/bin/python"
$RUN -c "import torch, numpy, accelerate; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), 'numpy', numpy.__version__, 'accelerate', accelerate.__version__)" | tee env_check.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a env_check.txt || true
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 20 > gpumem.log 2>&1 &
SMI=$!
SM=$B/runs/e0/smoke_${SLURM_JOB_ID:-x}
COMMON="--e0-train-npz $D/e0_train.npz --condition rendered --gradient-accumulate-every 1 --sequence-length 320 \
 --image-size 64 --sampling-timesteps 50 --train-fraction 0.99 --num-workers 6 --wandb-mode disabled \
 --random-seed 20261016 --sample-count 25 --classes 0"
BS=512
T0=$(date +%s)
if $RUN $C/e0_train.py $COMMON --batch-size $BS --run-dir $SM --train-steps 1000 --save-every 500 --validation-every 500 > train_smoke.log 2>&1; then
  echo "bs512 ok" | tee smoke_status.txt
else
  echo "bs512 FAILED; retry bs256 x accum2" | tee smoke_status.txt
  tail -20 train_smoke.log
  SM=${SM}_bs256
  BS=256
  COMMON=${COMMON/--gradient-accumulate-every 1/--gradient-accumulate-every 2}
  $RUN $C/e0_train.py $COMMON --batch-size $BS --run-dir $SM --train-steps 1000 --save-every 500 --validation-every 500 > train_smoke_bs256.log 2>&1 || echo "bs256 FAILED" | tee -a smoke_status.txt
fi
T1=$(date +%s); echo "train_1000_steps_seconds $((T1-T0))" | tee -a smoke_status.txt
$RUN $C/e0_train.py $COMMON --batch-size $BS --run-dir $SM --train-steps 1200 --save-every 500 --validation-every 500 > train_resume.log 2>&1 || echo "resume FAILED" | tee -a smoke_status.txt
T2=$(date +%s); echo "resume_200_steps_seconds $((T2-T1))" | tee -a smoke_status.txt
grep -a "Resumed from" train_resume.log | tee -a smoke_status.txt || true
ls -la $SM $SM/checkpoints | tee run_listing.txt || true
cp $SM/run_config*.json . || true
cp $SM/training_samples/generated/*001000* . 2>/dev/null || true
$RUN $C/e0_sample.py --run-dir $SM --milestone 2 --npz $D/e0_val.npz --condition rendered --sampling-timesteps 50 \
  --ddim-eta 0 --batch-size 512 --seed 0 --max-units 512 --out smoke_val512_preds.npz 2>&1 | grep -av "sampling loop" | tee sample_smoke.log || true
$RUN $C/e0_score.py e0 --npz $D/e0_val.npz --preds smoke_val512_preds.npz --method e0_smoke_step1000 --input rendered \
  --out smoke_val512_rows.csv --procs 8 2>&1 | tail -3 || true
$RUN $C/e0_score.py summary --rows smoke_val512_rows.csv --out smoke_val512_summary.csv 2>&1 | tail -2 || true
kill $SMI 2>/dev/null || true
tr '\r' '\n' < train_smoke.log | grep -av "sampling loop" | awk 'NR%50==1' | tail -40 > train_smoke_tail.txt || true
tr '\r' '\n' < train_resume.log | grep -av "sampling loop" | tail -15 > train_resume_tail.txt || true
echo "end $(date -Is)" | tee -a job_info.txt
ls -lh
