#!/bin/bash
# PENNET smoke driver (one approval card). Runs in the submit_job workdir with ./pennet_code.tar.gz staged.
#  G0 GPU capability >= 8.0 on the Slurm-allocated device; G1 unit tests; G2 resume equality; G3 SIGTERM save + resume;
#  G4 decode + e0_score (uhwr_eval 0.3.1) + selection + writer-cluster summary on 64-unit subsets;
#  then, only if every gate passed: sbatch the full training job (gpuidle, 1 GPU, requeue) and the afterok CPU scoring job.
set +e
set -u -o pipefail
U=$U; E=$U/exp/PENNET; SIF=$U/images/torch27.sif; VENV=$U/venvs/e0; D=$U/data/ouhdl_e0_v3
UE=$U/shared/uhwr_eval-0.3.1; S=$E/smoke; WD=$(pwd)
EXCL="gpu[003-005,013-014,017-023,101-148],dgx[001-005]"
mkdir -p $E/lib $E/logs $S/rows
fail() { echo "SMOKE GATE FAILED: $1" | tee -a $S/smoke_status.txt; cp $S/*.json $S/*.txt $S/*.log $WD/ 2>/dev/null; exit 1; }
echo "$(date -Is) smoke start job=${SLURM_JOB_ID:-none} host=$(hostname)" | tee $S/smoke_status.txt
CS=$(sha256sum pennet_code.tar.gz | cut -c1-8); C=$E/code_$CS
if [ -d $C ]; then
  mkdir -p $WD/chk && tar -xzf pennet_code.tar.gz -C $WD/chk && diff -r -x __pycache__ $WD/chk $C > /dev/null || fail "existing $C differs from tarball"
else
  mkdir -p $C && tar -xzf pennet_code.tar.gz -C $C
fi
[ -f $E/pennet_code_$CS.tar.gz ] || cp pennet_code.tar.gz $E/pennet_code_$CS.tar.gz
ln -sfn $U/e0/code_4cd40f57/uhwr_baselines $E/lib/uhwr_baselines
for f in e0_adapter.py e0_score.py; do cmp -s $C/$f $U/e0/code_4cd40f57/$f || fail "scorer copy differs: $f"; done
echo "code $C" | tee -a $S/smoke_status.txt
echo "${SLURM_JOB_ID:-none} smoke_driver $(hostname) $(date -Is)" >> $E/slurm_ids.txt
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  [ -n "${SLURM_JOB_GPUS:-}" ] || fail "no Slurm GPU allocation visible"
  export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
fi
nvidia-smi -i ${CUDA_VISIBLE_DEVICES%%,*} --query-gpu=index,name,compute_cap,pci.bus_id,uuid,memory.total --format=csv > $S/gpu_allocated.txt
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-}" >> $S/gpu_allocated.txt
SIFPID=""
if [ ! -f $E/sif_sha256.txt ]; then (sha256sum $SIF > $E/sif_sha256.txt.tmp && mv $E/sif_sha256.txt.tmp $E/sif_sha256.txt) & SIFPID=$!; fi
ENVG="PYTHONPATH=$C OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
RUNG="apptainer exec --nv -B $SCRATCH_ROOT $SIF env $ENVG $VENV/bin/python"
RUNT="apptainer exec --nv -B $SCRATCH_ROOT $SIF env $ENVG PYTHONPATH=$C:$UE $VENV/bin/python"
# ---- G0
$RUNG -c "import torch,sys; c=torch.cuda.get_device_capability(0); print(torch.__version__, torch.cuda.get_device_name(0), c); sys.exit(0 if c[0]>=8 else 3)" | tee $S/gpu.txt || fail "G0 GPU capability < 8.0"
$RUNG -m pip freeze > $E/env_freeze_e0.txt 2>/dev/null
(cd $D && sha256sum e0_*.npz) > $E/data_sha256.txt
# ---- G1
(cd $C && $RUNT $C/test_pennet.py --npz-dir $D --out $S/tests.json > $S/tests.log 2>&1) || fail "G1 unit tests (see tests.log)"
echo "$(date -Is) G1 ok" | tee -a $S/smoke_status.txt
# ---- G2
rm -rf $S/rA $S/rC $S/rD
TR="$C/pennet_train.py --train-npz $D/e0_train.npz --condition rendered --batch 512 --lr 1e-3 --clip 0.2 --log-every 10 --seed 20261016"
$RUNG $TR --run-dir $S/rA --steps 300 --ckpt-every 100 --milestone-every 100 --max-steps-this-process 200 > $S/rA_1.log 2>&1 || fail "G2 rA part 1"
$RUNG $TR --run-dir $S/rA --steps 300 --ckpt-every 100 --milestone-every 100 > $S/rA_2.log 2>&1 || fail "G2 rA part 2"
$RUNG $TR --run-dir $S/rC --steps 300 --ckpt-every 100 --milestone-every 100 > $S/rC.log 2>&1 || fail "G2 rC"
echo "$(date -Is) G2 runs done" | tee -a $S/smoke_status.txt
# ---- G3: SIGTERM to every process of the run (as Slurm does on preemption), then resume
$RUNG $TR --run-dir $S/rD --steps 100000 --ckpt-every 100000 --milestone-every 100000 > $S/rD_1.log 2>&1 &
APID=$!
for i in $(seq 1 120); do sleep 5; n=$(grep -c '"step"' $S/rD/train_log.jsonl 2>/dev/null); [ "${n:-0}" -ge 4 ] && break; done
pkill -TERM -f "pennet_train.py --train-npz $D/e0_train.npz .*--run-dir $S/rD" ; wait $APID; echo "rD part 1 rc=$?" >> $S/smoke_status.txt
SD=$(grep signal_15 $S/rD/ckpt_events.txt | head -1 | sed 's/.*step=\([0-9]*\).*/\1/')
[ -n "$SD" ] || fail "G3 no signal_15 checkpoint"
$RUNG $TR --run-dir $S/rD --steps $((SD+20)) --ckpt-every 100000 --milestone-every 100000 > $S/rD_2.log 2>&1 || fail "G3 resume after SIGTERM"
echo "$(date -Is) G3 runs done (signal step $SD)" | tee -a $S/smoke_status.txt
# ---- G4: decode + score chain on 64-unit subsets
DOK=1
for k in 1 2 3; do
  $RUNG $C/pennet_decode.py --weights $S/rC/milestones/ms_00$k.pt --npz $D/e0_val.npz --condition rendered --max-units 64 --out $S/val64_ms${k}_preds.npz >> $S/decode.log 2>&1 || DOK=0
done
$RUNG $C/pennet_decode.py --weights $S/rC/milestones/ms_003.pt --npz $D/e0_test.npz --condition dataset --max-units 64 --out $S/test_dataset_ms3_preds.npz >> $S/decode.log 2>&1 || DOK=0
$RUNG $C/pennet_decode.py --weights $S/rC/milestones/ms_003.pt --npz $D/e0_gold.npz --condition rendered --max-units 64 --out $S/gold_rendered_ms3_preds.npz >> $S/decode.log 2>&1 || DOK=0
LIB_SHA=977e27481b9c4c94dfc5bfe62541b02533524ab243c66d9bbc0f67098ea37160
LOCAL=/dev/shm/pennet_smoke_${SLURM_JOB_ID:-x}; trap 'rm -rf $LOCAL' EXIT
mkdir -p $LOCAL/numba_cache; cp -a $VENV $LOCAL/venv_e0
h=$(sha256sum $LOCAL/venv_e0/lib/python3.11/site-packages/llvmlite/binding/libllvmlite.so | cut -d' ' -f1); [ "$h" = "$LIB_SHA" ] || DOK=0
RUNS="apptainer exec -B $SCRATCH_ROOT -B $LOCAL $SIF env PYTHONPATH=$C:$UE:$E/lib OMP_NUM_THREADS=1 NUMBA_CACHE_DIR=$LOCAL/numba_cache $LOCAL/venv_e0/bin/python"
$RUNS -c "import uhwr_eval,sys; print('uhwr_eval', uhwr_eval.__version__, uhwr_eval.__file__); sys.exit(0 if uhwr_eval.__version__=='0.3.1' and uhwr_eval.__file__.startswith('$UE/') else 2)" | tee $S/uhwr_eval_version.txt || DOK=0
for k in 1 2 3; do
  $RUNS $C/e0_score.py e0 --npz $D/e0_val.npz --preds $S/val64_ms${k}_preds.npz --method pennet_smoke_ms$k --input rendered --out $S/rows/val64_ms${k}_rows.csv --procs 8 >> $S/score.log 2>&1 || DOK=0
done
$RUNS $C/e0_score.py e0 --npz $D/e0_test.npz --preds $S/test_dataset_ms3_preds.npz --method pennet_smoke_ms3 --input dataset --out $S/rows/test_dataset_ms3_rows.csv --procs 8 >> $S/score.log 2>&1 || DOK=0
$RUNS $C/e0_score.py e0 --npz $D/e0_gold.npz --preds $S/gold_rendered_ms3_preds.npz --method pennet_smoke_ms3 --input rendered --out $S/rows/gold_rendered_ms3_rows.csv --procs 8 >> $S/score.log 2>&1 || DOK=0
for f in $S/rows/*_rows.csv; do n=$(wc -l < $f); [ "$n" -eq 129 ] || { echo "$f has $n lines" >> $S/smoke_status.txt; DOK=0; }; done
[ "$(ls $S/rows/*_rows.csv 2>/dev/null | wc -l)" -eq 5 ] || DOK=0
$RUNS $C/pennet_select.py --rows $S/rows/val64_ms*_rows.csv --out $S/selection_smoke.json >> $S/score.log 2>&1 || DOK=0
$RUNS $C/pennet_summary_wc.py --rows $S/rows/test_dataset_ms3_rows.csv $S/rows/gold_rendered_ms3_rows.csv --out-prefix $S/summary_smoke_writercluster >> $S/score.log 2>&1 || DOK=0
$RUNS -c "
import pandas as pd, glob
n = sum(int((pd.read_csv(f)['error'].fillna('') != '').sum()) for f in glob.glob('$S/rows/*_rows.csv'))
print('row errors', n); open('$S/row_errors.txt','w').write(str(n))" >> $S/score.log 2>&1 || DOK=0
[ "$(cat $S/row_errors.txt 2>/dev/null)" = "0" ] || DOK=0
echo "$(date -Is) G4 decode/score ok=$DOK" | tee -a $S/smoke_status.txt
# ---- gate
$RUNG $C/smoke_check.py --smoke-dir $S --decode-ok $DOK > $S/smoke_check.log 2>&1 || fail "gate check (see smoke_gate.json)"
[ -n "$SIFPID" ] && wait $SIFPID
TL_H=$(python3 -c "import json,math; g=json.load(open('$S/smoke_gate.json')); print(min(96, max(16, math.ceil(g['eta_100k_h']*1.6 + 3))))")
echo "$(date -Is) all gates passed; full train time limit ${TL_H}h" | tee -a $S/smoke_status.txt
# ---- full chain (submitted only once)
if grep -q " train_full" $E/slurm_ids.txt 2>/dev/null; then
  fail "a train_full job is already recorded in slurm_ids.txt; not resubmitting"
fi
J1=$(sbatch --parsable --job-name=pennet_train --partition=gpuidle --gres=gpu:1 --cpus-per-task=8 --mem=96G \
     --time=${TL_H}:00:00 --exclude=$EXCL --requeue --open-mode=append --output=$E/logs/train_%j.out \
     --export=ALL,E=$E,C=$C $C/jobs/train_full.sh) || fail "sbatch train_full"
echo "$J1 train_full gpuidle 1gpu time=${TL_H}h submitted $(date -Is)" >> $E/slurm_ids.txt
J2=$(sbatch --parsable --job-name=pennet_score --partition=epyc-256 --cpus-per-task=16 --mem=60G --time=12:00:00 \
     --dependency=afterok:$J1 --output=$E/logs/score_%j.out --export=ALL,E=$E,C=$C $C/jobs/score_full.sh) || fail "sbatch score_full"
echo "$J2 score_full epyc-256 16cpu afterok:$J1 submitted $(date -Is)" >> $E/slurm_ids.txt
echo "submitted train $J1 score $J2" | tee -a $S/smoke_status.txt
sleep 90
squeue -j $J1,$J2 -o '%.10i %.12P %.14j %.8T %.10M %R' | tee $S/squeue_after_submit.txt
python3 $C/pennet_provenance.py --exp $E --code $C --mode smoke | tee -a $S/smoke_status.txt
cp $S/*.json $S/*.txt $S/*.log $S/*.csv $S/*.md $E/PROVENANCE_PENNET_smoke.json $E/slurm_ids.txt $WD/ 2>/dev/null
echo "$(date -Is) smoke driver end" | tee -a $S/smoke_status.txt
ls -lh $WD || true
