#!/bin/bash
# FMDX backbone smoke test (both clusters). Usage: smoke.sh OUTDIR LANE [PE_WEIGHTS]
# Tiny runs (first 3,000 LUO units, global batch 16 x accum 2): one per option type -> finite loss, DDP (2 GPUs), resume
# round-trip, milestone/inference file round-trip (FM-free load), GATE_FAIL at start and mid-run, sampling + scoring + selection,
# FT regime from a smoke milestone, race lock on the VPS hub, REPA target extraction on 16 units (CLUSTER-A only).
set -u
source $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh
SM=$1; LANE=$2; PEW=${3:-}; mkdir -p $SM; cd $SM
check_code || exit 3
NG=$(echo ${CUDA_VISIBLE_DEVICES:-} | tr ',' '\n' | grep -c .); [ "$NG" -ge 1 ] || NG=1
echo "$(utc) smoke start lane=$LANE host=$(hostname) gpus=[$(gpu_names)] ng=$NG"
# val250 renders for the sampling test (made here with the same gtset code if this lane has none, e.g. CLUSTER-B)
mkdir -p $DB
for s in casia_zh casia_en; do
  [ -f $DB/val250_${s}_render224.npz.meta.json ] || $PY $BB/prep_renders.py gtset --npz $D/m5_val_${s}.npz --out $DB/val250_${s}_render224.npz >> prep_val_render.log 2>&1
done
# p01 stand-in where the p01 224 renders are absent (CLUSTER-B): val250 casia_en npz + its npy (only tests the FT / pseudo code paths)
P01S=${SMOKE_P01_NPZ:-$P01_NPZ}; P01RS=${SMOKE_P01_R224:-$P01_R224}
# smoke render file: first 3,000 LUO units (decoded seq; the full rebuilt array may not exist yet)
SMR=$SM/luo_first3000_render224.npy
[ -f $SMR.meta.json ] || $PY $BB/prep_renders.py seqset --npz $LUO_NPZ --out $SMR --max-units 3000 --procs ${SLURM_CPUS_PER_TASK:-8} > prep_smoke_render.log 2>&1 || { echo "smoke render failed"; tail prep_smoke_render.log; exit 4; }
SMU=$SM/luo_first3000_render224__unit_ids.npy
TR="$WRAP $VENV/bin/python -m torch.distributed.run --standalone --nproc_per_node $NG --master_port $((29500 + ${SLURM_JOB_ID:-0} % 3000))"
COMMON=(--train-npz $LUO_NPZ --render-npy $SMR --max-units 3000 --batch-size 16 --grad-accum 2 --num-workers 2 --log-every 1 --val-every 3 --lane $LANE)
res() { echo "SMOKE $1 rc=$2 $(utc)" | tee -a $SM/smoke_results.txt; }
# 1 no-FM warm (DDP) with resume: 3 steps (stop-after), then resume to 6; milestones every 3
$TR $BB/bb_train.py --arm-id smoke_warm_nofm --run-dir $SM/warm_nofm --regime warm --init-ckpt $LUO_CKPT "${COMMON[@]}" --train-steps 6 --save-every 3 --resume-every 3 --stop-after 3 > warm_nofm_1.log 2>&1; res warm_nofm_part1 $?
$TR $BB/bb_train.py --arm-id smoke_warm_nofm --run-dir $SM/warm_nofm --regime warm --init-ckpt $LUO_CKPT "${COMMON[@]}" --train-steps 6 --save-every 3 --resume-every 3 > warm_nofm_2.log 2>&1; res warm_nofm_resume $?
# 2 DINOv3 REPA warm with dummy targets (random float16, D=1024, first 1,500 of the 3,000 units have targets)
$PY - <<'PY'
import numpy as np
u = np.load("luo_first3000_render224__unit_ids.npy", allow_pickle=False)[:1500]
rng = np.random.default_rng(0)
np.save("dummy_targets_dinov3.npy", rng.standard_normal((1500, 256, 1024)).astype(np.float16)); np.save("dummy_targets_unit_ids.npy", u)
print("dummy targets written")
PY
echo '{"fm":"dinov3-vitl16","status":"PENDING","note":"smoke fake verdict"}' > fake_verdict_pending.json
$TR $BB/bb_train.py --arm-id smoke_warm_dinov3 --run-dir $SM/warm_dinov3 --regime warm --init-ckpt $LUO_CKPT "${COMMON[@]}" --train-steps 6 --save-every 3 \
   --fm dinov3-vitl16 --verdict-file $SM/fake_verdict_pending.json --verdict-every 3 --repa-targets dummy_targets_dinov3.npy --repa-unit-ids dummy_targets_unit_ids.npy --repa-lambda 0.5 > warm_dinov3.log 2>&1; res warm_dinov3_repa_dummy $?
# 3 PE-init warm
if [ -n "$PEW" ] && [ -e "$PEW" ]; then
  $TR $BB/bb_train.py --arm-id smoke_warm_pe --run-dir $SM/warm_pe --regime warm --encoder pe_spatial_s16 --pe-weights $PEW --init-ckpt $LUO_CKPT "${COMMON[@]}" --train-steps 4 --save-every 2 > warm_pe.log 2>&1; res warm_pe_init $?
else res warm_pe_init_NO_WEIGHTS 99; fi
# 4 scratch
$TR $BB/bb_train.py --arm-id smoke_scratch --run-dir $SM$SCRATCH_ROOT --regime scratch "${COMMON[@]}" --train-steps 4 --save-every 2 > scratch.log 2>&1; res scratch $?
# 5 pseudo-label option, pooled (stand-in source: p01 train npz + its 224 renders)
$TR $BB/bb_train.py --arm-id smoke_pseudo --run-dir $SM/pseudo --regime warm --init-ckpt $LUO_CKPT "${COMMON[@]}" --train-steps 4 --save-every 2 \
   --pseudo-npz $P01S --pseudo-render-npy $P01RS > pseudo.log 2>&1; res pseudo_label_standin $?
# 6 GATE_FAIL at start and mid-run
echo '{"fm":"dinov3-vitl16","status":"FAIL","note":"smoke fake verdict"}' > fake_verdict_fail.json
$TR $BB/bb_train.py --arm-id smoke_gatefail_start --run-dir $SM/gatefail_start --regime warm --init-ckpt $LUO_CKPT "${COMMON[@]}" --train-steps 6 --save-every 3 \
   --fm dinov3-vitl16 --verdict-file $SM/fake_verdict_fail.json > gatefail_start.log 2>&1; rc=$?; [ -f gatefail_start/GATE_FAIL ] && [ ! -f gatefail_start/training_complete.json ]; res "gatefail_start(rc=$rc)" $?
cp fake_verdict_pending.json flip_verdict.json
( for i in $(seq 1 600); do grep -q '"step": 2' gatefail_mid/train_log.jsonl 2>/dev/null && { cp fake_verdict_fail.json flip_verdict.json; break; }; sleep 1; done ) &
$TR $BB/bb_train.py --arm-id smoke_gatefail_mid --run-dir $SM/gatefail_mid --regime warm --init-ckpt $LUO_CKPT "${COMMON[@]}" --train-steps 12 --save-every 4 \
   --fm dinov3-vitl16 --verdict-file $SM/flip_verdict.json --verdict-every 4 > gatefail_mid.log 2>&1; rc=$?; wait
[ -f gatefail_mid/GATE_FAIL ] && [ ! -f gatefail_mid/training_complete.json ]; res "gatefail_mid(rc=$rc)" $?
# 7 sampling (FM-free) + scoring + selection on 8 val250 units per set, 2 milestones of the no-FM warm run
mkdir -p warm_nofm/eval
for m in 1 2; do for s in casia_zh casia_en; do
  $PY $BB/bb_sample.py --ckpt warm_nofm/bb_ms${m}.pt --render $DB/val250_${s}_render224.npz --gt-npz $D/m5_val_${s}.npz --max-units 8 --seed 0 --out warm_nofm/eval/val__${s}__ms${m}_preds.npz >> sample.log 2>&1 || res sample_$m_$s 1
  $PY $C0/e0_score.py e0 --npz $D/m5_val_${s}.npz --preds warm_nofm/eval/val__${s}__ms${m}_preds.npz --method smoke_ms${m} --input rendered --out warm_nofm/eval/val__${s}__ms${m}_rows.csv --procs 4 >> score.log 2>&1 || res score_$m_$s 1
done; done
$PY $BB/vendor/m5_select.py --eval-dir warm_nofm/eval --valsets casia_zh,casia_en --out warm_nofm/eval/selection.json > select.log 2>&1; res sample_score_select $?
$PY - <<'PY' >> $SM/smoke_results.txt 2>&1
import sys, torch
sys.path.insert(0, "$U/exp/FMD/xs/code/fmdx_backbone_v1"); sys.path.insert(0, "$U/exp/FMD/xs/code/fmdx_backbone_v1/vendor")
import uhwr_route_common as U, bb_model as BM
ck = torch.load("warm_nofm/bb_ms2.pt", map_location="cpu", weights_only=False)
bad = [k for k in ck if k in ("projector", "opt", "ema", "trainable")]
unet, _ = BM.load_backbone_for_inference("warm_nofm/bb_ms2.pt", U, "cpu")
ck3 = torch.load("warm_dinov3/bb_ms2.pt", map_location="cpu", weights_only=False)
proj_keys = [k for k in ck3["ema_unet"] if "proj" in k and k.startswith("encoder") is False and k.startswith("net.")]
print("INFER_CHECK keys", sorted(ck), "bad", bad, "dinov3 file keys", sorted(ck3), "projector_like_keys", proj_keys[:3],
      "transformers_loaded", "transformers" in sys.modules, "n_params", sum(p.numel() for p in unet.parameters()))
PY
# 8 FT regime from the smoke milestone (p01, identity frame), 1 process
$PY $BB/bb_train.py --arm-id smoke_ft --run-dir $SM/ft --regime ft --init-ckpt $SM/warm_nofm/bb_ms2.pt --train-npz $P01S --render-npy $P01RS \
   --batch-size 16 --grad-accum 2 --num-workers 2 --log-every 1 --val-every 2 --train-steps 4 --save-every 2 --seed 20261034 --lane $LANE > ft.log 2>&1; res ft_regime $?
# 9 race lock on the VPS hub (skipped inside the CLUSTER-B container: tested by clusterB_smoke.sbatch on the host)
#   same arm id twice from this lane (WIN, WIN as owner) and once as the other lane (LOSE)
if command -v ssh >/dev/null 2>&1; then
other=$([ $LANE = clusterA ] && echo clusterB || echo clusterA)
T="SMOKE_${LANE}_${SLURM_JOB_ID:-x}"
a=$(race_lock $T $LANE); b=$(race_lock $T $LANE); c=$(race_lock $T $other)
echo "RACE_LOCK first=$a again=$b other_lane=$c" | tee -a $SM/smoke_results.txt
[ "$a" = WIN ] && [ "$b" = WIN ] && [[ "$c" == LOSE* ]]; res race_lock $?
$SSHV "cmd /c rmdir /s /q $VPS_ROOT\\uhwr\\stage\\exp\\FMD\\xs\\race\\$T" >/dev/null 2>&1
else res race_lock_no_ssh_in_container 98; fi
# 10 REPA target extraction on 16 units (CLUSTER-A: FM venv)
if [ $LANE = clusterA ]; then
  python3 -c "import numpy as np; np.save('$SM/subset16.npy', np.arange(16, dtype=np.int64))"
  echo '{"fm":"dinov3-vitl16","status":"PENDING","note":"smoke"}' > vd_pending/verdict_dinov3-vitl16.json 2>/dev/null || { mkdir -p vd_pending; echo '{"fm":"dinov3-vitl16","status":"PENDING"}' > vd_pending/verdict_dinov3-vitl16.json; }
  $WRAPFM FMD_HARNESS=$HC HF_HOME=$R/exp/FMD/weights/.hf_home_student PYTHONPATH=$BB:$HC:$R/exp/FMD/code/fmd_harness_xs:$UE:$X/pylibs $FMDVENV/bin/python -u $BB/prep_repa_targets.py --fm dinov3-vitl16 \
     --render-npy $SMR --render-unit-ids $SMU --subset-pos $SM/subset16.npy --subset-name smoke16 --out-dir $SM/targets --verdict-dir $SM/vd_pending --chunk 8 > targets.log 2>&1; res repa_targets_dinov3_16 $?
  mkdir -p vd_fail; cp fake_verdict_fail.json vd_fail/verdict_dinov3-vitl16.json
  $WRAPFM FMD_HARNESS=$HC PYTHONPATH=$BB:$HC:$R/exp/FMD/code/fmd_harness_xs:$UE:$X/pylibs $FMDVENV/bin/python -u $BB/prep_repa_targets.py --fm dinov3-vitl16 --render-npy $SMR --render-unit-ids $SMU \
     --subset-pos $SM/subset16.npy --subset-name smoke16b --out-dir $SM/targets_fail --verdict-dir $SM/vd_fail > targets_fail.log 2>&1; [ -f targets_fail/dinov3-vitl16/SKIPPED_GATE_FAIL.json ]; res repa_targets_skip_on_fail $?
fi
grep -h '"event": "train"' */train_log.jsonl | python3 -c "
import sys, json, math
r=[json.loads(l) for l in sys.stdin]; bad=[x for x in r if not x.get('finite')]
print('LOSS_FINITE n_records', len(r), 'non_finite', len(bad), 'den_range', min(x['loss_den'] for x in r), max(x['loss_den'] for x in r))" | tee -a $SM/smoke_results.txt
echo "$(utc) smoke end"; cat $SM/smoke_results.txt
