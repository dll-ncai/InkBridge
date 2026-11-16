#!/bin/bash
# CLUSTER-A CPU smoke of the sm_v1 python paths (no GPU): identity check + 3 training steps + 4-unit sampling (2 DDIM steps) for both backbone kinds.
set -uo pipefail; exec 2>&1
source $U/exp/RQF/code/sm_v1/sm_env.sh
PYC="apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env $ENVB PYTHONPATH=$SMC:$PYP OMP_NUM_THREADS=8 $VENV/bin/python"
S=$Q/runs/_smoke_sm_cpu_${SLURM_JOB_ID:-x}; mkdir -p $S
NB=$X/runs/NOFM_W_clusterA/bb_ms3.pt
for K in luo bb; do
  B=$LUO_CKPT; RA=(); RR=(); [ $K = bb ] && { B=$NB; RA=(--render-npy $P01_R224); RR=(--render $R/exp/FMD/data/fmd_render224_test.npz); }
  echo "== $K identity"; $PYC $SMC/sma_identity_check.py $K $B || exit 11
  echo "== $K train 3 steps"; $PYC $SMC/sma_train.py --backbone-kind $K "${RA[@]}" --stage 1 --run-dir $S/$K --backbone $B --train-npz $P01_NPZ --scripts ouhdl --script-override ouhdl \
     --n-experts 1 --rank 16 --lora-alpha 16 --lora-targets cross,attn,linear --lb-coef 0 --batch-size 4 --grad-accum 1 --train-steps 3 --save-every 3 --val-every 100 \
     --learning-rate 1e-4 --keep-last 0 --num-workers 0 --seed 20261034 --device cpu --no-amp --log-every 1 | tail -4 || exit 12
  echo "== $K sample"; $PYC $SMC/sma_sample.py --kind $K --ckpt $S/$K/checkpoints/checkpoint_000001.pt --backbone $B --gt-npz $R/data/ouhdl_e0_v3/e0_test.npz "${RR[@]}" \
     --set-name test_ouhdl --route $ROUT/manifest_test_ouhdl.csv --max-units 4 --sampling-timesteps 2 --batch-size 4 --out $S/$K/test_ouhdl__SMOKE_${K}_ft_ms1__seed0_preds.npz || exit 13
  $PYC -c "import json;m=json.load(open('$S/$K/test_ouhdl__SMOKE_${K}_ft_ms1__seed0_preds.json'));print('routing',m['routing'])"
done
echo "== manifest unit coverage"
$PYC - <<'PY'
import numpy as np, csv
R="$U"
G={"test_ouhdl":R+"/data/ouhdl_e0_v3/e0_test.npz","test_casia_zh":R+"/exp/X4a/data/x4a_casia_zh.npz","test_casia_en":R+"/exp/X4a/data/x4a_casia_en.npz","test_hpl_tamil":R+"/exp/X4a/data/x4a_hpl_tamil.npz",
   "test_iam500":R+"/data/m5_multiscript_v1/m5_test_iam500.npz","german_unipen_de_test":R+"/data/d2_v2/unipen_de/e0_test.npz","german_onhw_all":R+"/data/d2_v2/onhw/e0_all.npz"}
for s,p in G.items():
    ids=np.load(p)["unit_ids"].astype(str); m={r["unit_id"]:float(r["p_rtl"]) for r in csv.DictReader(open(f"{R}/exp/D3_ROUTE/router/manifest_{s}.csv"))}
    print(s, "n", len(ids), "missing", sum(u not in m for u in ids), "adapter_on", sum(m.get(u,0)>=0.5 for u in ids))
PY
for f in smr_warm.sh smr_luo.sh sma_chain.sh ft_job_smr.sh sm_env.sh; do bash -n $SMC/$f && echo "syntax ok $f"; done
echo SMOKE_CPU_DONE
