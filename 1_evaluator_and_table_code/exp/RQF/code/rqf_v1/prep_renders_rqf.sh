#!/bin/bash
#SBATCH -p cpuidle
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o $U/exp/RQF/logs/prep_render_%x_%j.out
# RQF R2c: 224-px train renders for the p10 / p30 Urdu fine-tune subsets (m5_ft_ouhdl_p10/p30.npz, existing subset files), with the SAME code that built the p01 renders
# (fmd_student_v1/prep_renders.py --p01 <npz>: OUHD-L v3 ligature strokes -> e0_adapter.unit_eval_ink -> render224, thickness 7; byte-identity check vs the subset's cond_rendered) + FMDX npz2npy.
# Usage: sbatch -J prep_<FRAC> prep_renders_rqf.sh p10|p30
set -uo pipefail; exec 2>&1
FRAC=$1; source $U/exp/FMD/code/fmd_student_v1/jobs/common.sh
source $U/exp/RQF/code/rqf_v1/rqf_env.sh
OUTD=$Q/data/render224/${FRAC}_harness; mkdir -p $OUTD; NPZ=$R/data/m5_multiscript_v1/m5_ft_ouhdl_${FRAC}.npz
echo "$(utc) start $FRAC host=$(hostname)"
apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env $ENVB OMP_NUM_THREADS=8 PYTHONPATH=$UE:$HC:$R/data/ouhdl_ligatures_v3/build:$R/data/ouhdl_ligatures_v3:$R/e0/code_2be9e593:$STU/vendor \
  $PYE0 $STU/prep_renders.py --p01 $NPZ --out-dir $OUTD > $OUTD/prep_renders.json 2> $OUTD/prep_renders.err || { tail -20 $OUTD/prep_renders.err; exit 4; }
(cd $OUTD && mv fmd_render224_p01train.npz fmd_render224_${FRAC}train.npz && mv fmd_render224_p01train.npz.meta.json fmd_render224_${FRAC}train.npz.meta.json && mv fmd_render224_p01train_mx.npz fmd_render224_${FRAC}train_mx.npz 2>/dev/null; mv fmd_render224_p01train_mx.npz.meta.json fmd_render224_${FRAC}train_mx.npz.meta.json 2>/dev/null; true)
$PY $BB/prep_renders.py npz2npy --npz $NPZ --render-npz $OUTD/fmd_render224_${FRAC}train.npz --out $Q/data/render224/${FRAC}train_render224.npy || exit 5
python3 - $FRAC <<'PY'
import hashlib, json, sys, os, glob
f = sys.argv[1]; R = "$U"; Q = R + "/exp/RQF"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as g:
        for b in iter(lambda: g.read(1 << 24), b""): h.update(b)
    return h.hexdigest()
ins = {"subset_npz": R + "/data/m5_multiscript_v1/m5_ft_ouhdl_%s.npz" % f, "student_prep_renders.py": R + "/exp/FMD/code/fmd_student_v1/prep_renders.py", "fmdx_prep_renders.py": R + "/exp/FMD/xs/code/fmdx_backbone_v1/prep_renders.py"}
outs = {"harness_npz": Q + "/data/render224/%s_harness/fmd_render224_%strain.npz" % (f, f), "npy": Q + "/data/render224/%strain_render224.npy" % f}
d = {"schema": "rqf_provenance/1", "kind": "rqf_render224_train_subset", "fraction": f, "created_utc": os.popen("date -u +%FT%TZ").read().strip(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
     "transform": "same code as p01 render (fmd_student_v1 prep_renders.py, ligatures_v3 strokes -> unit_eval_ink -> render224 thickness 7); npz2npy in the order of the subset npz",
     "subset_origin": "existing writer-stratified subset file m5_ft_ouhdl_%s.npz (built in M5, not rebuilt here)" % f,
     "inputs": {k: {"path": v, "sha256": sha(v)} for k, v in ins.items()}, "outputs": {k: {"path": v, "sha256": sha(v)} for k, v in outs.items()},
     "prep_report": json.load(open(Q + "/data/render224/%s_harness/prep_renders.json" % f)) if os.path.exists(Q + "/data/render224/%s_harness/prep_renders.json" % f) else None}
json.dump(d, open(Q + "/data/render224/PROVENANCE_render224_%s.json" % f, "w"), indent=1); print("provenance written")
PY
echo "$(utc) end $FRAC"
