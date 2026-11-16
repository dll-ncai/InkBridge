# M5 prep: install code e33511dd, env records, data build smoke (6 CASIA files, 300/40 units) then full m5_multiscript_v1
set -uo pipefail
TAG=e33511dd
B=$U; M=$B/exp/M5; MC=$M/code_$TAG; D=$B/data/m5_multiscript_v1
mkdir -p $M/prep $M/env $M/jobs $M/code_tarballs $M/smoke
echo "$(date -u +%FT%TZ) prep_start host=$(hostname) slurm=${SLURM_JOB_ID:-none} partition=${SLURM_JOB_PARTITION:-none}" >> $M/prep/job_history.txt
cp ./cmd.sh $M/jobs/cmd_prep_${SLURM_JOB_ID:-x}.sh 2>/dev/null || cp "$0" $M/jobs/cmd_prep_${SLURM_JOB_ID:-x}.sh 2>/dev/null || true
echo "e33511dd7c5cf6" >/dev/null
GOT=$(sha256sum m5_code_e33511dd.tar.gz | cut -c1-64); echo "tarball sha256 $GOT"
[ "${GOT:0:8}" = "$TAG" ] || { echo "tarball sha mismatch"; exit 3; }
cp m5_code_e33511dd.tar.gz $M/code_tarballs/
if [ ! -d $MC ]; then mkdir -p $MC && tar -xzf m5_code_e33511dd.tar.gz -C $MC; fi
cp $B/exp/e0q/code/code_4cd40f57_sha256.txt $MC/code_4cd40f57_sha256.txt
(cd $MC && sha256sum -c m5_code_sha256.txt) || exit 3
(cd $B/e0/code_4cd40f57 && sha256sum -c --quiet $MC/code_4cd40f57_sha256.txt) && echo "E0 eval code 4cd40f57 verified"
echo "e0_train.py (base of m5_train.py) sha256: $(sha256sum $B/e0/code_4cd40f57/e0_train.py | cut -c1-64) (expected 640ad9067b9e313e3356375721ee4d9f051b7b1c0503dd9ca137306b88ceb774)" | tee $M/env/m5_train_base.txt
diff <(sed -n '/^def main/,$p' $B/e0/code_4cd40f57/e0_train.py) <(sed -n '/^def main/,$p' $MC/m5_train.py) > $M/env/m5_train_vs_e0_train_main.diff || true
IMG=$B/images/torch27.sif; VENV=$B/venvs/e0
apptainer exec -B $SCRATCH_ROOT $IMG $VENV/bin/pip freeze > $M/env/pip_freeze_e0.txt
[ -s $M/env/torch27_sif_sha256.txt ] || sha256sum $IMG > $M/env/torch27_sif_sha256.txt
git -C $B/repos/nagamatsu rev-parse HEAD > $M/env/repo_commit.txt || true
cat $M/env/torch27_sif_sha256.txt $M/env/repo_commit.txt
RUNCPU="apptainer exec -B $SCRATCH_ROOT $IMG env PYTHONPATH=$MC:$B/e0/code_4cd40f57 OMP_NUM_THREADS=1 $VENV/bin/python"
# ---- smoke build
SD=$M/smoke/data_build
CMD_S="m5_build.py --data $B/data --x4a $B/exp/X4a/data --out $SD --procs 16 --n-train 300 --n-val 40 --limit-files 6"
$RUNCPU $MC/$CMD_S 2>&1 | tail -25 || { echo SMOKE_BUILD_FAILED; exit 1; }
$RUNCPU $MC/m5_datacard.py --out $SD --code-dir $MC --code-tarball $M/code_tarballs/m5_code_e33511dd.tar.gz --env-dir $M/env --command "$CMD_S" || exit 1
python3 - <<PY || { echo SMOKE_CHECK_FAILED; exit 1; }
import json, numpy as np
s = json.load(open("$SD/build_stats_M5.json"))
assert all(e[-1] == "n_errors=0" for e in s["errors"].values()), s["errors"]
z = np.load("$SD/m5_joint_train.npz"); print("smoke joint", {k: z[k].shape for k in z.files}); assert z["cond_rendered"].shape[1:] == (64, 64)
assert np.diff(z["seq_offsets"]).max() <= 320
print("smoke ok")
PY
echo "$(date -u +%FT%TZ) smoke_build_ok" >> $M/prep/job_history.txt
# ---- full build
CMD_F="m5_build.py --data $B/data --x4a $B/exp/X4a/data --out $D --procs 16 --n-train 20000 --n-val 250"
if [ ! -s $D/PROVENANCE_m5_multiscript_v1.json ]; then
  $RUNCPU $MC/$CMD_F 2>&1 | tail -40 || { echo FULL_BUILD_FAILED; exit 1; }
  $RUNCPU $MC/m5_datacard.py --out $D --code-dir $MC --code-tarball $M/code_tarballs/m5_code_e33511dd.tar.gz --env-dir $M/env --command "$CMD_F" || exit 1
fi
echo "$(date -u +%FT%TZ) full_build_ok" >> $M/prep/job_history.txt
mkdir -p out; cp $D/DATA_CARD.md $D/PROVENANCE_m5_multiscript_v1.json $D/seqlen_check_M5.csv $D/build_stats_M5.json $D/ft_subsets_M5.json out/ || true
cp $M/env/m5_train_vs_e0_train_main.diff $M/env/m5_train_base.txt $M/prep/job_history.txt out/ || true
ls -lh $D out
