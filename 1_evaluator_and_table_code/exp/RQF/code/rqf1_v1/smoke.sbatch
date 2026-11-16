#!/bin/bash
#SBATCH --job-name=rqf1_smoke
#SBATCH --partition=epyc-256
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=$U/exp/RQF/R1/logs/%x_%j.out
exec 2>&1
R=$U; FC=$R/exp/CAND/code/frame_code_a44e78b0; Q=$R/exp/RQF/code/rqf1_v1; O=$R/exp/RQF/R1/smoke; IMG=$R/images/torch27.sif; VE=$R/venvs/e0; REPO=$R/repos/nagamatsu
X=$R/exp/FMD/xs; BB=$X/code/fmdx_backbone_v1; HC=$R/exp/FMD/code/fmd_harness_v1; C0=$R/e0/code_4cd40f57; UE=$R/shared/uhwr_eval-0.3.1
PYG="apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env PYTHONPATH=$FC/lib:$REPO/src:$FC:$Q OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 $VE/bin/python"
$PYG $Q/rqf1_sample.py --label LUO_ms6 --run-dir exp/M5/runs/luo --milestone 6 --stage $R --out-dir $O --batch-size 4 --seeds 0 --max-units 4 --sets test_iam500_mir test_casia_en test_casia_en_mir german_onhw_all --frames identity mirror_x 2>&1 | grep -av 'sampling loop' | tail -6; echo rc1=${PIPESTATUS[0]}
PYB="apptainer exec -B $SCRATCH_ROOT -B /tmp $IMG env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BB_DIR=$BB FCODE=$FC PYTHONPATH=$REPO/src:$BB:$BB/vendor:$FC:$HC:$C0:$UE OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 $VE/bin/python"
$PYB $Q/bb_frame_sample.py --ckpt $X/runs/NOFM_W_clusterA/bb_ms3.pt --label NOFM_W_bb3 --set test_ouhdl --render $R/exp/FMD/data/renders_224/test.npz --gt-npz $R/data/ouhdl_e0_v3/e0_test.npz --out-dir $O --batch-size 4 --seeds 0 --max-units 4 --frames identity mirror_x 2>&1 | grep -av 'sampling loop' | tail -6; echo rc2=${PIPESTATUS[0]}
ls $O/preds | head -20
