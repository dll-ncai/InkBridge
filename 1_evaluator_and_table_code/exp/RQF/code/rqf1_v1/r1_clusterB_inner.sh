#!/bin/bash
# RQF R1 sampling inside the CLUSTER-B Pyxis container ($U_B bind-mounted at $U). Same sampler calls as r1_job.sh; out dir exp/RQF/R1/clusterB_out.
set -uo pipefail; exec 2>&1
JOB=$1
R=$U; FC=$R/exp/CAND/code/frame_code_a44e78b0; Q=$R/exp/RQF/code/rqf1_v1; O=$R/exp/RQF/R1/clusterB_out
VE=$R/venvs/e0; REPO=$R/repos/nagamatsu; X=$R/exp/FMD/xs; BB=$X/code/fmdx_backbone_v1
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
mkdir -p $O/preds
GPUN=$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1)
echo "$(utc) $JOB(clusterB) start host=$(hostname) slurm=${SLURM_JOB_ID:-x} gpu=[$GPUN]" | tee -a $O/job_history_R1_clusterB.txt
PYG="env CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 PYTHONPATH=$FC/lib:$REPO/src:$FC:$Q OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled PYTHONDONTWRITEBYTECODE=1 $VE/bin/python"
S="$PYG $Q/rqf1_sample.py --stage $R --out-dir $O --batch-size 256 --seeds 0 1 2"
fs() { grep -av "sampling loop" | tail -3; }
rc=0
case $JOB in
 R1a_LUO|R1a_M5)
  if [ $JOB = R1a_LUO ]; then A="--label LUO_ms6 --run-dir exp/M5/runs/luo --milestone 6"; else A="--label M5_ms9 --run-dir exp/M5/runs/joint --milestone 9"; fi
  $S $A --sets test_iam500 test_casia_en test_hpl_tamil --frames identity 2>&1 | fs; r1=${PIPESTATUS[0]}
  $S $A --sets test_iam500_mir test_casia_en_mir test_hpl_tamil_mir --frames identity mirror_x 2>&1 | fs; r2=${PIPESTATUS[0]}; rc=$((r1+r2));;
 R1b_E0)
  $S --label E0_ms8 --run-dir runs/e0/e0_v3_rendered_100k_bs512 --milestone 8 --sets german_unipen_de_test test_iam500 test_casia_en german_onhw_all --frames identity mirror_x 2>&1 | fs; rc=${PIPESTATUS[0]};;
 R1c_NOFM|R1c_PEINIT)
  if [ $JOB = R1c_NOFM ]; then CK=$X/runs/NOFM_W_clusterA/bb_ms3.pt; LAB=NOFM_W_bb3; else CK=$X/runs/PEINIT_W_clusterA/bb_ms4.pt; LAB=PEINIT_W_bb4; fi
  HC=$R/exp/FMD/code/fmd_harness_v1; C0=$R/e0/code_4cd40f57; UE=$R/shared/uhwr_eval-0.3.1
  PYB="env CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BB_DIR=$BB FCODE=$FC PYTHONPATH=$REPO/src:$BB:$BB/vendor:$FC:$HC:$C0:$UE OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONDONTWRITEBYTECODE=1 $VE/bin/python"
  $PYB $Q/bb_frame_sample.py --ckpt $CK --label $LAB --set test_ouhdl --render $R/exp/FMD/data/renders_224/test.npz --gt-npz $R/data/ouhdl_e0_v3/e0_test.npz --out-dir $O --batch-size 256 --seeds 0 1 2 --frames identity mirror_x 2>&1 | fs; rc=${PIPESTATUS[0]};;
 *) echo "unknown job $JOB"; exit 9;;
esac
echo "$(utc) $JOB(clusterB) end rc=$rc gpu=[$GPUN]" | tee -a $O/job_history_R1_clusterB.txt
exit $rc
