#!/bin/bash
# RQF R1: CLUSTER-A -> VPS staging for the CLUSTER-B copies. Manifest = sha256 of every file the CLUSTER-B copies need that is NOT already identical on CLUSTER-B (push set) + existing-on-CLUSTER-B files to verify.
set -u
R=$U; A=$R/exp/RQF/R1/stage; mkdir -p $A; K=$HOME/.ssh/uhwr_staging_vps_ed25519; V=Administrator@<ip>
SSHV="ssh -o BatchMode=yes -o ConnectTimeout=30 -i $K $V"
PUSH="exp/RQF/R1/data/rqf_mir_iam500.npz exp/RQF/R1/data/rqf_mir_casia_en.npz exp/RQF/R1/data/rqf_mir_hpl_tamil.npz exp/RQF/R1/data/PROVENANCE_R1_mirrored_sets.json exp/RQF/code/rqf1_v1/rqf1_sample.py exp/RQF/code/rqf1_v1/bb_frame_sample.py exp/RQF/code/rqf1_v1/r1_clusterB_inner.sh exp/RQF/code/rqf1_v1/clusterB_r1.sbatch exp/RQF/code/rqf1_v1/r1_lock.sh exp/FMD/data/renders_224/test.npz exp/FMD/xs/runs/NOFM_W_clusterA/bb_ms3.pt exp/FMD/xs/runs/PEINIT_W_clusterA/bb_ms4.pt"
VERIFY="exp/M5/runs/luo/checkpoints/checkpoint_000006.pt exp/M5/runs/joint/checkpoints/checkpoint_000009.pt runs/e0/e0_v3_rendered_100k_bs512/checkpoints/checkpoint_000008.pt exp/X4a/data/x4a_casia_en.npz exp/X4a/data/x4a_hpl_tamil.npz data/m5_multiscript_v1/m5_test_iam500.npz data/d2_v2/unipen_de/e0_test.npz data/d2_v2/onhw/e0_all.npz data/ouhdl_e0_v3/e0_test.npz exp/CAND/code/frame_code_a44e78b0/frame_sample.py exp/CAND/code/frame_code_a44e78b0/e0_frames.py exp/CAND/code/frame_code_a44e78b0/e0_dataset.py exp/CAND/code/frame_code_a44e78b0/e0_adapter.py exp/FMD/xs/code/fmdx_backbone_v1/bb_model.py exp/FMD/xs/code/fmdx_backbone_v1/bb_sample.py exp/FMD/xs/code/fmdx_backbone_v1/vendor/uhwr_route_common.py"
cd $R; : > $A/MANIFEST_push.sha256; : > $A/MANIFEST_verify.sha256
for f in $PUSH; do sha256sum $f >> $A/MANIFEST_push.sha256; done
for f in $VERIFY; do sha256sum $f >> $A/MANIFEST_verify.sha256; done
echo "$(date -u +%FT%TZ) manifests done"
for f in $PUSH; do
  d=$(dirname $f); $SSHV "powershell -NoProfile -Command \"New-Item -ItemType Directory -Force $VPS_ROOT/uhwr/stage/$d | Out-Null\""
  scp -q -i $K -o BatchMode=yes $R/$f $V:$VPS_ROOT/uhwr/stage/$f; echo "scp $f rc=$?"
  hl=$(grep " $f\$" $A/MANIFEST_push.sha256 | cut -c1-64); hv=$($SSHV "powershell -NoProfile -Command \"(Get-FileHash $VPS_ROOT/uhwr/stage/$f -Algorithm SHA256).Hash\"" | tr -d '\r' | tail -1 | tr A-F a-f)
  [ "$hl" = "$hv" ] && echo "push ok $f" || echo "PUSH SHA MISMATCH $f local=$hl vps=$hv"
done
echo "$(date -u +%FT%TZ) STAGE_VPS_DONE"
