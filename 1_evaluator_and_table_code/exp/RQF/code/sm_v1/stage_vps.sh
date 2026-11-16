#!/bin/bash
# RQF A2.3 staging CLUSTER-A -> VPS (tar over ssh; sha256 manifest verified on the VPS side by sha256 listing pulled back)
set -u; R=$U; L=$R/exp/RQF/logs/sm_stage_vps.log
SSHV="ssh -o BatchMode=yes -o ConnectTimeout=20 -i $HOME/.ssh/uhwr_staging_vps_ed25519 Administrator@<ip>"
echo "$(date -u +%FT%TZ) start" >> $L
cd $R/exp/RQF/code && tar -cf - --exclude='slurm-*' sm_v1 | $SSHV "tar -xf - -C $VPS_ROOT/uhwr/code" >> $L 2>&1 && echo "$(date -u +%FT%TZ) sm_v1 ok" >> $L
cd $R/exp/D2_MOE/code && tar -cf - --exclude='__pycache__' --exclude='slurm-*' d2moe_code_9a020b75 | $SSHV "tar -xf - -C $VPS_ROOT/uhwr/code" >> $L 2>&1 && echo "$(date -u +%FT%TZ) d2moe ok" >> $L
cd $R && tar -cf - exp/FMD/xs/runs/NOFM_W_clusterA/bb_ms3.pt exp/FMD/xs/runs/PEINIT_W_clusterA/bb_ms4.pt exp/RQF/data/render224/p10train_render224.npy exp/RQF/data/render224/p10train_render224.npy.meta.json exp/RQF/data/render224/PROVENANCE_render224_p10.json | $SSHV "tar -xf - -C $VPS_ROOT/uhwr/stage" >> $L 2>&1 && echo "$(date -u +%FT%TZ) data ok" >> $L
sha256sum exp/FMD/xs/runs/NOFM_W_clusterA/bb_ms3.pt exp/FMD/xs/runs/PEINIT_W_clusterA/bb_ms4.pt exp/RQF/data/render224/p10train_render224.npy > $R/exp/RQF/admin/sm_stage_vps.sha256
echo "$(date -u +%FT%TZ) done" >> $L

