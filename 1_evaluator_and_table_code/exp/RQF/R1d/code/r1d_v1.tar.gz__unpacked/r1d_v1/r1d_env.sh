#!/bin/bash
# RQF A2.5 R1d shared paths (sourced). Absolute paths only; no sibling env sourced.
R=$U; Q=$R/exp/RQF; RD=$Q/R1d; RC=$RD/code/r1d_v1; X=$R/exp/FMD/xs; BB=$X/code/fmdx_backbone_v1
FCODE=$R/exp/CAND/code/frame_code_a44e78b0; REPO=$R/repos/nagamatsu; HC=$R/exp/FMD/code/fmd_harness_v1; C0=$R/e0/code_4cd40f57
UE=$R/shared/uhwr_eval-0.3.1; TIMMDEPS=$X/code/pydeps_timm; IMG=$R/images/torch27.sif; VE=$R/venvs/e0; VEV=$R/venvs/uhwr_eval02
ENVB="PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONHASHSEED=0 WANDB_MODE=disabled PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID"
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
r1d_log() { echo "$(utc) [rqf-R1d, <agent>] $*" >> $Q/queue/rqf_log.md; }
