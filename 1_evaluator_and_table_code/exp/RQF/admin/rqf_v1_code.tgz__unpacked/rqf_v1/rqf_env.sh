#!/bin/bash
# RQF round shared env (prereg PREREG_RQ_finalise_2026-11-07, sha256 04766e97...). Sources the FMDX common.sh unchanged.
source $U/exp/FMD/xs/code/fmdx_backbone_v1/jobs/common.sh
Q=$R/exp/RQF; QC=$Q/code/rqf_v1; QR=$Q/runs; QP=$Q/preds; QL=$Q/logs; QS=$Q/score_out
MON=$X/code/fmdx_monitor_v1; MD=$X/data/monitor_v1; MMR=$X/runs/fmdx_monitor
CAND=$R/exp/CAND; FCODE=$CAND/code/frame_code_a44e78b0; CCODE=$CAND/controls/code
rqf_log() { mkdir -p $Q/queue; echo "$(utc) $*" >> $Q/queue/rqf_log.md; }
SIX="test_casia_zh test_casia_en test_hpl_tamil test_iam500 german_unipen_de_test german_onhw_all"
