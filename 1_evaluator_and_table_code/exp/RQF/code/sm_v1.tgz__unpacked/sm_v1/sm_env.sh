#!/bin/bash
# RQF A2.3 one-model arms (SM-R / SM-A) shared env. Sources rqf_v1/rqf_env.sh (-> FMDX common.sh) unchanged.
source $U/exp/RQF/code/rqf_v1/rqf_env.sh
SMC=$Q/code/sm_v1; ROUT=$R/exp/D3_ROUTE/router
SETS7="test_ouhdl $SIX"
# set -> gt npz | render224 (bb kind) ; same files as rqf_v1/warm_ft.sh and post_run.sh GTMAP
gt_of() { case $1 in test_ouhdl) echo $R/data/ouhdl_e0_v3/e0_test.npz;; test_casia_zh) echo $R/exp/X4a/data/x4a_casia_zh.npz;; test_casia_en) echo $R/exp/X4a/data/x4a_casia_en.npz;;
  test_hpl_tamil) echo $R/exp/X4a/data/x4a_hpl_tamil.npz;; test_iam500) echo $R/data/m5_multiscript_v1/m5_test_iam500.npz;; german_unipen_de_test) echo $R/data/d2_v2/unipen_de/e0_test.npz;;
  german_onhw_all) echo $R/data/d2_v2/onhw/e0_all.npz;; esac; }
r224_of() { case $1 in test_ouhdl) echo $R/exp/FMD/data/fmd_render224_test.npz;; *) echo $MD/test_${1}_render224.npz;; esac; }
