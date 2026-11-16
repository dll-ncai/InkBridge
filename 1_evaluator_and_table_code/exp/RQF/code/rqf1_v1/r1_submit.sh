#!/bin/bash
# RQF R1: submit the whole chain (sampling -> scoring array -> tables). Run once; prints ids.
set -u
O=$U/exp/RQF/R1; Q=$U/exp/RQF/code/rqf1_v1; L=$U/exp/RQF/queue/rqf_log.md
utc() { date -u +%FT%TZ; }
mkdir -p $O/tasks; cd $O
for b in LUO_ms6:R1a_LUO M5_ms9:R1a_M5; do lab=${b%%:*}; j=${b##*:}; : > tasks/$j.txt
  for s in test_iam500 test_casia_en test_hpl_tamil; do echo "${s}__${lab}_identity__seed*_preds.npz|$j" >> tasks/$j.txt
    for fr in identity mirror_x; do echo "${s}_mir__${lab}_${fr}__seed*_preds.npz|$j" >> tasks/$j.txt; done; done; done
: > tasks/R1b_E0.txt
for s in german_unipen_de_test test_iam500 test_casia_en german_onhw_all; do for fr in identity mirror_x; do echo "${s}__E0_ms8_${fr}__seed*_preds.npz|R1b_E0" >> tasks/R1b_E0.txt; done; done
for b in NOFM_W_bb3:R1c_NOFM PEINIT_W_bb4:R1c_PEINIT; do lab=${b%%:*}; j=${b##*:}; : > tasks/$j.txt
  for fr in identity mirror_x; do echo "test_ouhdl__${lab}_${fr}__seed*_preds.npz|$j" >> tasks/$j.txt; done; done
declare -A SID AID; ALL=""
for j in R1a_LUO R1a_M5 R1b_E0 R1c_NOFM R1c_PEINIT; do
  id=$(sbatch --parsable -J rqf1_$j -p gpuidle --gres=gpu:a4000:1 $Q/gpu.sbatch $j); SID[$j]=$id
  echo "$(utc) [rqf-s1, <agent>] SUBMIT R1 sampling $j job=$id lane=clusterA:gpuidle:a4000 deps=none" >> $L
  n=$(wc -l < tasks/$j.txt)
  a=$(TASKS=$O/tasks/$j.txt sbatch --parsable -J rqf1_score_$j --array=1-$n --dependency=afterok:$id --kill-on-invalid-dep=yes --export=ALL,TASKS $Q/score.sbatch); AID[$j]=$a; ALL="$ALL:$a"
  echo "$(utc) [rqf-s1, <agent>] SUBMIT R1 scoring $j job=$a lane=clusterA:epyc-256 deps=afterok:$id" >> $L
  echo "$j sample=$id score=$a tasks=$n"
done
t=$(sbatch --parsable -J rqf1_tables --dependency=afterany$ALL --kill-on-invalid-dep=yes $Q/tables.sbatch)
echo "$(utc) [rqf-s1, <agent>] SUBMIT R1 tables job=$t lane=clusterA:epyc-256 deps=afterany$ALL" >> $L
echo "tables=$t"
python3 - <<PY
import json
d={"sampling":{j:"${SID[R1a_LUO]}" for j in []}}
d={"sampling":{"R1a_LUO":"${SID[R1a_LUO]}","R1a_M5":"${SID[R1a_M5]}","R1b_E0":"${SID[R1b_E0]}","R1c_NOFM":"${SID[R1c_NOFM]}","R1c_PEINIT":"${SID[R1c_PEINIT]}"},
   "scoring":{"R1a_LUO":"${AID[R1a_LUO]}","R1a_M5":"${AID[R1a_M5]}","R1b_E0":"${AID[R1b_E0]}","R1c_NOFM":"${AID[R1c_NOFM]}","R1c_PEINIT":"${AID[R1c_PEINIT]}"},"tables":"$t","build":"24964033","smoke":"24964260"}
json.dump(d,open("$O/logs/R1_jobs.json","w"),indent=1)
PY
