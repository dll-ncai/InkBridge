#!/bin/bash
# RQF A2.5 R1d chain submitter (login node). Per sampling unit: two race copies (gpuidle A4000 | gpuidle H100).
# Per base: apply (afterany: that base's copies; fails if a pool file is missing) -> score array x7 (afterok) -> read (afterok); summary (afterany reads).
set -u
source $U/exp/RQF/R1d/code/r1d_v1/r1d_env.sh
SB="env -i HOME=$HOME USER=$USER PATH=/usr/bin:/bin /usr/bin/sbatch --parsable"
mkdir -p $RD/logs $RD/race $RD/queue; CSV=$RD/queue/r1d_queue.csv; [ -s $CSV ] || echo "exp,arm,lane,job_id,depends_on,outputs_path" > $CSV
SIX="test_casia_zh test_casia_en test_hpl_tamil test_iam500 german_unipen_de_test german_onhw_all"
GA="-p gpuidle --gres=gpu:a4000:1 -c 8 --mem=24G"; GH="-p gpuidle --gres=gpu:h100:1 -c 8 --mem=32G"; OPT="--requeue --open-mode=append -o $RD/logs/%x_%j.out"
declare -A DEP
race() { # race UNIT BASE TIME ARGS...
  local u=$1 b=$2 t=$3; shift 3
  local a=$($SB -J r1d_$u $GA -t $t $OPT $RC/samp.sh $u "$@"); local h=$($SB -J r1d_$u $GH -t $t $OPT $RC/samp.sh $u "$@")
  echo "$a $h" > $RD/race/$u.ids; DEP[$b]="${DEP[$b]:-}:$a:$h"
  r1d_log "SUBMIT R1d samp $u job=$a lane=clusterA:gpuidle-a4000 race_with=$h"; r1d_log "SUBMIT R1d samp $u job=$h lane=clusterA:gpuidle-h100 race_with=$a"
  echo "R1d,samp_$u,clusterA:gpuidle-a4000,$a,race:$h,$RD/preds/$b" >> $CSV; echo "R1d,samp_$u,clusterA:gpuidle-h100,$h,race:$a,$RD/preds/$b" >> $CSV; }
race LUO6 LUO_ms6 08:00:00 e0 LUO_ms6 exp/M5/runs/luo 6 $RD/preds/LUO_ms6 mirror_x $SIX
race M5u M5_ms9 06:00:00 e0 M5_joint_ms9 exp/M5/runs/joint 9 $RD/preds/M5_ms9 mirror_x test_ouhdl
race M56 M5_ms9 08:00:00 e0 M5_joint_ms9 exp/M5/runs/joint 9 $RD/preds/M5_ms9 mirror_x $SIX
for b in NOFM_W_bb PEINIT_W_bb; do
  read LAB CK < <(python3 -c "import json;e=json.load(open('$RD/plan.json'))['bases']['$b'];print(e['label'],e['ckpt'])")
  T6=$(python3 -c "import json;e=json.load(open('$RD/plan.json'))['bases']['$b']['sets'];print(' '.join(s+':'+x['render']+':'+x['gt'] for s,x in e.items() if s!='test_ouhdl'))")
  TU=$(python3 -c "import json;x=json.load(open('$RD/plan.json'))['bases']['$b']['sets']['test_ouhdl'];print('test_ouhdl:'+x['render']+':'+x['gt'])")
  k=${b%%_*}
  race ${k}u_id $b 08:00:00 bb $LAB $CK $RD/preds/$b identity $TU
  race ${k}u_mir $b 08:00:00 bb $LAB $CK $RD/preds/$b mirror_x $TU
  race ${k}6 $b 12:00:00 bb $LAB $CK $RD/preds/$b mirror_x $T6
done
RS=""
for b in LUO_ms6 M5_ms9 NOFM_W_bb PEINIT_W_bb; do
  ap=$($SB -J r1d_apply_$b -p cpuidle --requeue -c 4 --mem=32G -t 08:00:00 -d afterany${DEP[$b]} -o $RD/logs/%x_%j.out $RC/py.sh r1d_infer.py --plan $RD/plan.json --base $b)
  sc=$($SB -J r1d_score_$b -p cpuidle --requeue -c 16 --mem=48G -t 10:00:00 --array=1-7 -d afterok:$ap --kill-on-invalid-dep=yes -o $RD/logs/%x_%A_%a.out $RC/score.sh $b)
  rd=$($SB -J r1d_read_$b -p cpuidle --requeue -c 4 --mem=24G -t 03:00:00 -d afterok:$sc --kill-on-invalid-dep=yes -o $RD/logs/%x_%j.out $RC/py.sh r1d_read.py $RD/plan.json $b)
  RS="$RS:$rd"
  r1d_log "SUBMIT R1d $b apply=$ap (afterany${DEP[$b]}) score_array=$sc (afterok:$ap) read=$rd (afterok:$sc) lane=clusterA:cpuidle"
  echo "R1d,apply_$b,clusterA:cpuidle,$ap,afterany${DEP[$b]},$RD/apply/$b" >> $CSV; echo "R1d,score_$b,clusterA:cpuidle,$sc,afterok:$ap,$RD/score_out/rows" >> $CSV
  echo "R1d,read_$b,clusterA:cpuidle,$rd,afterok:$sc,$RD/read/$b" >> $CSV
done
sm=$($SB -J r1d_summary -p cpuidle --requeue -c 2 --mem=8G -t 01:00:00 -d afterany$RS -o $RD/logs/%x_%j.out $RC/py.sh r1d_summary.py)
r1d_log "SUBMIT R1d summary job=$sm (afterany$RS) lane=clusterA:cpuidle"; echo "R1d,summary,clusterA:cpuidle,$sm,afterany$RS,$RD/read" >> $CSV
echo SUBMIT_DONE
