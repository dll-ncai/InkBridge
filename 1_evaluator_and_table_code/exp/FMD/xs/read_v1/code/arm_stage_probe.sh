#!/bin/bash
# Prints one CSV line per arm: arm,clusterA_dir,last_step,training_complete,best_milestone,ft_complete,ft_best,slurm_state
X=$U/exp/FMD/xs; R=$X/runs
echo "arm,last_step,training_complete,best_milestone,gate_fail,ft_last_step,ft_complete,ft_best,clusterA_job,slurm_state"
tail -n +2 $X/admin/arm_queue.csv | tr -d '\r' | while IFS=, read arm regime fm state rj dj vc rest; do
  case "$state" in CANCELLED*) continue;; esac
  D=$R/${arm}_clusterA; F=$R/fmdx_monitor/ft/${arm}_FT_clusterA
  ls=$(tail -n 1 $D/train_log.jsonl 2>/dev/null | grep -o '"step": *[0-9]*' | grep -o '[0-9]*$')
  tc=$([ -f $D/training_complete.json ] && echo 1 || echo 0)
  bm=$(cat $D/eval/best_milestone.txt 2>/dev/null | head -c 20 | tr -d '\n,')
  gf=$([ -f $D/GATE_FAIL ] && echo 1 || echo 0)
  fs=$(tail -n 1 $F/train_log.jsonl 2>/dev/null | grep -o '"step": *[0-9]*' | grep -o '[0-9]*$')
  fc=$([ -f $F/training_complete.json ] && echo 1 || echo 0)
  fb=$(cat $F/eval/best_milestone.txt 2>/dev/null | head -c 20 | tr -d '\n,')
  ss=""; [ -n "$rj" ] && ss=$(sacct -n -X -j ${rj%.*} -o State 2>/dev/null | head -1 | tr -d ' ')
  echo "$arm,$ls,$tc,$bm,$gf,$fs,$fc,$fb,${rj%.*},$ss"
done
