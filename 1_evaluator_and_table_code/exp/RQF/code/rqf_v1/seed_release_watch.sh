#!/bin/bash
# RQF seed-release watcher (repair <agent>, 12 Nov 2026). <author>'s rule: seed 1 first; seed-2/3 held job is released once seed-1 FT of the same arm/track is COMPLETED
# (sacct state, or for sm_* arms won on the VPS: runs/<TAG>_vps/RELAYED + SAMPLING_DONE). nice=500 before release. Polls 30 min until 2026-11-15T23:59Z. ONEPASS=1 = single pass.
# Only touches held jobs named sm_SM[AR]_*_s2026103[56], rqf_LUOC40_*_s2026103[56]*, a24en_*/a24zh_*/r2d_* ending _35/_36. fx_* (FMD) never touched. Never touches 25055037.
set -u
R=$U/exp/RQF; LOG=$R/queue/rqf_log.md; END=$(date -d 2026-11-15T23:59:00Z +%s); SKIP=" 25055037 "
L() { echo "$(date -u +%FT%TZ) [repair <agent>] $*" >> $LOG; }
s1name() { local n=$1 b
  case $n in
    sm_SM[AR]_*_s2026103[56]) echo "${n%_s2026103?}_s20261034";;
    rqf_LUOC40_*_s2026103[56]*) echo "${n/_s2026103[56]/_s20261034}";;
    a24en_*_3[56]|a24zh_*_3[56]|r2d_*_3[56]) b=${n%_3[56]}; echo "${b/_samp/}_34";;
  esac; }
while :; do
  declare -A ST=(); while IFS='|' read -r id n s; do ST[$n]=${s%% *}; done < <(sacct -u <user> -X -n -P -S 2026-11-07 -o JobID,JobName%80,State 2>/dev/null | grep -v '\.' | sort -t'|' -k1,1n)
  while IFS='|' read -r id n; do
    case "$SKIP" in *" $id "*) continue;; esac
    s1=$(s1name "$n"); [ -z "$s1" ] && continue
    ok=0; [ "${ST[$s1]:-}" = COMPLETED ] && ok=1
    case $n in sm_*) t=${s1#sm_}; [ -f $R/runs/${t}_vps/RELAYED ] && [ -f $R/runs/${t}_vps/SAMPLING_DONE ] && ok=2;; esac
    [ $ok -eq 0 ] && continue
    nb=$(scontrol show job $id 2>/dev/null | grep -o 'Nice=[0-9-]*')
    if scontrol update jobid=$id nice=500 && scontrol release $id; then
      L "RELEASE job=$id $n: seed-1 $s1 FT done (${ST[$s1]:-none}$([ $ok -eq 2 ] && echo ", VPS relayed SAMPLING_DONE")); $nb -> $(scontrol show job $id 2>/dev/null | grep -o 'Nice=[0-9-]*') then released"
    else L "RELEASE_FAILED job=$id $n"; fi
  done < <(squeue -u <user> -h -t PD -o '%i|%j|%r' | grep '|JobHeldUser' | cut -d'|' -f1,2)
  [ "${ONEPASS:-0}" = 1 ] && break
  [ $(date +%s) -ge $END ] && break
  sleep 1800
done
L "seed-release watcher end"
