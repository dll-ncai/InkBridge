#!/bin/bash
# RQF seed-priority watcher (<author> 10 Nov): jobs in HELD (seed 2/3 GPU jobs of this track) are released once no seed-1 (s20261034) GPU job of this track is PENDING without a dependency (i.e. all have started).
# Stops itself 14 Nov 17:30Z. Logs every release in rqf_log.md. Usage: hold_release.sh "<jobids held>"
H="$1"; L=$U/exp/RQF/queue/rqf_log.md; END=$(date -u -d '2026-11-14 17:30:00' +%s)
while [ $(date +%s) -lt $END ]; do
  LEFT=""; for j in $H; do squeue -h -j $j -o '%T' 2>/dev/null | grep -q PENDING && LEFT="$LEFT $j"; done
  [ -z "$LEFT" ] && { echo "$(date -u +%FT%TZ) [rqf-s2 <agent> hold-watch] all held jobs started/ended; watcher end" >> $L; exit 0; }
  P1=$(squeue -h -u <user> -t PD -o '%i|%j|%r' | grep 's20261034' | grep -v rqfpost | grep -v 'Dependency' | grep -v rqf_tables | cut -d'|' -f1 | tr '\n' ' ')
  if [ -z "$P1" ]; then
    for j in $LEFT; do scontrol release $j && echo "$(date -u +%FT%TZ) [rqf-s2 <agent> hold-watch] RELEASE job=$j (no seed-1 GPU job of this track is pending any more)" >> $L; done
    exit 0
  fi
  sleep 600
done
echo "$(date -u +%FT%TZ) [rqf-s2 <agent> hold-watch] watcher end 14 Nov 17:30Z; still held: $H (release manually: scontrol release <id>)" >> $L
