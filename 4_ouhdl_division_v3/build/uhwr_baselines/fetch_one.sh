#!/bin/bash
# Fetch one 64 MiB range of the OUHD-L core tar, resuming (appending) after dropped connections.
URL=https://zenodo.org/api/records/20642162/files/ouhdl_v1.0_core.tar/content
SIZE=8610631680; CH=67108864
i=$1; s=$((i*CH)); e=$((s+CH-1)); [ $e -ge $SIZE ] && e=$((SIZE-1)); f=$(printf "parts/part_%04d" $i); want=$((e-s+1))
for t in $(seq 1 40); do
  have=$(stat -f%z $f 2>/dev/null || echo 0)
  [ "$have" -eq "$want" ] && exit 0
  [ "$have" -gt "$want" ] && { rm -f $f; have=0; }
  curl -sS --max-time 1800 -r $((s+have))-$e $URL >> $f 2>/dev/null
done; echo FAIL $f
