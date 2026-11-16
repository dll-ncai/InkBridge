#!/bin/bash
# Parallel ranged download of OUHD-L v1.0 core tar from Zenodo (single stream ~0.2 MB/s).
cd "$(dirname "$0")"; mkdir -p parts
N=$(( (8610631680 + 67108864 - 1) / 67108864 ))
seq 0 $((N-1)) | xargs -P 32 -n 1 ./fetch_one.sh
