#!/bin/bash
#SBATCH -J xs_git
#SBATCH -p cpuidle
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o $U/exp/FMD/xs/logs/git_%j.out
cd $U/exp/FMD/xs/weights && mkdir -p github && cd github
for r in CompVis/cleandift NVlabs/RADIO 0xnomy/urdu-deepseek-ocr facebookresearch/perception_models; do n=$(basename $r); [ -d $n ] || git clone -q https://github.com/$r.git $n 2>&1 | tail -2; echo $r $(git -C $n rev-parse HEAD) $(git -C $n log -1 --format=%cI); done
