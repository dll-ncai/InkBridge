#!/bin/bash
#SBATCH -J xs_git2
#SBATCH -p cpuidle
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 02:00:00
#SBATCH -o $U/exp/FMD/xs/logs/git2_%j.out
cd $U/exp/FMD/xs/weights/github
for r in joanrod/star-vector; do n=$(basename $r); [ -d $n ] || git clone -q --depth 50 https://github.com/$r.git $n 2>&1 | tail -2; echo $r $(git -C $n rev-parse HEAD) $(git -C $n log -1 --format=%cI); done
