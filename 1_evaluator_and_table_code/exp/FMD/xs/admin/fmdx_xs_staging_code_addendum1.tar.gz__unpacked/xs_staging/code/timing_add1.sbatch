#!/bin/bash
#SBATCH -J xs_time_add1
#SBATCH -p gpuidle
#SBATCH --gres=gpu:a4000:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx[001-005],gpu[004-005,013-014,017-023,119-148]
#SBATCH -o $U/exp/FMD/xs/logs/timing_add1_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
apptainer exec --nv --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py time a4000 starvector-8b-im2svg chandra-ocr-2 lightonocr-3-4b lightonocr-3-0.8b trocr-base-handwritten trocr-small-handwritten tipsv1-s14 tipsv1-b14 tipsv1-l14 tipsv1-so400m14 webssl-dino300m-224 webssl-dino2b-224 trocr-base-hw-hist-swe-2
