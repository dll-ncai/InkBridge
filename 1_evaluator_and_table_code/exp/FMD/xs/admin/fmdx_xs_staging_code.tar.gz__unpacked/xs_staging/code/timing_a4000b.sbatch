#!/bin/bash
#SBATCH -J xs_time_a4000b
#SBATCH -p gpuidle
#SBATCH --gres=gpu:a4000:1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx[001-005],gpu[004-005,013-014,017-023,119-148]
#SBATCH -o $U/exp/FMD/xs/logs/timing_a4000_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
apptainer exec --nv --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py time a4000 c-radiov3-b c-radiov3-l c-radiov3-h c-radiov3-g c-radiov4-h c-radiov4-so400m pe-spatial-t16-512 pe-spatial-s16-512 pe-spatial-b16-512 pe-spatial-l14-448 pe-spatial-g14-448 pe-core-b16-224 siglip2-giant-opt-patch16-256 siglip2-so400m-patch14-224 dinov2-large dinov3-vits16 dinov3-vitb16 dinov3-vith16plus dinov3-convnext-large
