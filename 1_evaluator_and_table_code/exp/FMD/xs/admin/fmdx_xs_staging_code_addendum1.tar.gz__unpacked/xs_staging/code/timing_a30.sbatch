#!/bin/bash
#SBATCH -J xs_time_a30
#SBATCH -p gpuidle
#SBATCH --gres=gpu:a30:1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH --requeue
#SBATCH --exclude=dgx[001-005],gpu[004-005,013-014,017-023,119-148]
#SBATCH -o $U/exp/FMD/xs/logs/timing_a30_%j.out
R=$U
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
apptainer exec --nv --bind $R $R/images/torch27.sif $R/exp/FMD/venv/bin/python $R/exp/FMD/xs/code/xs_probe.py time a30 dinov3-vitl16 sd21-dift-gate cleandift-sd21 cleandift-sd15 c-radiov3-b c-radiov3-l c-radiov3-h c-radiov3-g c-radiov4-h c-radiov4-so400m pe-spatial-t16-512 pe-spatial-s16-512 pe-spatial-b16-512 pe-spatial-l14-448 pe-spatial-g14-448 pe-core-b16-224 siglip2-giant-opt-patch16-256 siglip2-so400m-patch14-224 dinov2-large dinov3-vits16 dinov3-vitb16 dinov3-vith16plus dinov3-convnext-large dinov3-vit7b16 tipsv2-l14 tipsv2-so400m14 tipsv2-g14 qwen2-vl-2b-instruct qaari-urdu-vision starvector-1b-im2svg deepseek-ocr dinov2-reg-large webssl-dino1b-224 aimv2-large-patch14-224 vit-mae-large depth-anything-v2-large trocr-large-handwritten
