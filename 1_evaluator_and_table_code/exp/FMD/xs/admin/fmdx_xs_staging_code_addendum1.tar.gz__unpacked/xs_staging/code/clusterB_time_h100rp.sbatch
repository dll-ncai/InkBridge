#!/bin/bash
#SBATCH -J xs_clusterB_time_h100rp
#SBATCH -p H100-RP
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --requeue
#SBATCH -o $U_B/exp/FMD/xs/logs/clusterB_time_h100rp_%j.out
D=$U_B; IMG=$D/images/pytorch_2.7.1-cuda12.8-cudnn9-runtime.sqsh
CT="srun --ntasks=1 --container-image=$IMG --container-mounts=$D:$U,/tmp:/tmp --container-workdir=$U"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
$CT env PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=$U/exp/FMD/xs/pylibs_clusterB python $U/exp/FMD/xs/code/xs_probe.py time h100rp c-radiov3-b c-radiov3-l c-radiov3-g c-radiov4-h pe-spatial-l14-448 pe-spatial-g14-448 siglip2-giant-opt-patch16-256 cleandift-sd21 sd21-dift-gate
