#!/bin/bash
#SBATCH -J xs_clusterB_setup
#SBATCH -p batch
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH -o $U_B/exp/FMD/xs/logs/clusterB_setup_%j.out
D=$U_B; IMG=$D/images/pytorch_2.7.1-cuda12.8-cudnn9-runtime.sqsh
CT="srun --ntasks=1 --container-image=$IMG --container-mounts=$D:$U,/tmp:/tmp --container-workdir=$U"
P=$U/exp/FMD/xs/pylibs_clusterB
$CT python -c "import sys,torch;print(sys.version.split()[0],torch.__version__)"
$CT python -m pip install --no-deps --target $P transformers==5.22.1 tokenizers==0.23.3 safetensors==0.8.0 huggingface_hub==2.4.0 hf-xet==1.8.0 httpx2==2.16.0 httpcore2==2.16.0 h11==0.16.0 anyio==4.16.0 click==8.5.0 typer==0.27.3 shellingham==1.5.4 rich==15.0.0 markdown-it-py==4.2.0 mdurl==0.1.2 regex==2026.9.29 diffusers==0.42.1 accelerate==1.12.0 sentencepiece==0.2.2 brotli==1.2.1 idna==3.20 dnspython==2.7.0 importlib_metadata==10.1.0 annotated-doc==0.0.5 fsspec==2025.5.1 typing_extensions==4.16.0 timm==1.0.30 einops==0.8.2 open_clip_torch==3.3.0 ftfy==6.3.1 2>&1 | tail -3
$CT python -m pip freeze --path $P > $D/exp/FMD/xs/pylibs_clusterB_freeze.txt
$CT python $U/exp/FMD/xs/code/stage_clusterB.py
