#!/bin/bash
#SBATCH -J xs_pkgs
#SBATCH -p cpuidle
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 01:00:00
#SBATCH -o $U/exp/FMD/xs/logs/pkgs_%j.out
R=$U
PY=$R/exp/FMD/venv/bin/python
apptainer exec --bind $R $R/images/torch27.sif $PY - <<'EOF'
import importlib
for m in ["timm","einops","open_clip","peft","flash_attn","xformers","transformers","diffusers","torchvision","omegaconf","easydict","addict","ftfy","accelerate","cv2","sentencepiece","tiktoken","scipy","pandas","PIL","bitsandbytes","cairosvg","depth_anything_3","qwen_vl_utils","safetensors","yaml","regex","huggingface_hub"]:
    try:
        x=importlib.import_module(m); print("HAVE",m, getattr(x,"__version__","?"), flush=True)
    except Exception as e: print("MISSING",m,type(e).__name__,flush=True)
EOF
mkdir -p $R/exp/FMD/xs/pylibs
apptainer exec --bind $R $R/images/torch27.sif $PY -m pip install --no-deps --target $R/exp/FMD/xs/pylibs "timm>=1.0.20" einops open_clip_torch ftfy easydict addict 2>&1 | tail -5
apptainer exec --bind $R $R/images/torch27.sif $PY -m pip freeze --path $R/exp/FMD/xs/pylibs > $R/exp/FMD/xs/pylibs_freeze.txt; cat $R/exp/FMD/xs/pylibs_freeze.txt
