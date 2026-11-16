
import os, json
os.environ["HF_HOME"] = "$U/exp/FMD/xs/weights/.hf_home"
from huggingface_hub import snapshot_download
for r in ["facebook/deit-tiny-patch16-224", "facebook/deit-small-patch16-224", "facebook/deit-base-patch16-224"]:
    p = snapshot_download(r, allow_patterns=["config.json", "preprocessor_config.json", "*.safetensors", "pytorch_model.bin"]); print(r, p)
