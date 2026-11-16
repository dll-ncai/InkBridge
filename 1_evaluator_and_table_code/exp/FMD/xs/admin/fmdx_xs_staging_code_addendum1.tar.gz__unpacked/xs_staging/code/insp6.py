
import json, glob, os
W="$U/exp/FMD/xs/weights/"
for n in ["chandra-ocr-2","chandra","dots-ocr","dots-ocr-base","lightonocr-3-4b","lightonocr-3-0.8b","paddleocr-vl-1.5","theia-tiny-cddsv","theia-base-cdiv","starvector-8b-im2svg","surya-ocr-2-persian-hw","qari-ocr-v0.3-vl-2b","qari-ocr-0.4.0-vl-4b","arabic-english-hw-ocr-v3","pe-core-g14-448-timm","pe-lang-l14-448-timm","florence-2-base","webssl-dino300m-224","tipsv1-b14"]:
    p=W+n+"/config.json"
    if not os.path.exists(p): print(n,"(not yet)", sorted(os.listdir(W+n))[:6] if os.path.isdir(W+n) else ""); continue
    c=json.load(open(p)); vc=c.get("vision_config") or {}
    print(n, c.get("model_type"), c.get("architectures"), "auto_map:", list((c.get("auto_map") or {}).items())[:2], "| vision:", {k:vc.get(k) for k in ("model_type","hidden_size","patch_size","image_size","depth","num_hidden_layers","spatial_merge_size","num_heads") if k in vc}, "| files:", [f for f in sorted(os.listdir(W+n)) if f.endswith(".py")][:5])
