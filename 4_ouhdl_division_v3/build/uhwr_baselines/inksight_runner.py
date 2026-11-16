"""InkSight zero-shot inference over a manifest of crop images (run in the 'inksight' env).

Follows google-research/inksight colab.ipynb word-level inference exactly:
  model = from_pretrained_keras("Derendering/InkSight-Small-p"); cf = signatures['serving_default']
  image = scale_and_pad(crop)  (224 x 224, aspect kept, black padding, as in the notebook)
  prompt "Recognize and derender." ; if the decoded ink is empty -> fallback "Derender the ink."
  output tokens -> detokenize() -> strokes in 224 space -> mapped back to crop pixels
  with (p - (dx, dy)) / ratio.
Usage: python inksight_runner.py manifest.json out.jsonl [prompt_mode]
  manifest.json: list of {"id":..., "png": path}
  prompt_mode: "default" (notebook: recognize+derender with fallback) or "vanilla"
"""
import json
import os
import re
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.environ.get("INKSIGHT_REPO", "ext/inksight"))
from utils.tensorflow import tf  # noqa: E402  (repo's TF import with XLA flags)
import tensorflow_text  # noqa: E402,F401
from huggingface_hub import from_pretrained_keras  # noqa: E402


def text_to_tokens(text):
    return [int(t) for t in re.findall(r"<ink_token_(\d+)>", text)]


def detokenize(tokens):
    coordinate_length = 224
    n = coordinate_length + 1
    vocabulary_size = n * 2 + 1
    start_token = n * 2
    if any(t < 0 or t >= vocabulary_size for t in tokens):
        raise ValueError("bad ink token")
    idx, res, cur = 0, [], []
    while idx < len(tokens):
        t = tokens[idx]
        if t == start_token:
            if cur:
                res.append(cur)
            cur = []
            idx += 1
        else:
            if idx + 1 < len(tokens) and tokens[idx + 1] != start_token:
                x, y = tokens[idx], tokens[idx + 1] - n
                if 0 <= x <= coordinate_length and 0 <= y <= coordinate_length:
                    cur.append([x, y])
                idx += 2
            else:
                idx += 1
    if cur:
        res.append(cur)
    return res


def scale_and_pad(original, pad_black=True):
    ratio = min(224 / original.width, 224 / original.height)
    original_np = np.array(original)
    new_crop = original.resize((int(original.width * ratio), int(original.height * ratio)))
    p = [original_np[1, 1], original_np[1, original_np.shape[-1] - 1],
         original_np[original_np.shape[0] - 1, 1], original_np[original_np.shape[0] - 1, original_np.shape[-1] - 1]]
    avg = np.rint(np.mean(p, axis=0)).astype(np.uint8)
    color = tuple(avg) if not pad_black else (0, 0, 0)
    new_image = Image.new(new_crop.mode, (224, 224), color)
    dx = (224 - new_crop.width) // 2
    dy = (224 - new_crop.height) // 2
    new_image.paste(new_crop, (dx, dy))
    return new_image, ratio, dx, dy


def main():
    manifest, out_path = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "default"
    items = json.load(open(manifest))
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["id"] for l in open(out_path)}
    local = os.environ.get("INKSIGHT_MODEL")  # local copy of the HF repo files (same weights)
    model = tf.saved_model.load(local) if local else from_pretrained_keras("Derendering/InkSight-Small-p")
    cf = model.signatures["serving_default"]
    prompts = ("Recognize and derender.", "Derender the ink.") if mode == "default" else ("Derender the ink.", None)
    with open(out_path, "a") as fo:
        for it in items:
            if it["id"] in done:
                continue
            t0 = time.time()
            crop = Image.open(it["png"]).convert("RGB")
            image, ratio, dx, dy = scale_and_pad(crop)
            enc = tf.reshape(tf.io.encode_jpeg(np.array(image)[:, :, :3]), (1, 1))
            used = prompts[0]
            txt = cf(**{"input_text": tf.constant([used], dtype=tf.string), "image/encoded": enc})["output_0"].numpy()[0][0].decode()
            ink = detokenize(text_to_tokens(txt))
            if not ink and prompts[1]:
                used = prompts[1]
                txt = cf(**{"input_text": tf.constant([used], dtype=tf.string), "image/encoded": enc})["output_0"].numpy()[0][0].decode()
                ink = detokenize(text_to_tokens(txt))
            strokes = [[[(x - dx) / ratio, (y - dy) / ratio] for x, y in s] for s in ink]
            recog = txt.split("<ink_token")[0].strip()
            fo.write(json.dumps(dict(id=it["id"], prompt=used, strokes=strokes, recognized=recog,
                                     secs=time.time() - t0)) + "\n")
            fo.flush()


if __name__ == "__main__":
    main()
