# FMDX staging - addendum summary (addendum2) (facts)

Source: weights/MANIFEST_v2.json (created 2026-11-03T20:58Z). Token map = H x W at 224-px input unless the input_px column says otherwise; load-test conventions only (not prereg extractors). units/s = A4000, load-test forward, 4 val1000 renders batch-timed (xs_probe.py time).

| local dir | repo id | revision | licence | GB | load status | token map(s) | input px | units/s A4000 |
|---|---|---|---|---|---|---|---|---|
| sd3.5-large-diffusers-layout | stabilityai/stable-diffusion-3.5-large | `ceddf0a7fd` | other | 27.63 | ok | {"s0.261_block19": [64, 64], "s0.261_block28": [64, 64], "s0.101_block19": [64, 64], "s0.101_block28": [64, 64]} | 1024 | - |
| sd3.5-large-turbo-diffusers-layout | stabilityai/stable-diffusion-3.5-large-turbo | `ec07796fc0` | other | 27.63 | ok | {"s0.261_block19": [64, 64], "s0.261_block28": [64, 64], "s0.101_block19": [64, 64], "s0.101_block28": [64, 64]} | 1024 | - |
| sd3.5-medium-diffusers-layout | stabilityai/stable-diffusion-3.5-medium | `b940f670f0` | other | 16.27 | ok | {"s0.261_block12": [64, 64], "s0.261_block18": [64, 64], "s0.101_block12": [64, 64], "s0.101_block18": [64, 64]} | 1024 | - |
| starcoder2-7b | bigcode/starcoder2-7b | `bb9afde76d` | bigcode-openrail-m | 14.35 | loaded_no_extractor | - | 224 | - |

4 models staged in addendum 1.

## Blocked (addendum 1)

- starcoderbase-1b (bigcode/starcoderbase-1b): GatedRepoError: 403 Client Error. (Request ID: Root=1-6aea37ba-30954824216cafa909332c90;602eb82c-87b1-42d7-bc50-69903c500711)

Cannot access gated repo for url https://huggingface.co/bigcode/starcoderbase-1b/resolve/182f0165fdf8da9c9935901eec65c94337f01c11/model.safetensors.
Access to model bigcode/starcoderbase-1b is restricted and you are not in the authorized list. Visit https://huggingface.co/bigcode/starcode