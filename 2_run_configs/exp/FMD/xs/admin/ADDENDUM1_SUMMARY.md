# FMDX staging - addendum 1 summary (facts)

Source: weights/MANIFEST_v2.json (created 2026-11-03T16:40Z). Token map = H x W at 224-px input unless the input_px column says otherwise; load-test conventions only (not prereg extractors). units/s = A4000, load-test forward, 4 val1000 renders batch-timed (xs_probe.py time).

| local dir | repo id | revision | licence | GB | load status | token map(s) | input px | units/s A4000 |
|---|---|---|---|---|---|---|---|---|
| aimv2-1b-patch14-224 | apple/aimv2-1B-patch14-224 | `1f99ee5baf` | apple-amlr | 9.88 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |
| aimv2-3b-patch14-224 | apple/aimv2-3B-patch14-224 | `0c4b6633c3` | apple-amlr | 21.77 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |
| aimv2-huge-patch14-224 | apple/aimv2-huge-patch14-224 | `172051d9df` | apple-amlr | 5.45 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |
| aimv2-large-patch14-224-distilled | apple/aimv2-large-patch14-224-distilled | `064e34411d` | apple-amlr | 2.47 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |
| arabic-english-hw-ocr-v3 | sherif1313/Arabic-English-handwritten-OCR-v3 | `a404b5ba97` | apache-2.0 | 7.53 | ok | {"pre_merge": [16, 16], "last": [8, 8]} | 224 | - |
| arabic-hw-ocr-4bit | sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v2 | `7ff7c98866` | apache-2.0 | 2.42 | error | - | 224 | - |
| arabic-ocr-deepseek-ocr-2 | loay/Arabic-OCR-DeepSeek-OCR-2 | `3b27977413` | apache-2.0 | 6.79 | error | - | 224 | - |
| chandra-ocr-2 | datalab-to/chandra-ocr-2 | `af93b47dba` | openrail | 10.61 | ok | {"pre_merge": [14, 14], "last": [7, 7]} | 224 | a4000=154.358 |
| chandra | datalab-to/chandra | `bd40c21576` | openrail | 17.55 | ok | {"pre_merge": [14, 14], "last": [7, 7]} | 224 | - |
| depth-anything-3-base | depth-anything/DA3-BASE | `f4a6c9b3c9` | apache-2.0 | 0.54 | error | - | 224 | - |
| depth-anything-3-giant-1.1 | depth-anything/DA3-GIANT-1.1 | `72ee9f89ce` | cc-by-nc-4.0 | 5.42 | error | - | 224 | - |
| dinov2-base | facebook/dinov2-base | `f9e44c814b` | apache-2.0 | 0.35 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |
| dinov2-giant | facebook/dinov2-giant | `611a9d42f2` | apache-2.0 | 4.55 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |
| dinov2-reg-base | facebook/dinov2-with-registers-base | `a1d738ccfa` | apache-2.0 | 0.35 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |
| dinov2-reg-giant | facebook/dinov2-with-registers-giant | `8d0d49f77f` | apache-2.0 | 4.55 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |
| dinov2-reg-small | facebook/dinov2-with-registers-small | `0d9846e56b` | apache-2.0 | 0.09 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |
| dinov3-vit7b16-sat493m | facebook/dinov3-vit7b16-pretrain-sat493m | `915e1b5391` | other | 26.86 | ok | {"last": [14, 14], "mid": [14, 14]} | 224 | - |
| dinov3-vitl16-sat493m | facebook/dinov3-vitl16-pretrain-sat493m | `f692fa42da` | other | 1.21 | ok | {"last": [14, 14], "mid": [14, 14]} | 224 | - |
| dots-ocr-base | dots-studio/dots.ocr.base | `ca7fe48210` | mit | 6.09 | loaded_no_extractor | - | 224 | - |
| dots-ocr | dots-studio/dots.ocr | `c0111ce6bc` | mit | 6.09 | loaded_no_extractor | - | 224 | - |
| florence-2-base | microsoft/Florence-2-base | `5ca5edf5bd` | mit | 0.47 | ok | {"last": [7, 7]} | 224 | - |
| florence-2-large-ft | microsoft/Florence-2-large-ft | `4a12a2b54b` | mit | 1.54 | ok | {"last": [7, 7]} | 224 | - |
| flux1-dev | black-forest-labs/FLUX.1-dev | `3de623fc3c` | other | 33.75 | ok | {"mid": [32, 32], "last_double": [32, 32]} | 512 | - |
| flux1-schnell | black-forest-labs/FLUX.1-schnell | `741f7c3ce8` | apache-2.0 | 33.73 | ok | {"mid": [32, 32], "last_double": [32, 32]} | 512 | - |
| lightonocr-3-0.8b | lightonai/LightOnOCR-3-0.8B | `36d8636925` | apache-2.0 | 1.74 | ok | {"pre_merge": [14, 14], "last": [7, 7]} | 224 | a4000=284.48 |
| lightonocr-3-4b | lightonai/LightOnOCR-3-4B | `b71010f095` | apache-2.0 | 9.11 | ok | {"pre_merge": [14, 14], "last": [7, 7]} | 224 | a4000=154.536 |
| paddleocr-vl-1.0 | PaddlePaddle/PaddleOCR-VL | `7fa00a8c55` | apache-2.0 | 1.93 | loaded_no_extractor | - | 224 | - |
| paddleocr-vl-1.5 | PaddlePaddle/PaddleOCR-VL-1.5 | `2a4195faa5` | apache-2.0 | 1.93 | loaded_no_extractor | - | 224 | - |
| pe-core-g14-448-timm | facebook/vit_pe_core_gigantic_patch14_448_timm | `f537e43432` | apache-2.0 | 7.53 | ok | {"last": [16, 16]} | 224 | - |
| pe-core-g14-448 | facebook/PE-Core-G14-448 | `a604668008` | apache-2.0 | 9.68 | ok | {"last": [16, 16]} | 224 | - |
| pe-core-s16-384 | facebook/PE-Core-S16-384 | `aabf3b9905` | apache-2.0 | 0.35 | no load test | - | 224 | - |
| pe-core-t16-384 | facebook/PE-Core-T16-384 | `8b4722741a` | apache-2.0 | 0.28 | no load test | - | 224 | - |
| pe-lang-g14-448-timm | facebook/vit_pe_lang_gigantic_patch14_448_timm | `2ba6d57d72` | apache-2.0 | 6.96 | ok | {"last": [16, 16]} | 224 | - |
| pe-lang-l14-448-timm | facebook/vit_pe_lang_large_patch14_448_timm | `7eb04e215d` | apache-2.0 | 1.17 | ok | {"last": [16, 16]} | 224 | - |
| qari-ocr-0.4.0-vl-4b | NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct | `aa65b4a463` | apache-2.0 | 0.33 | ok | {"pre_merge": [14, 14], "last": [7, 7]} | 224 | - |
| qari-ocr-v0.3-vl-2b | NAMAA-Space/Qari-OCR-v0.3-VL-2B-Instruct | `dabe11e399` | apache-2.0 | 4.43 | ok | {"pre_merge": [16, 16], "last": [8, 8]} | 224 | - |
| qwen2.5-vl-3b-instruct | Qwen/Qwen2.5-VL-3B-Instruct | `66285546d2` |  | 7.52 | ok | {"pre_merge": [16, 16], "last": [8, 8]} | 224 | - |
| qwen3-vl-4b-instruct | Qwen/Qwen3-VL-4B-Instruct | `ebb281ec70` | apache-2.0 | 8.89 | ok | {"pre_merge": [14, 14], "last": [7, 7]} | 224 | - |
| sam3.1 | facebook/sam3.1 | `daa6319184` | other | 3.51 | error | - | 224 | - |
| sam3 | facebook/sam3 | `3c879f3982` | other | 6.90 | ok | {"last": [72, 72]} | 1008 | - |
| sd3-medium-diffusers | stabilityai/stable-diffusion-3-medium-diffusers | `ea42f8cef0` | other | 21.06 | ok | {"mid": [32, 32], "last": [32, 32]} | 512 | - |
| siglip-large-patch16-384-cfg | google/siglip-large-patch16-384 | `ce005573a4` | apache-2.0 | 0.00 | no load test | - | 224 | - |
| siglip2-large-patch16-256 | google/siglip2-large-patch16-256 | `787800c899` | apache-2.0 | 3.56 | ok | {"last": [14, 14], "mid": [14, 14]} | 224 | - |
| siglip2-so400m-patch16-256 | google/siglip2-so400m-patch16-256 | `e8708ab72d` | apache-2.0 | 4.58 | ok | {"last": [14, 14], "mid": [14, 14]} | 224 | - |
| starvector-8b-im2svg | starvector/starvector-8b-im2svg | `518beea8dc` | apache-2.0 | 15.02 | ok | {"last": [14, 14]} | 224 | a4000=63.434 |
| surya-ocr-2-persian-hw | Reza2kn/surya-ocr-2-persian-handwriting-369k | `992238bfee` | openrail | 1.33 | ok | {"pre_merge": [14, 14], "last": [7, 7]} | 224 | - |
| theia-base-cddsv | theaiinstitute/theia-base-patch16-224-cddsv | `f251f815f6` | other | 0.75 | ok | {"last": [14, 14]} | 224 | - |
| theia-base-cdiv | theaiinstitute/theia-base-patch16-224-cdiv | `90eaf9cf0b` | other | 0.56 | ok | {"last": [14, 14]} | 224 | - |
| theia-small-cdiv | theaiinstitute/theia-small-patch16-224-cdiv | `b07fb29f3f` | other | 0.15 | ok | {"last": [14, 14]} | 224 | - |
| theia-tiny-cddsv | theaiinstitute/theia-tiny-patch16-224-cddsv | `8ce99426c2` | other | 0.06 | ok | {"last": [14, 14]} | 224 | - |
| theia-tiny-cdiv | theaiinstitute/theia-tiny-patch16-224-cdiv | `996d2eac80` | other | 0.04 | ok | {"last": [14, 14]} | 224 | - |
| tipsv1-b14 | google/tipsv1-b14 | `424f5e0ee5` | apache-2.0 | 0.78 | ok | {"last": [16, 16]} | 224 | a4000=127.736 |
| tipsv1-g14 | google/tipsv1-g14 | `79a0b1548c` | apache-2.0 | 6.10 | ok | {"last": [16, 16]} | 224 | - |
| tipsv1-l14 | google/tipsv1-l14 | `05c1dd400a` | apache-2.0 | 1.95 | ok | {"last": [16, 16]} | 224 | a4000=40.827 |
| tipsv1-s14 | google/tipsv1-s14 | `109d6693a2` | apache-2.0 | 0.22 | ok | {"last": [16, 16]} | 224 | a4000=436.826 |
| tipsv1-so400m14 | google/tipsv1-so400m14 | `9bf0d1426d` | apache-2.0 | 3.45 | ok | {"last": [16, 16]} | 224 | a4000=28.506 |
| trocr-base-handwritten | microsoft/trocr-base-handwritten | `eaacaf452b` | mit | 1.33 | ok | {"last": [24, 24]} | 384 | a4000=61.912 |
| trocr-base-hw-hist-swe-2 | Riksarkivet/trocr-base-handwritten-hist-swe-2 | `aa79fcb185` | apache-2.0 | 1.54 | ok | {"last": [24, 24]} | 384 | a4000=55.569 |
| trocr-small-handwritten | microsoft/trocr-small-handwritten | `b4648cfa17` |  | 0.25 | ok | {"last": [24, 24]} | 384 | a4000=198.903 |
| webssl-dino2b-224 | facebook/webssl-dino2b-full2b-224 | `cd5893e3fd` | cc-by-nc-4.0 | 8.34 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | a4000=6.438 |
| webssl-dino300m-224 | facebook/webssl-dino300m-full2b-224 | `8529cdb3fb` | cc-by-nc-4.0 | 1.21 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | a4000=40.495 |
| webssl-dino3b-224 | facebook/webssl-dino3b-full2b-224 | `2d015c340b` | cc-by-nc-4.0 | 11.79 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |
| webssl-dino5b-224 | facebook/webssl-dino5b-full2b-224 | `88006b18b9` | cc-by-nc-4.0 | 19.75 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |
| webssl-dino7b-224 | facebook/webssl-dino7b-full8b-224 | `c6085463ea` | cc-by-nc-4.0 | 25.80 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |
| webssl-mae1b-224 | facebook/webssl-mae1b-full2b-224 | `5880aefedb` | cc-by-nc-4.0 | 4.55 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |
| webssl-mae3b-224 | facebook/webssl-mae3b-full2b-224 | `a73c7f2812` | cc-by-nc-4.0 | 11.83 | ok | {"last": [16, 16], "mid": [16, 16]} | 224 | - |

66 models staged in addendum 1.

## Blocked (addendum 1)

- sd3.5-large-diffusers-layout (stabilityai/stable-diffusion-3.5-large): GatedRepoError: 403 Client Error. (Request ID: Root=1-6aea0350-7bb4b2fe27d491541b5c0c9d;03bb760f-2c09-45f3-9e41-f421b94d247d)

Cannot access gated repo for url https://huggingface.co/stabilityai/stable-diffusion-3.5-large/resolve/ceddf0a7fdf2064ea28e2213e3b84e4afa170a0f/scheduler/scheduler_config.json.
Access to model stabilityai/stable-diffusion-3.5-large is restricted and you are not in the authorized list. Vis
- sd3.5-large-turbo-diffusers-layout (stabilityai/stable-diffusion-3.5-large-turbo): GatedRepoError: 403 Client Error. (Request ID: Root=1-6aea0356-6b76ed895aa7ebd645286553;79649ccf-d2c3-4c80-a622-fe0010b72b16)

Cannot access gated repo for url https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo/resolve/ec07796fc06b096cc56de9762974a28f4c632eda/scheduler/scheduler_config.json.
Access to model stabilityai/stable-diffusion-3.5-large-turbo is restricted and you are not in the authoriz
- sd3.5-medium-diffusers-layout (stabilityai/stable-diffusion-3.5-medium): GatedRepoError: 403 Client Error. (Request ID: Root=1-6aea0348-313da3527e047f193531b6e1;47a50d21-4aee-41bf-ab53-1d36de6c5019)

Cannot access gated repo for url https://huggingface.co/stabilityai/stable-diffusion-3.5-medium/resolve/b940f670f0eda2d07fbb75229e779da1ad11eb80/text_encoder_2/model.fp16.safetensors.
Access to model stabilityai/stable-diffusion-3.5-medium is restricted and you are not in the authorized l