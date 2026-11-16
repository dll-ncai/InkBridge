# CAPACITY NOTE - FMDX staging (track XS staging, <agent>)

Built 2026-11-03T14:40Z.  Facts only.  Sources: admin/timing/*.json (CLUSTER-A), admin/timing_clusterB/*.json (CLUSTER-B), admin/loadtest/*.json, admin/live_state_2026-11-03.json, admin/luo_render_probe.json, data/m5_multiscript_v1/DATA_CARD.md.

## 1. Training data of the LUO backbone (units per source)

LUO backbone = `exp/M5/runs/luo` trained on `data/m5_multiscript_v1/m5_luo_train.npz` (sha256 690441b6bd7c119d774066622c53f23a0aca6e62825a04014351bfba98260b67; 80,000 units; Urdu left out; 100k steps; effective batch 512 sequences/step per prereg A6).  Counts read from the npz (`scripts`, `sample_ids`):

| source (script tag) | units | writers (sample_ids) | train pool units | units with step > 1 px (val250 rate) |
|---|---|---|---|---|
| CASIA-OLHWDB zh (casia_zh) | 20,000 | 552 | 2,105,324 | 46.8% (val) |
| CASIA-OLHWDB en (casia_en) | 20,000 | 552 | 28,699 | 0.0% (val) |
| HPL Tamil (hpl_tamil) | 20,000 | 111 | 49,156 | 2.0% (val) |
| IAM-OnDB components (iam) | 20,000 | 382 | 81,659 | 0.0% (val) |
| **total LUO** | **80,000** | | | |

Not in the LUO mix: OUHD-L Urdu (20,000 sampled in the M5 joint set; p01 FT subset = 336 units / 183 writers, p10 = 3,362, p30 = 10,085, p100 = 33,615), German UNIPEN-de and OnHW (zero-shot).  Screen sets rendered at 224 px (this track): casia_zh / casia_en / hpl_tamil / iam val250 and the UNIPEN-de train stand-in (3,927 units, 12 writers); 64-px re-render byte-identical for every unit.  Files: `exp/FMD/xs/data/xs_render224_*.npz` (+ `.meta.json`).

**Training-render fact (LUO units carry no ground-truth ink):** `m5_luo_train.npz` has `seq` (Nagamatsu) and `cond_rendered` (64 px) but no `gt_points`.  224-px renders therefore have to be derived by `e0_adapter.decode(seq)` -> `fmd_harness.render.render224`.  Probe (admin/luo_render_probe.json, 125 random units per script, seed 20261103):

| script | units probed | decoded-seq 64-px render byte-identical to stored cond_rendered | max abs diff | mean abs diff (grey levels) |
|---|---|---|---|---|
| casia_zh | 125 | 0 | 207 | 6.7653 |
| casia_en | 125 | 0 | 243 | 5.0217 |
| hpl_tamil | 125 | 0 | 220 | 3.0015 |
| iam | 125 | 0 | 193 | 3.9704 |

Speed (1 CPU core): decode 0.11 ms, render64 0.66 ms, render224 0.51 ms per unit -> 80,000 units ~ 0.8 CPU-min.  RTL/mirror_x framing of training renders is not included in this figure.

## 2. Per-FM extraction rate (1,000 val1000 renders, strict fp32, TF32 off, forward + copy of returned layers to CPU, weight load excluded)

| FM | token map at 224 px | channels | a4000 u/s | code sha (first run) |
|---|---|---|---|---|
| dinov3-vitl16 | {"last": [14, 14], "mid": [14, 14]} | {"last": 1024, "mid": 1024} | 56.4 | 4b629160 |
| sd21-dift-gate | {"t261_up1": [48, 48]} | {"t261_up1": 1280} | 1.2 | 4b629160 |
| cleandift-sd21 | {"up1": [48, 48], "up2": [96, 96]} | {"up1": 1280, "up2": 640} | 1.7 | 0ccce9a0 |
| cleandift-sd15 | {"up1": [32, 32], "up2": [64, 64]} | {"up1": 1280, "up2": 640} | 4.4 | 0ccce9a0 |
| pe-spatial-b16-512 | {"last": [14, 14]} | {"last": 768} | 190.8 | 0ccce9a0 |
| pe-spatial-l14-448 | {"last": [16, 16]} | {"last": 1024} | 45.4 | 0ccce9a0 |
| pe-spatial-g14-448 | {"last": [16, 16]} | {"last": 1536} | 8.4 | 0ccce9a0 |
| c-radiov3-l | {"last": [14, 14]} | {"last": 1024} | 65.8 | 0ccce9a0 |
| c-radiov3-h | {"last": [14, 14]} | {"last": 1280} | 32.0 | 0ccce9a0 |
| c-radiov3-g | {"last": [14, 14]} | {"last": 1536} | 17.9 | 0ccce9a0 |
| c-radiov4-h | {"last": [14, 14]} | {"last": 1280} | 31.9 | 0ccce9a0 |
| c-radiov4-so400m | {"last": [14, 14]} | {"last": 1152} | 47.0 | 0ccce9a0 |
| siglip2-giant-opt-patch16-256 | {"last": [14, 14], "mid": [14, 14]} | {"last": 1536, "mid": 1536} | 17.4 | 0ccce9a0 |
| dinov3-vith16plus | {"last": [14, 14], "mid": [14, 14]} | {"last": 1280, "mid": 1280} | 20.4 | 0ccce9a0 |
| dinov3-vit7b16 | {"last": [14, 14], "mid": [14, 14]} | {"last": 4096, "mid": 4096} | 18.3 | 0ccce9a0 |
| tipsv2-so400m14 | {"last": [16, 16]} | {"last": 1152} | 30.3 | 0ccce9a0 |
| tipsv2-g14 | {"last": [16, 16]} | {"last": 1536} | 11.6 | 0ccce9a0 |
| dinov3-convnext-large | {"last": [7, 7], "mid": [14, 14]} | {"last": 1536, "mid": 768} | 87.7 | 0ccce9a0 |
| siglip2-so400m-patch14-224 | {"last": [16, 16], "mid": [16, 16]} | {"last": 1152, "mid": 1152} | 31.5 | 0ccce9a0 |
| dinov2-large | {"last": [16, 16], "mid": [16, 16]} | {"last": 1024, "mid": 1024} | 42.1 | 0ccce9a0 |
| aimv2-large-patch14-224 | {"last": [16, 16], "mid": [16, 16]} | {"last": 1024, "mid": 1024} | 47.0 | 0ccce9a0 |
| c-radiov3-b | {"last": [14, 14]} | {"last": 768} | 221.9 | 0ccce9a0 |
| deepseek-ocr | {"sam_last": [4, 4], "clip_last": [4, 4]} | {"sam_last": 1024, "clip_last": 1024} | 493.3 | 0ccce9a0 |
| depth-anything-v2-large | {"last": [16, 16], "mid": [16, 16]} | {"last": 1024, "mid": 1024} | 46.3 | 0ccce9a0 |
| dinov2-reg-large | {"last": [16, 16], "mid": [16, 16]} | {"last": 1024, "mid": 1024} | 43.8 | 0ccce9a0 |
| dinov3-vitb16 | {"last": [14, 14], "mid": [14, 14]} | {"last": 768, "mid": 768} | 181.9 | 0ccce9a0 |
| dinov3-vits16 | {"last": [14, 14], "mid": [14, 14]} | {"last": 384, "mid": 384} | 537.1 | 0ccce9a0 |
| pe-core-b16-224 | {"last": [14, 14]} | {"last": 768} | 186.9 | 0ccce9a0 |
| pe-spatial-s16-512 | {"last": [14, 14]} | {"last": 384} | 598.1 | 0ccce9a0 |
| pe-spatial-t16-512 | {"last": [14, 14]} | {"last": 192} | 1146.9 | 0ccce9a0 |
| qaari-urdu-vision | {"pre_merge": [16, 16], "last": [8, 8]} | {"pre_merge": 1280, "last": 1536} | 77.2 | 0ccce9a0 |
| qwen2-vl-2b-instruct | {"pre_merge": [16, 16], "last": [8, 8]} | {"pre_merge": 1280, "last": 1536} | 76.9 | 0ccce9a0 |
| starvector-1b-im2svg | {"last": [16, 16]} | {"last": 1024} | 46.1 | 0ccce9a0 |
| tipsv2-l14 | {"last": [16, 16]} | {"last": 1024} | 42.8 | 0ccce9a0 |
| trocr-large-handwritten | {"last": [24, 24]} | {"last": 1024} | 20.8 | 0ccce9a0 |
| vit-mae-large | {"last": [14, 14], "mid": [14, 14]} | {"last": 1024, "mid": 1024} | 64.9 | 0ccce9a0 |
| webssl-dino1b-224 | {"last": [16, 16], "mid": [16, 16]} | {"last": 1536, "mid": 1536} | 12.0 | 0ccce9a0 |

GPUs: a4000 = NVIDIA RTX A4000.  `bf16`: dinov3-vit7b16 is loaded in bf16 (27 GB fp32 does not fit a 16/24 GB card); every other row is fp32.  sd21-dift-gate = DIFT gate layer only (768 px, 4 noise draws x 1 timestep, UNet stopped after up_block 1); cleandift-sd21 = one clean-latent pass at t = 0 through the full UNet at 768 px (load-test convention, NOT the official `get_features` path).

## 3. GPU-hours for REPA targets

GPU-h = units / (units/s) / 3600 on one GPU of that type, one pass over the units, extraction only (no probe/training time).  Rows use the measured rate of the FM on that GPU type.

### priority set (control has no FM; DINOv3-L, CleanDIFT-SD2.1, SD2.1-DIFT gate, PE-Spatial-L14 as PE proxy, C-RADIOv4-H)

| FM | a4000: 80k units GPU-h |
|---|---|
| dinov3-vitl16 | 0.39 |
| cleandift-sd21 | 12.80 |
| sd21-dift-gate | 17.99 |
| pe-spatial-l14-448 | 0.49 |
| c-radiov4-h | 0.70 |
| **sum over rows with a rate** | **32.37** |
| rows without a rate | 0 |

Subset size that keeps the sum at <= 24 GPU-h (per tag, linear scaling): a4000: 59305 units

### all roster candidates with a measured rate

| FM | a4000: 80k units GPU-h |
|---|---|
| dinov3-vitl16 | 0.39 |
| sd21-dift-gate | 17.99 |
| cleandift-sd21 | 12.80 |
| cleandift-sd15 | 5.10 |
| pe-spatial-b16-512 | 0.12 |
| pe-spatial-l14-448 | 0.49 |
| pe-spatial-g14-448 | 2.63 |
| c-radiov3-l | 0.34 |
| c-radiov3-h | 0.69 |
| c-radiov3-g | 1.24 |
| c-radiov4-h | 0.70 |
| c-radiov4-so400m | 0.47 |
| siglip2-giant-opt-patch16-256 | 1.28 |
| dinov3-vith16plus | 1.09 |
| dinov3-vit7b16 | 1.22 |
| tipsv2-so400m14 | 0.73 |
| tipsv2-g14 | 1.91 |
| dinov3-convnext-large | 0.25 |
| siglip2-so400m-patch14-224 | 0.71 |
| dinov2-large | 0.53 |
| **sum over rows with a rate** | **50.69** |
| rows without a rate | 0 |

Subset size that keeps the sum at <= 24 GPU-h (per tag, linear scaling): a4000: 37879 units
