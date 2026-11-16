# FMDX gate read (PROVISIONAL_2026-11-05T2029Z)

Snapshot of CLUSTER-A files taken 2026-11-05T20:29Z (exp/FMD/xs/read_v1/). Prereg FMD_PREREG_2026-11-03.md sha256 7a71c715dc5ca1cf146f829b8c7f498811e1330e1eefa2bf1b3b60fddbc5d154 (A0-A9). Rules applied: Section 5 (Urdu), A3.5 (per-set SHAPE/ORDER, xs_pass), A3.6 (status, early FAIL), A3.c.3 (cut-off 2026-11-06T23:59:59Z), A4.3(a) (cancel rule). Statuses recomputed from rows by code/fmdx_gate_read.py; verdict files read only for comparison. Read by the XS gate read track (<agent>; <agent> because verdict-setting is an <agent> task under the 30 Oct model policy and the <agent> API failed earlier).

**PROVISIONAL: read before the screen cut-off. Rows added before 2026-11-06T23:59:59Z can change these statuses; the FINAL read follows the cut-off.**

## Screen [reason: A3 screen-all roster]

- FMs with a verdict file or rows: 112. Recomputed status counts: FAIL 85, PASS_XS 19, PASS_URDU_ONLY 6, PENDING 2.
- Mismatches with verdict_*.json (status, xs_pass, urdu_pass, sets_shape/order/passed): 0. Per-set t0_ci_lo, complete flags and order_passed also compared: differences are label format only (file appends "(zeroshot,zeroshot)" to zero-shot criteria; 10 FM-set cells).
- Not assessed: inkfm: not assessed: no public weights (verdicts/blocked/blocked_inkfm.json); sam3.1: not assessed: no extractor (A3.a.2/A3.b; verdict status_reason).

| fm | family | status | Urdu T0 CI lo | Urdu order passed | sets complete | shape passed | order passed | sets passed | arms | cancellation check |
|---|---|---|---|---|---|---|---|---|---|---|
| c-radiov3-b | encoder | PASS_XS | 0.04079 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 5 | 5 | 5 | 5 | 2 | OK no gate cancellation |
| c-radiov3-g | encoder | PASS_XS | 0.05785 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 5 | 5 | 5 | 5 | 2 | OK no gate cancellation |
| c-radiov3-h | encoder | PASS_XS | 0.06157 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 5 | 5 | 5 | 5 | 2 | OK no gate cancellation |
| c-radiov3-l | encoder | PASS_XS | 0.05201 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 5 | 5 | 5 | 5 | 2 | OK no gate cancellation |
| c-radiov4-h | encoder | PASS_XS | 0.03910 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 5 | 5 | 5 | 5 | 2 | OK no gate cancellation |
| c-radiov4-so400m | encoder | PASS_XS | 0.02797 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 5 | 5 | 5 | 5 | 2 | OK no gate cancellation |
| dinov3-vit7b16 | encoder | PASS_XS | 0.02490 | T2:rprec;T3:rprec;T4:spearman | 5 | 5 | 5 | 5 | 2 | OK no gate cancellation |
| dinov3-vitb16 | encoder | PASS_XS | 0.01928 | T2:rprec;T3:rprec | 5 | 5 | 4 | 4 | 2 | OK no gate cancellation |
| dinov3-vitl16-sat493m | encoder | PASS_XS | 0.06527 | T1:acc8;T2:rprec;T3:rprec | 5 | 5 | 4 | 4 | 0 | no arm_queue rows |
| flux1-dev | diffusion | PASS_XS | 0.01742 | T1:acc8;T2:rprec;T3:rprec | 5 | 5 | 5 | 5 | 2 | OK no gate cancellation |
| flux1-schnell | diffusion | PASS_XS | 0.01682 | T1:acc8;T2:rprec;T3:rprec | 5 | 5 | 5 | 5 | 2 | OK no gate cancellation |
| inksight_small_p | ink | PASS_XS | 0.02456 |  | 5 | 5 | 3 | 3 | 2 | OK no gate cancellation |
| sd3-medium-diffusers | diffusion | PASS_XS | 0.01244 | T2:rprec;T3:rprec | 5 | 5 | 5 | 5 | 2 | OK no gate cancellation |
| sd3.5-large-diffusers-layout | diffusion | PASS_XS | 0.01627 | T1:acc8;T2:rprec;T3:rprec | 5 | 5 | 4 | 4 | 2 | OK no gate cancellation |
| sd3.5-large-turbo-diffusers-layout | diffusion | PASS_XS | 0.01534 | T1:acc8;T2:rprec;T3:rprec | 5 | 5 | 4 | 4 | 2 | OK no gate cancellation |
| sd3.5-medium-diffusers-layout | diffusion | PASS_XS | 0.01422 | T2:rprec;T3:rprec | 5 | 5 | 4 | 4 | 2 | OK no gate cancellation |
| vit-mae-large | encoder | PASS_XS | 0.06930 | T1:acc8;T2:rprec;T3:rprec | 5 | 5 | 5 | 5 | 2 | OK no gate cancellation |
| webssl-mae1b-224 | encoder | PASS_XS | 0.01871 | T1:acc8;T2:rprec;T3:rprec | 5 | 5 | 5 | 5 | 0 | no arm_queue rows |
| webssl-mae300m-224 | encoder | PASS_XS | 0.06598 | T1:acc8;T2:rprec;T3:rprec | 5 | 5 | 5 | 5 | 2 | OK no gate cancellation |
| chandra-ocr-2 | VLM | PASS_URDU_ONLY | 0.02608 | T2:rprec | 5 | 4 | 1 | 1 | 0 | no arm_queue rows |
| cleandift-sd21 | diffusion | PASS_URDU_ONLY | -0.00698 | T1:acc8;T2:rprec;T3:rprec | 5 | 3 | 5 | 3 | 2 | OK no gate cancellation |
| dinov3-vitl16 | encoder | PASS_URDU_ONLY | -0.00030 | T2:rprec | 5 | 4 | 2 | 2 | 2 | OK no gate cancellation |
| dinov3-vits16 | encoder | PASS_URDU_ONLY | 0.00366 | T2:rprec | 5 | 5 | 0 | 0 | 2 | OK no gate cancellation |
| pe-spatial-b16-512 | encoder | PASS_URDU_ONLY | 0.01315 | T3:rprec | 5 | 5 | 2 | 2 | 2 | OK no gate cancellation |
| qwen3.5-4b | VLM | PASS_URDU_ONLY | 0.03575 | T2:rprec;T3:rprec | 5 | 4 | 1 | 1 | 2 | OK no gate cancellation |
| inkfm | nan | not assessed: no public weights (verdicts/blocked/blocked_inkfm.json) |  |  | 0 | 0 | 0 | 0 | 2 | OK no gate cancellation |
| sam3.1 | nan | not assessed: no extractor (A3.a.2/A3.b; verdict status_reason) |  |  | 0 | 0 | 0 | 0 | 0 | no arm_queue rows |
| aimv2-1b-patch14-224 | encoder | FAIL | -0.16107 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| aimv2-3b-patch14-224 | encoder | FAIL | -0.13415 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| aimv2-huge-patch14-224 | encoder | FAIL | -0.18341 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| aimv2-large-patch14-224 | encoder | FAIL | -0.19005 |  | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| aimv2-large-patch14-224-distilled | encoder | FAIL | -0.17401 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| arabic-english-hw-ocr-v3 | VLM | FAIL | -0.18114 | T2:rprec;T3:rprec | 5 | 0 | 1 | 0 | 0 | no arm_queue rows |
| arabic-ocr-deepseek-ocr-2 | VLM | FAIL | -0.26092 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| chandra | VLM | FAIL | -0.12973 | T2:rprec;T3:rprec | 5 | 0 | 3 | 0 | 0 | no arm_queue rows |
| cleandift-sd15 | diffusion | FAIL | -0.02261 | T1:acc8;T2:rprec;T3:rprec | 5 | 0 | 5 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| deepseek-ocr | VLM | FAIL | -0.13728 | T3:rprec;T4:spearman | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| deepseek-ocr-2 | VLM | FAIL | -0.26092 |  | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| depth-anything-3-base | encoder | FAIL | -0.02574 |  | 5 | 1 | 0 | 0 | 0 | no arm_queue rows |
| depth-anything-3-giant-1.1 | encoder | FAIL | -0.01067 | T2:rprec;T3:rprec | 5 | 1 | 3 | 1 | 0 | no arm_queue rows |
| depth-anything-3-large-1.1 | encoder | FAIL | -0.02083 | T2:rprec;T3:rprec | 5 | 1 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| depth-anything-v2-large | encoder | FAIL | -0.05529 | T2:rprec | 5 | 0 | 3 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| dinov2-base | encoder | FAIL | -0.09090 | T2:rprec | 5 | 0 | 1 | 0 | 0 | no arm_queue rows |
| dinov2-giant | encoder | FAIL | -0.05424 | T2:rprec;T3:rprec;T4:spearman | 5 | 0 | 3 | 0 | 0 | no arm_queue rows |
| dinov2-large | encoder | FAIL | -0.09493 | T2:rprec | 5 | 0 | 2 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| dinov2-reg-base | encoder | FAIL | -0.08719 | T2:rprec | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| dinov2-reg-giant | encoder | FAIL | -0.07013 | T2:rprec | 5 | 0 | 2 | 0 | 0 | no arm_queue rows |
| dinov2-reg-large | encoder | FAIL | -0.10101 | T2:rprec | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| dinov2-reg-small | encoder | FAIL | -0.10724 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| dinov3-convnext-base | encoder | FAIL | -0.09885 |  | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| dinov3-convnext-large | encoder | FAIL | -0.13752 |  | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| dinov3-convnext-small | encoder | FAIL | -0.09153 |  | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| dinov3-convnext-tiny | encoder | FAIL | -0.03986 | T2:rprec | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| dinov3-vith16plus | encoder | FAIL | -0.01631 | T2:rprec | 5 | 2 | 2 | 2 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| dinov3-vits16plus | encoder | FAIL | -0.01748 |  | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| dots-ocr | VLM | FAIL | -0.22628 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| florence-2-base | encoder | FAIL | -0.03750 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 5 | 0 | 5 | 0 | 0 | no arm_queue rows |
| florence-2-large | encoder | FAIL | -0.03384 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 5 | 0 | 5 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| florence-2-large-ft | encoder | FAIL | -0.06247 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 5 | 0 | 5 | 0 | 0 | no arm_queue rows |
| got-ocr2-hf | VLM | FAIL | -0.30058 |  | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| lightonocr-2-1b | VLM | FAIL | -0.13289 | T2:rprec | 5 | 0 | 1 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| paddleocr-vl-1.6 | VLM | FAIL | -0.21158 |  | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| pe-core-b16-224 | encoder | FAIL | -0.16233 |  | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| pe-core-g14-448 | encoder | FAIL | -0.25525 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| pe-core-l14-336 | encoder | FAIL | -0.26003 |  | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| pe-core-s16-384 | encoder | FAIL | -0.16144 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| pe-core-t16-384 | encoder | FAIL | -0.11149 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| pe-lang-g14-448-timm | encoder | FAIL | -0.17828 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| pe-lang-l14-448-timm | encoder | FAIL | -0.19033 | T2:rprec | 5 | 0 | 1 | 0 | 0 | no arm_queue rows |
| pe-spatial-g14-448 | encoder | FAIL | -0.06411 | T2:rprec | 5 | 0 | 1 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| pe-spatial-l14-448 | encoder | FAIL | -0.05158 |  | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| pe-spatial-s16-512 | encoder | FAIL | 0.00336 |  | 5 | 4 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| pe-spatial-t16-512 | encoder | FAIL | 0.00175 |  | 5 | 1 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| qaari-0.1-urdu-ocr | VLM | FAIL | -0.10225 | T3:rprec | 5 | 0 | 1 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| qari-ocr-v0.3-vl-2b | VLM | FAIL | -0.10691 | T3:rprec | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| qwen2-vl-2b-instruct | VLM | FAIL | -0.10328 | T2:rprec;T3:rprec | 5 | 0 | 1 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| qwen3-vl-8b-instruct | VLM | FAIL | -0.18007 |  | 5 | 0 | 2 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| sam2.1-hiera-base-plus | encoder | FAIL | -0.01996 | T2:rprec | 5 | 0 | 3 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| sam3 | encoder | FAIL | -0.01944 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 5 | 0 | 5 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| sd1.5-dift | diffusion | FAIL | -0.02838 | T1:acc8;T2:rprec;T3:rprec | 5 | 0 | 5 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| sd2.1-dift | diffusion | FAIL | -0.01268 | T1:acc8;T2:rprec;T3:rprec | 5 | 1 | 5 | 1 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| siglip2-base-patch16-224 | encoder | FAIL | -0.10871 | T2:rprec;T3:rprec | 5 | 0 | 2 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| siglip2-giant-opt-patch16-256 | encoder | FAIL | -0.13624 | T2:rprec;T3:rprec | 5 | 0 | 4 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| siglip2-giant-opt-patch16-384 | encoder | FAIL | -0.13616 | T2:rprec;T3:rprec;T4:spearman | 5 | 0 | 4 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| siglip2-large-patch16-256 | encoder | FAIL | -0.13128 | T2:rprec;T3:rprec | 5 | 0 | 4 | 0 | 0 | no arm_queue rows |
| siglip2-so400m-patch14-224 | encoder | FAIL | -0.16072 | T2:rprec;T3:rprec | 5 | 0 | 3 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| siglip2-so400m-patch16-256 | encoder | FAIL | -0.15147 | T2:rprec;T3:rprec | 5 | 0 | 3 | 0 | 0 | no arm_queue rows |
| starvector-1b-im2svg | vector | FAIL | -0.18607 |  | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| starvector-8b-im2svg | vector | FAIL | -0.12811 | T2:rprec;T3:rprec | 5 | 0 | 3 | 0 | 0 | no arm_queue rows |
| theia-base-cddsv | encoder | FAIL | 0.06688 |  | 5 | 5 | 0 | 0 | 0 | no arm_queue rows |
| theia-base-cdiv | encoder | FAIL | 0.05840 |  | 5 | 5 | 0 | 0 | 0 | no arm_queue rows |
| theia-small-cdiv | encoder | FAIL | 0.05525 |  | 5 | 5 | 0 | 0 | 0 | no arm_queue rows |
| theia-tiny-cddsv | encoder | FAIL | 0.04439 |  | 5 | 5 | 0 | 0 | 0 | no arm_queue rows |
| theia-tiny-cdiv | encoder | FAIL | 0.05715 |  | 5 | 5 | 0 | 0 | 0 | no arm_queue rows |
| tipsv1-b14 | encoder | FAIL | -0.12751 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| tipsv1-g14 | encoder | FAIL | -0.09591 | T2:rprec;T3:rprec | 5 | 0 | 3 | 0 | 0 | no arm_queue rows |
| tipsv1-l14 | encoder | FAIL | -0.11158 | T2:rprec | 5 | 0 | 1 | 0 | 0 | no arm_queue rows |
| tipsv1-s14 | encoder | FAIL | -0.12005 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| tipsv1-so400m14 | encoder | FAIL | -0.13302 | T2:rprec | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| tipsv2-b14 | encoder | FAIL | -0.08383 | T2:rprec | 5 | 0 | 1 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| tipsv2-g14 | encoder | FAIL | -0.06146 | T2:rprec;T3:rprec;T4:spearman | 5 | 0 | 2 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| tipsv2-l14 | encoder | FAIL | -0.07838 | T2:rprec | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| tipsv2-so400m14 | encoder | FAIL | -0.07837 | T2:rprec;T4:spearman | 5 | 0 | 1 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| trocr-base-handwritten | encoder | FAIL | -0.18842 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| trocr-base-hw-hist-swe-2 | encoder | FAIL | -0.11585 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| trocr-large-handwritten | encoder | FAIL | -0.18207 |  | 5 | 0 | 1 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| trocr-small-handwritten | encoder | FAIL | -0.17756 |  | 5 | 0 | 0 | 0 | 0 | no arm_queue rows |
| urdu-ocr-deepseek-nomy | VLM | FAIL | -0.13728 | T3:rprec;T4:spearman | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| urdu-ocr-deepseek-ocr2-kingabzpro | VLM | FAIL | -0.26092 |  | 5 | 0 | 0 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| webssl-dino1b-224 | encoder | FAIL | -0.08009 | T2:rprec;T3:rprec | 5 | 0 | 2 | 0 | 2 | OK all arms CANCELLED_GATE_FAIL after first FAIL verdict |
| webssl-dino2b-224 | encoder | FAIL | -0.03950 | T2:rprec;T3:rprec;T4:spearman | 5 | 0 | 4 | 0 | 0 | no arm_queue rows |
| webssl-dino300m-224 | encoder | FAIL | -0.06549 | T2:rprec;T3:rprec | 5 | 0 | 4 | 0 | 0 | no arm_queue rows |

Per-set CI lo values (T0 and best order criterion, comparators, n units / writers): fmdx_gate_read_sets.csv. All rows: fmdx_screen_table.csv.

## Cancellations [reason: A4.3(a) cancel rule check]

- arm_queue.csv rows: 138. States: CANCELLED_GATE_FAIL 88, SUBMITTED 28, PINNED_CLUSTER-B 17, DONE 3, QUEUED 2.
- CANCELLED_GATE_FAIL entries in cancellation_log.md: 88 (88 unique arms). All belong to FMs with recomputed status FAIL; for every entry a FAIL verdict (current or history) has created_utc <= the cancellation time.
- FAIL FMs: 85; with arm rows: 44 (all arms CANCELLED_GATE_FAIL); without arm rows: 41.
- PASS_XS / PASS_URDU_ONLY / control arms with a gate cancellation: 0. Control arms: NOFM_W=DONE, PEINIT_W=DONE, NOFM_S=PINNED_CLUSTER-B, PEINIT_S=PINNED_CLUSTER-B.
- Non-FAIL FMs without arm_queue rows: chandra-ocr-2 (PASS_URDU_ONLY), dinov3-vitl16-sat493m (PASS_XS), webssl-mae1b-224 (PASS_XS). Their names appear in cand_log.md only as controller verdict-sync lines.
- cand_log.md race cancellations (losing copies) are not gate cancellations and are not counted here.

## Not covered by this read

- priority_score and m_ord/m_t0 were not recomputed (they set queue order, not status).
- Probe device (CUDA/CPU) per row is not tabulated here; it is recorded in the per-unit metas on CLUSTER-A.
