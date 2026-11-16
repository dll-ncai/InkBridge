# RESULTS_FMDX v1.1 (FINAL screen)

Screen read: **FINAL (cut-off 2026-11-06T23:59:59Z recorded by the encoders cut-off job 24850413; snapshot 2026-11-07T00:47Z)**. Warm tables snapshot: FMDX_warm_contrasts.csv mtime 2026-11-06T00:42Z, fetched 2026-11-07T15:00Z. Arm status captured: 2026-11-07T15:01:45Z. Prereg exp/FMD/plans/FMD_PREREG_2026-11-03.md, sha256 7a71c715dc5ca1cf146f829b8c7f498811e1330e1eefa2bf1b3b60fddbc5d154. Written by the FMDX XS gate read track, <agent>. All backbone and FT results are EXPLORATORY (single training seed). Facts only.

Artifacts: [GATE_READ.md]({{artifact:8422a091-0250-424e-97d5-9584609ba1bb}}), [fmdx_screen_table.csv]({{artifact:bcefeaef-2be6-457f-a049-b883c47dbe4e}}), [fmdx_gate_read.csv]({{artifact:4531786f-7d6c-4d25-8bab-f337fd87f737}}), [fmdx_gate_read_sets.csv]({{artifact:8fa95c9b-420d-4611-a64c-18972d05a1be}}), [fmdx_warm_read.csv]({{artifact:e4831eb1-2d9c-40a4-8343-b682ae7aa365}}), [fmdx_warm_read_clauses.csv]({{artifact:dad0c417-0e0f-4383-a4cd-07808d5c20a7}}), [fmdx_screen_heatmap.png]({{artifact:b28900d4-2f6b-449c-ac75-66e6758f3c59}}), [fmdx_warm_forest.png]({{artifact:528f5e0d-4c83-457b-9450-e97eb58759c3}}), [arm_status_v1_1.csv]({{artifact:1e0b41bd-0c48-4aaf-bbfb-709c6279d1eb}}).

## 1. FM screen, Urdu + 5 cross-script sets [reason: A3 screen-all roster; A3.5/A3.6 status]

- FMs: 116. Status (recomputed from rows): FAIL 85, PASS_XS 19, PASS_URDU_ONLY 6, PENDING 6. Mismatches against verdict_*.json: 0.
- Shape margin 0.01035 (T0 IoU CI lo > -0.01035); order: at least one criterion CI lo > 0; xs_pass = SHAPE on all 5 cross-script sets AND ORDER on >= 3. Comparator per set and target: best of {LUO ms6, M5 joint ms9} x {mid16, cat16}; Urdu: best nominee encoder probe. Writer-cluster bootstrap B = 2,000.
- Columns: min T0 CI lo and min best-order CI lo are taken over the 5 cross-script sets with rows.

| fm | family | status | Urdu T0 CI lo | Urdu order passed | min xs T0 CI lo | min xs best-order CI lo | sets shape / order / both |
|---|---|---|---|---|---|---|---|
| flux1-dev | diffusion | PASS_XS | 0.01742 | T1:acc8;T2:rprec;T3:rprec | 0.00682 | 0.02320 | 5 / 5 / 5 |
| flux1-schnell | diffusion | PASS_XS | 0.01682 | T1:acc8;T2:rprec;T3:rprec | 0.00566 | 0.02897 | 5 / 5 / 5 |
| sd3-medium-diffusers | diffusion | PASS_XS | 0.01244 | T2:rprec;T3:rprec | -0.00059 | 0.00305 | 5 / 5 / 5 |
| sd3.5-large-diffusers-layout | diffusion | PASS_XS | 0.01627 | T1:acc8;T2:rprec;T3:rprec | 0.00486 | -0.00634 | 5 / 4 / 4 |
| sd3.5-large-turbo-diffusers-layout | diffusion | PASS_XS | 0.01534 | T1:acc8;T2:rprec;T3:rprec | 0.00392 | -0.00427 | 5 / 4 / 4 |
| sd3.5-medium-diffusers-layout | diffusion | PASS_XS | 0.01422 | T2:rprec;T3:rprec | 0.00146 | -0.00752 | 5 / 4 / 4 |
| c-radiov3-b | encoder | PASS_XS | 0.04079 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 0.01995 | 0.03872 | 5 / 5 / 5 |
| c-radiov3-g | encoder | PASS_XS | 0.05785 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 0.03019 | 0.03703 | 5 / 5 / 5 |
| c-radiov3-h | encoder | PASS_XS | 0.06157 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 0.03501 | 0.04055 | 5 / 5 / 5 |
| c-radiov3-l | encoder | PASS_XS | 0.05201 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 0.02539 | 0.02151 | 5 / 5 / 5 |
| c-radiov4-h | encoder | PASS_XS | 0.03910 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 0.01461 | 0.02709 | 5 / 5 / 5 |
| c-radiov4-so400m | encoder | PASS_XS | 0.02797 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | 0.00443 | 0.00912 | 5 / 5 / 5 |
| dinov3-vit7b16 | encoder | PASS_XS | 0.02490 | T2:rprec;T3:rprec;T4:spearman | 0.00048 | 0.02271 | 5 / 5 / 5 |
| dinov3-vitb16 | encoder | PASS_XS | 0.01928 | T2:rprec;T3:rprec | -0.00174 | -0.00582 | 5 / 4 / 4 |
| dinov3-vitl16-sat493m | encoder | PASS_XS | 0.06527 | T1:acc8;T2:rprec;T3:rprec | 0.03929 | -0.02176 | 5 / 4 / 4 |
| vit-mae-large | encoder | PASS_XS | 0.06930 | T1:acc8;T2:rprec;T3:rprec | 0.04188 | 0.04031 | 5 / 5 / 5 |
| webssl-mae1b-224 | encoder | PASS_XS | 0.01871 | T1:acc8;T2:rprec;T3:rprec | 0.01335 | 0.03983 | 5 / 5 / 5 |
| webssl-mae300m-224 | encoder | PASS_XS | 0.06598 | T1:acc8;T2:rprec;T3:rprec | 0.03800 | 0.03043 | 5 / 5 / 5 |
| inksight_small_p | ink | PASS_XS | 0.02456 |  | 0.00308 | -0.18669 | 5 / 3 / 3 |
| chandra-ocr-2 | VLM | PASS_URDU_ONLY | 0.02608 | T2:rprec | -0.02530 | -0.10839 | 4 / 1 / 1 |
| qwen3.5-4b | VLM | PASS_URDU_ONLY | 0.03575 | T2:rprec;T3:rprec | -0.02632 | -0.09624 | 4 / 1 / 1 |
| cleandift-sd21 | diffusion | PASS_URDU_ONLY | -0.00698 | T1:acc8;T2:rprec;T3:rprec | -0.01497 | 0.01719 | 3 / 5 / 3 |
| dinov3-vitl16 | encoder | PASS_URDU_ONLY | -0.00030 | T2:rprec | -0.01213 | -0.06774 | 4 / 2 / 2 |
| dinov3-vits16 | encoder | PASS_URDU_ONLY | 0.00366 | T2:rprec | -0.00740 | -0.08044 | 5 / 0 / 0 |
| pe-spatial-b16-512 | encoder | PASS_URDU_ONLY | 0.01315 | T3:rprec | -0.00680 | -0.02585 | 5 / 2 / 2 |
| inkfm | nan | not assessed: no public weights (verdicts/blocked/blocked_inkfm.json) |  |  |  |  | 0 / 0 / 0 |
| sam3.1 | nan | not assessed: no extractor (A3.a.2/A3.b; verdict status_reason) |  |  |  |  | 0 / 0 / 0 |
| webssl-dino3b-224 | nan | not assessed: screen cut-off (A3.c.3) |  |  |  |  | 0 / 0 / 0 |
| webssl-dino5b-224 | nan | not assessed: screen cut-off (A3.c.3) |  |  |  |  | 0 / 0 / 0 |
| webssl-dino7b-224 | nan | not assessed: screen cut-off (A3.c.3) |  |  |  |  | 0 / 0 / 0 |
| webssl-mae3b-224 | nan | not assessed: screen cut-off (A3.c.3) |  |  |  |  | 0 / 0 / 0 |
| arabic-english-hw-ocr-v3 | VLM | FAIL | -0.18114 | T2:rprec;T3:rprec | -0.25712 | -0.09869 | 0 / 1 / 0 |
| arabic-ocr-deepseek-ocr-2 | VLM | FAIL | -0.26092 |  | -0.30672 | -0.08416 | 0 / 0 / 0 |
| chandra | VLM | FAIL | -0.12973 | T2:rprec;T3:rprec | -0.22335 | -0.08717 | 0 / 3 / 0 |
| deepseek-ocr | VLM | FAIL | -0.13728 | T3:rprec;T4:spearman | -0.19408 | -0.02766 | 0 / 0 / 0 |
| deepseek-ocr-2 | VLM | FAIL | -0.26092 |  | -0.30672 | -0.08416 | 0 / 0 / 0 |
| dots-ocr | VLM | FAIL | -0.22628 |  | -0.34176 | -0.12377 | 0 / 0 / 0 |
| got-ocr2-hf | VLM | FAIL | -0.30058 |  | -0.41248 | -0.15718 | 0 / 0 / 0 |
| lightonocr-2-1b | VLM | FAIL | -0.13289 | T2:rprec | -0.22393 | -0.03178 | 0 / 1 / 0 |
| paddleocr-vl-1.6 | VLM | FAIL | -0.21158 |  | -0.29036 | -0.10163 | 0 / 0 / 0 |
| qaari-0.1-urdu-ocr | VLM | FAIL | -0.10225 | T3:rprec | -0.15228 | -0.13258 | 0 / 1 / 0 |
| qari-ocr-v0.3-vl-2b | VLM | FAIL | -0.10691 | T3:rprec | -0.16643 | -0.13630 | 0 / 0 / 0 |
| qwen2-vl-2b-instruct | VLM | FAIL | -0.10328 | T2:rprec;T3:rprec | -0.16198 | -0.12451 | 0 / 1 / 0 |
| qwen3-vl-8b-instruct | VLM | FAIL | -0.18007 |  | -0.27341 | -0.11869 | 0 / 2 / 0 |
| urdu-ocr-deepseek-nomy | VLM | FAIL | -0.13728 | T3:rprec;T4:spearman | -0.19408 | -0.02766 | 0 / 0 / 0 |
| urdu-ocr-deepseek-ocr2-kingabzpro | VLM | FAIL | -0.26092 |  | -0.30672 | -0.08416 | 0 / 0 / 0 |
| cleandift-sd15 | diffusion | FAIL | -0.02261 | T1:acc8;T2:rprec;T3:rprec | -0.04127 | 0.00757 | 0 / 5 / 0 |
| sd1.5-dift | diffusion | FAIL | -0.02838 | T1:acc8;T2:rprec;T3:rprec | -0.05786 | 0.00711 | 0 / 5 / 0 |
| sd2.1-dift | diffusion | FAIL | -0.01268 | T1:acc8;T2:rprec;T3:rprec | -0.02586 | 0.02883 | 1 / 5 / 1 |
| aimv2-1b-patch14-224 | encoder | FAIL | -0.16107 |  | -0.23074 | -0.21444 | 0 / 0 / 0 |
| aimv2-3b-patch14-224 | encoder | FAIL | -0.13415 |  | -0.19473 | -0.19688 | 0 / 0 / 0 |
| aimv2-huge-patch14-224 | encoder | FAIL | -0.18341 |  | -0.25060 | -0.20727 | 0 / 0 / 0 |
| aimv2-large-patch14-224 | encoder | FAIL | -0.19005 |  | -0.27514 | -0.20703 | 0 / 0 / 0 |
| aimv2-large-patch14-224-distilled | encoder | FAIL | -0.17401 |  | -0.24101 | -0.22689 | 0 / 0 / 0 |
| depth-anything-3-base | encoder | FAIL | -0.02574 |  | -0.05681 | -0.05886 | 1 / 0 / 0 |
| depth-anything-3-giant-1.1 | encoder | FAIL | -0.01067 | T2:rprec;T3:rprec | -0.03555 | -0.03524 | 1 / 3 / 1 |
| depth-anything-3-large-1.1 | encoder | FAIL | -0.02083 | T2:rprec;T3:rprec | -0.05968 | -0.04532 | 1 / 0 / 0 |
| depth-anything-v2-large | encoder | FAIL | -0.05529 | T2:rprec | -0.05990 | -0.04896 | 0 / 3 / 0 |
| dinov2-base | encoder | FAIL | -0.09090 | T2:rprec | -0.09803 | -0.09455 | 0 / 1 / 0 |
| dinov2-giant | encoder | FAIL | -0.05424 | T2:rprec;T3:rprec;T4:spearman | -0.06134 | -0.01365 | 0 / 3 / 0 |
| dinov2-large | encoder | FAIL | -0.09493 | T2:rprec | -0.09586 | -0.07960 | 0 / 2 / 0 |
| dinov2-reg-base | encoder | FAIL | -0.08719 | T2:rprec | -0.10405 | -0.10044 | 0 / 0 / 0 |
| dinov2-reg-giant | encoder | FAIL | -0.07013 | T2:rprec | -0.07787 | -0.05321 | 0 / 2 / 0 |
| dinov2-reg-large | encoder | FAIL | -0.10101 | T2:rprec | -0.11221 | -0.09172 | 0 / 0 / 0 |
| dinov2-reg-small | encoder | FAIL | -0.10724 |  | -0.15204 | -0.11967 | 0 / 0 / 0 |
| dinov3-convnext-base | encoder | FAIL | -0.09885 |  | -0.12798 | -0.15329 | 0 / 0 / 0 |
| dinov3-convnext-large | encoder | FAIL | -0.13752 |  | -0.16730 | -0.14951 | 0 / 0 / 0 |
| dinov3-convnext-small | encoder | FAIL | -0.09153 |  | -0.11883 | -0.11575 | 0 / 0 / 0 |
| dinov3-convnext-tiny | encoder | FAIL | -0.03986 | T2:rprec | -0.06372 | -0.06526 | 0 / 0 / 0 |
| dinov3-vith16plus | encoder | FAIL | -0.01631 | T2:rprec | -0.04452 | -0.03830 | 2 / 2 / 2 |
| dinov3-vits16plus | encoder | FAIL | -0.01748 |  | -0.03798 | -0.10535 | 0 / 0 / 0 |
| florence-2-base | encoder | FAIL | -0.03750 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | -0.06704 | 0.01213 | 0 / 5 / 0 |
| florence-2-large | encoder | FAIL | -0.03384 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | -0.05382 | 0.01042 | 0 / 5 / 0 |
| florence-2-large-ft | encoder | FAIL | -0.06247 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | -0.08940 | 0.00294 | 0 / 5 / 0 |
| pe-core-b16-224 | encoder | FAIL | -0.16233 |  | -0.21173 | -0.13278 | 0 / 0 / 0 |
| pe-core-g14-448 | encoder | FAIL | -0.25525 |  | -0.34874 | -0.25191 | 0 / 0 / 0 |
| pe-core-l14-336 | encoder | FAIL | -0.26003 |  | -0.39031 | -0.28451 | 0 / 0 / 0 |
| pe-core-s16-384 | encoder | FAIL | -0.16144 |  | -0.20409 | -0.13563 | 0 / 0 / 0 |
| pe-core-t16-384 | encoder | FAIL | -0.11149 |  | -0.18316 | -0.17703 | 0 / 0 / 0 |
| pe-lang-g14-448-timm | encoder | FAIL | -0.17828 |  | -0.24732 | -0.08505 | 0 / 0 / 0 |
| pe-lang-l14-448-timm | encoder | FAIL | -0.19033 | T2:rprec | -0.25636 | -0.07525 | 0 / 1 / 0 |
| pe-spatial-g14-448 | encoder | FAIL | -0.06411 | T2:rprec | -0.10422 | -0.08652 | 0 / 1 / 0 |
| pe-spatial-l14-448 | encoder | FAIL | -0.05158 |  | -0.09309 | -0.11319 | 0 / 0 / 0 |
| pe-spatial-s16-512 | encoder | FAIL | 0.00336 |  | -0.01212 | -0.06268 | 4 / 0 / 0 |
| pe-spatial-t16-512 | encoder | FAIL | 0.00175 |  | -0.02673 | -0.14219 | 1 / 0 / 0 |
| sam2.1-hiera-base-plus | encoder | FAIL | -0.01996 | T2:rprec | -0.05088 | -0.02973 | 0 / 3 / 0 |
| sam3 | encoder | FAIL | -0.01944 | T1:acc8;T2:rprec;T3:rprec;T4:spearman | -0.02726 | 0.02664 | 0 / 5 / 0 |
| siglip2-base-patch16-224 | encoder | FAIL | -0.10871 | T2:rprec;T3:rprec | -0.17304 | -0.01510 | 0 / 2 / 0 |
| siglip2-giant-opt-patch16-256 | encoder | FAIL | -0.13624 | T2:rprec;T3:rprec | -0.17641 | -0.00019 | 0 / 4 / 0 |
| siglip2-giant-opt-patch16-384 | encoder | FAIL | -0.13616 | T2:rprec;T3:rprec;T4:spearman | -0.17830 | -0.01227 | 0 / 4 / 0 |
| siglip2-large-patch16-256 | encoder | FAIL | -0.13128 | T2:rprec;T3:rprec | -0.19826 | -0.01141 | 0 / 4 / 0 |
| siglip2-so400m-patch14-224 | encoder | FAIL | -0.16072 | T2:rprec;T3:rprec | -0.23609 | -0.03238 | 0 / 3 / 0 |
| siglip2-so400m-patch16-256 | encoder | FAIL | -0.15147 | T2:rprec;T3:rprec | -0.21791 | -0.02313 | 0 / 3 / 0 |
| theia-base-cddsv | encoder | FAIL | 0.06688 |  | 0.03915 | -0.06931 | 5 / 0 / 0 |
| theia-base-cdiv | encoder | FAIL | 0.05840 |  | 0.03111 | -0.04926 | 5 / 0 / 0 |
| theia-small-cdiv | encoder | FAIL | 0.05525 |  | 0.03241 | -0.10109 | 5 / 0 / 0 |
| theia-tiny-cddsv | encoder | FAIL | 0.04439 |  | 0.00456 | -0.10747 | 5 / 0 / 0 |
| theia-tiny-cdiv | encoder | FAIL | 0.05715 |  | 0.02543 | -0.10538 | 5 / 0 / 0 |
| tipsv1-b14 | encoder | FAIL | -0.12751 |  | -0.15087 | -0.07747 | 0 / 0 / 0 |
| tipsv1-g14 | encoder | FAIL | -0.09591 | T2:rprec;T3:rprec | -0.11375 | -0.03415 | 0 / 3 / 0 |
| tipsv1-l14 | encoder | FAIL | -0.11158 | T2:rprec | -0.13493 | -0.08221 | 0 / 1 / 0 |
| tipsv1-s14 | encoder | FAIL | -0.12005 |  | -0.17015 | -0.09569 | 0 / 0 / 0 |
| tipsv1-so400m14 | encoder | FAIL | -0.13302 | T2:rprec | -0.16935 | -0.08538 | 0 / 0 / 0 |
| tipsv2-b14 | encoder | FAIL | -0.08383 | T2:rprec | -0.10019 | -0.08544 | 0 / 1 / 0 |
| tipsv2-g14 | encoder | FAIL | -0.06146 | T2:rprec;T3:rprec;T4:spearman | -0.07539 | -0.06188 | 0 / 2 / 0 |
| tipsv2-l14 | encoder | FAIL | -0.07838 | T2:rprec | -0.09112 | -0.07210 | 0 / 0 / 0 |
| tipsv2-so400m14 | encoder | FAIL | -0.07837 | T2:rprec;T4:spearman | -0.09379 | -0.07875 | 0 / 1 / 0 |
| trocr-base-handwritten | encoder | FAIL | -0.18842 |  | -0.26044 | -0.03872 | 0 / 0 / 0 |
| trocr-base-hw-hist-swe-2 | encoder | FAIL | -0.11585 |  | -0.15013 | -0.05500 | 0 / 0 / 0 |
| trocr-large-handwritten | encoder | FAIL | -0.18207 |  | -0.27030 | -0.05066 | 0 / 1 / 0 |
| trocr-small-handwritten | encoder | FAIL | -0.17756 |  | -0.24210 | -0.15480 | 0 / 0 / 0 |
| webssl-dino1b-224 | encoder | FAIL | -0.08009 | T2:rprec;T3:rprec | -0.10143 | -0.06957 | 0 / 2 / 0 |
| webssl-dino2b-224 | encoder | FAIL | -0.03950 | T2:rprec;T3:rprec;T4:spearman | -0.05604 | -0.01515 | 0 / 4 / 0 |
| webssl-dino300m-224 | encoder | FAIL | -0.06549 | T2:rprec;T3:rprec | -0.07067 | -0.02354 | 0 / 4 / 0 |
| starvector-1b-im2svg | vector | FAIL | -0.18607 |  | -0.26538 | -0.07182 | 0 / 0 / 0 |
| starvector-8b-im2svg | vector | FAIL | -0.12811 | T2:rprec;T3:rprec | -0.15652 | -0.03513 | 0 / 3 / 0 |

## 2. Gates and cancellations [reason: A4.3(a) cancel rule]

- Gate-cancelled arms (CANCELLED_GATE_FAIL): 88 arms of 44 FMs; all 44 FMs have status FAIL; each cancellation is later than that FM's first FAIL verdict file.
- FAIL FMs without arm rows: 41. Non-FAIL FMs with a gate cancellation: 0. Control arms gate-cancelled: 0.
- Non-FAIL FMs without arm rows at the screen snapshot: sam3.1, webssl-dino3b-224, webssl-dino5b-224, webssl-dino7b-224, webssl-mae3b-224. arm_queue.csv rows for dinov3-vitl16-sat493m, webssl-mae1b-224 and chandra-ocr-2 were written at 2026-11-05T20:38:27-30Z (CLUSTER-A 24914901-06, submitted held).
- Details: GATE_READ.md.

## 3. Prereg amendments A3-A9 [reason: provenance of the rules applied]

Current file sha256 7a71c715dc5ca1cf146f829b8c7f498811e1330e1eefa2bf1b3b60fddbc5d154 contains A1-A9. Header of each FMDX amendment:

- A3 (2026-11-03T13:05:33Z): CROSS-SCRIPT SCREEN OF ALL CANDIDATE FMs (lane stage FMDX). This text was written before any FMDX screen output existed. At 13:00Z on 3 Nov, `exp/FMD/xs/{screen,features,...
- A4 (2026-11-03T13:05:33Z): QUEUED GATED FM-GUIDED URDU-LEFT-OUT BACKBONE ARMS (lane stage FMDX). Written before any FMDX arm was queued, started or sampled. Pre-amendment file: `FMD_PREREG_2026-11-...
- A5 (2026-11-03T13:08:47Z): Two clarifications; nothing else changes....
- A6 (2026-11-03T13:32:40Z): correction of the effective batch in A4.2. LUO ms6 trained with Trainer1D (repo 7c5cf623, Accelerator(split_batches=True)) at batch 128 global x grad-accum 4 = 512 sequen...
- A3.a (2026-11-03T14:56:43Z): ROSTER ADDITIONS FROM THE XS DISCOVERY LIST, AND LOADER STATUS OF SIX A3.3 MODELS. Written before any feature, probe, zero-shot output or REPA target of the models added ...
- A3.b (2026-11-03T15:09:00Z): ROSTER ADDITION facebook/sam3.1. Written before any feature, probe, zero-shot output or REPA target of sam3.1 existed. The append script checked `exp/FMD/xs/{features,scr...
- A3.c (2026-11-03T15:19:38Z): LONG-TAIL ROSTER ADDITIONS. Coverage: every id that the XS staging script `exp/FMD/xs/code/stage_xs_v3.py` stages or queues, plus every `exp/FMD/xs/weights/*.manifest_par...
- A7 (2026-11-03T16:41:32Z): A7 CLUSTER-A GPU cap for the FMD lane...
- A8 (2026-11-04T00:48:39Z): A8 Sampling device for FMDX reads...
- A9 (2026-11-05T19:56:15Z): A9 Sampling device, per job...

## 4. Infrastructure incidents

### VPS stage target overwrite (3 Nov) [reason: data integrity of REPA targets on the VPS lane]
- Screen-encoders jobs 24845238 (15:48-16:14Z) and 24846129 (16:14-16:23Z) plus a login push at ~16:23Z tar-streamed exp/FMD/xs/targets into the VPS stage and were killed mid-stream. Audit 24846202: all dinov3 files MATCH; the 440 dinov3-vitl16_W run recorded target sha bc161a92... at load, equal to CLUSTER-A; no restart.
- Source: runs/DECISION_RECORD_backbone_addendum_2.md; runs/HANDOFF_FMDX_monitor.md

### CLUSTER-A gpuidle preemption (5 Nov ~16:00Z) [reason: arm lane capacity]
- About 16:00Z all account GPU jobs in gpuidle were preempted; preemption-resume restarts the batch-sampler seed (seed + step) as in any requeue; layout unchanged (32x16). Partitions were widened (llms).
- Source: runs/DECISION_RECORD_backbone_addendum_7.md (H6)

### <node> failures (4 Nov 19:00-21:30Z) [reason: failed arms, requeue]
- 11 WARM jobs FAILED exit 7 on llms node <node> (RTX PRO 6000 Blackwell; lead: CUDA illegal memory access in 2-GPU DDP). PEINIT_W 24843710 FAILED exit 8 (VAL_SAMPLING_FAILED, <node>, 18:16Z) after training completed (step 40000); milestone 4 selected by sel_job 24877089. Requeue: 7 encoder arms to CLUSTER-B (3709973-79 then 3710045-51), 4 to CLUSTER-A (24904693-96, after 24904414-17 FAILED exit 4 GPU_CC_LT_8 on GTX 1070 nodes); controller v1.6 (auto-resubmit <= 2).
- Source: runs/DECISION_RECORD_backbone_addendum_8.md; admin/arm_queue.csv

### DiT REPA target OOM (4 Nov) [reason: blocked FLUX/SD3 arms]
- FLUX/SD3 target array 24866395: 33 FAILED, 3 COMPLETED. Cause in slurm-24866395_<task>.err: torch.OutOfMemoryError placing the transformer on 16 GB A4000s; fm_registry_dit.json had no DiT entries (kw={}, no vram_budget_gb streaming). Fix: registry entries micro=1, vram_budget_gb=10.5 (sha256 8f021a10aa1de64f...).
- Source: runs/DECISION_RECORD_backbone_addendum_8.md

### Warm monitor replaced after API failures (5 Nov) [reason: monitoring continuity]
- The <agent> warm monitor stopped on API failures and was replaced by an <agent> monitor (frame 44e9387d) about 12:30Z on 5 Nov; its tables jobs are 24900959 (interim) and 24914502.
- Source: lead session record (no CLUSTER-A file read for this item)

## 5. Finished warm arms, A4.5 cells [reason: A4.5 reads; EXPLORATORY (single training seed)]

Source table FMDX_warm_contrasts.csv (snapshot FMDX_warm_contrasts.csv mtime 2026-11-06T00:42Z, fetched 2026-11-07T15:00Z); all device types listed below; DEVICE_MIXED is False for every row read. U-WIN: AIoU CI lo > 0, pen-up F1 v2 > -0.0141, strict > -0.01065, direction > -0.01245 (test_ouhdl, FT vs pooled COMP-p01). X-NI: composed AIoU CI lo > -0.010 on all six non-Urdu test sets vs LUO ms6. FM-ATTRIBUTED per comparator: cell U AIoU > -0.01035 and the three order NI clauses AND at least one of the four CI lo > 0. Promotion: FM status PASS_XS AND U-WIN AND X-NI; controls never promoted.

| arm | FM status at read | U-WIN | X-NI | FM-ATTRIBUTED vs NOFM_W | FM-ATTRIBUTED vs PEINIT_W | promotion | milestones | devices | DEVICE_MIXED |
|---|---|---|---|---|---|---|---|---|---|
| NOFM_W | control | True | True | n/a (control) | n/a (control) | not promotable (control) | bb:3;cmp:1;ft:1 | RTX A4000 | False |
| PEINIT_W | control | True | True | n/a (control) | n/a (control) | not promotable (control) | bb:4;cmp:2;ft:2 | RTX A4000 | False |
| c-radiov3-h_W | PASS_XS | True | True | False | False | PROMOTED | bb:4;cmp:4;ft:4 | RTX A4000 | False |

### NOFM_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.04834 | 0.04623 | 0.05061 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.07476 | 0.06608 | 0.08305 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.02668 | 0.01882 | 0.03497 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | -0.00576 | -0.01058 | -0.00101 | -0.01245 | True | 4386 | 32 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.03794 | 0.03515 | 0.04092 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.00875 | 0.00683 | 0.01069 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.02056 | 0.01815 | 0.02355 | -0.01 | True | 500 | 52 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.01536 | 0.01299 | 0.01773 | -0.01 | True | 500 | 63 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.01945 | 0.01335 | 0.02499 | -0.01 | True | 827 | 7 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.02690 | 0.02615 | 0.02764 | -0.01 | True | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 |

### PEINIT_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.05952 | 0.05727 | 0.06196 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.11259 | 0.10251 | 0.12226 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.04885 | 0.04128 | 0.05702 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | -0.00223 | -0.00774 | 0.00274 | -0.01245 | True | 4389 | 32 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.05704 | 0.05399 | 0.06033 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.01306 | 0.01135 | 0.01483 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.02946 | 0.02687 | 0.03243 | -0.01 | True | 500 | 52 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.02364 | 0.02077 | 0.02685 | -0.01 | True | 500 | 63 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.02723 | 0.01632 | 0.03632 | -0.01 | True | 827 | 7 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.04570 | 0.04487 | 0.04653 | -0.01 | True | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 |

### c-radiov3-h_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.03411 | 0.03248 | 0.03577 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.06357 | 0.05773 | 0.06918 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.03311 | 0.02773 | 0.03923 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.00288 | -0.00141 | 0.00726 | -0.01245 | True | 4385 | 32 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.03596 | 0.03327 | 0.03884 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.00623 | 0.00424 | 0.00847 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.01815 | 0.01611 | 0.02087 | -0.01 | True | 500 | 52 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.01309 | 0.01076 | 0.01555 | -0.01 | True | 500 | 63 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.01048 | -0.00196 | 0.01928 | -0.01 | True | 827 | 7 (writer) | RTX A4000 / RTX A4000 |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.02664 | 0.02589 | 0.02738 | -0.01 | True | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U AIoU NI | NOFM_W | test_ouhdl | -0.01422 | -0.01597 | -0.01255 | -0.01035 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U penup_f1_v2 NI | NOFM_W | test_ouhdl | -0.01120 | -0.01800 | -0.00413 | -0.0141 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U strict_order_acc NI | NOFM_W | test_ouhdl | 0.00643 | -0.00038 | 0.01401 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U direction_acc NI | NOFM_W | test_ouhdl | 0.00873 | 0.00513 | 0.01242 | -0.01245 | True | 4397 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U aiou win | NOFM_W | test_ouhdl | -0.01422 | -0.01597 | -0.01255 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U penup_f1_v2 win | NOFM_W | test_ouhdl | -0.01120 | -0.01800 | -0.00413 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U strict_order_acc win | NOFM_W | test_ouhdl | 0.00643 | -0.00038 | 0.01401 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U direction_acc win | NOFM_W | test_ouhdl | 0.00873 | 0.00513 | 0.01242 | 0.0 | True | 4397 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_zh | -0.00197 | -0.00394 | 0.00008 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_en | -0.00252 | -0.00378 | -0.00132 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_hpl_tamil | -0.00242 | -0.00393 | -0.00090 | -0.01 | True | 500 | 52 (writer) | RTX A4000 / RTX A4000 |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_iam500 | -0.00228 | -0.00373 | -0.00079 | -0.01 | True | 500 | 63 (writer) | RTX A4000 / RTX A4000 |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_unipen_de_test | -0.00897 | -0.01559 | -0.00493 | -0.01 | False | 827 | 7 (writer) | RTX A4000 / RTX A4000 |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_onhw_all | -0.00026 | -0.00075 | 0.00028 | -0.01 | True | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U AIoU NI | PEINIT_W | test_ouhdl | -0.02540 | -0.02717 | -0.02354 | -0.01035 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U penup_f1_v2 NI | PEINIT_W | test_ouhdl | -0.04902 | -0.05702 | -0.04081 | -0.0141 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U strict_order_acc NI | PEINIT_W | test_ouhdl | -0.01574 | -0.02191 | -0.01003 | -0.01065 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U direction_acc NI | PEINIT_W | test_ouhdl | 0.00529 | 0.00078 | 0.00963 | -0.01245 | True | 4398 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U aiou win | PEINIT_W | test_ouhdl | -0.02540 | -0.02717 | -0.02354 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U penup_f1_v2 win | PEINIT_W | test_ouhdl | -0.04902 | -0.05702 | -0.04081 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U strict_order_acc win | PEINIT_W | test_ouhdl | -0.01574 | -0.02191 | -0.01003 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_FT_vs_control | F U direction_acc win | PEINIT_W | test_ouhdl | 0.00529 | 0.00078 | 0.00963 | 0.0 | True | 4398 | 32 (writer) | RTX A4000 / RTX A4000 |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_zh | -0.02107 | -0.02336 | -0.01869 | -0.01 | False | 500 | 60 (writer) | RTX A4000 / RTX A4000 |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_en | -0.00683 | -0.00807 | -0.00560 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_hpl_tamil | -0.01131 | -0.01280 | -0.00955 | -0.01 | False | 500 | 52 (writer) | RTX A4000 / RTX A4000 |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_iam500 | -0.01055 | -0.01260 | -0.00869 | -0.01 | False | 500 | 63 (writer) | RTX A4000 / RTX A4000 |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_unipen_de_test | -0.01675 | -0.02006 | -0.01523 | -0.01 | False | 827 | 7 (writer) | RTX A4000 / RTX A4000 |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_onhw_all | -0.01906 | -0.01963 | -0.01849 | -0.01 | False | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 |

## 6. Arm status list [reason: A4.3(f) deadline reporting] (captured 2026-11-07T15:01:45Z)

| arm | queue state | backbone last step | backbone complete | selected milestone | FT last step | FT complete | FT selected | CLUSTER-A job (state) | CLUSTER-B job (state) | read status |
|---|---|---|---|---|---|---|---|---|---|---|
| NOFM_W | DONE | 40000 | 1 | 3 | 10000 | 1 | 1 | 24843707 (COMPLETED) |  () | read (section 5) |
| PEINIT_W | DONE | 40000 | 1 | 4 | 10000 | 1 | 2 | 24843710 (FAILED) |  () | read (section 5) |
| dinov3-vitl16_W | SUBMITTED |  | 0 |  |  | 0 |  | 24843708 (CANCELLED+) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| cleandift-sd21_W | DONE | 40000 | 1 | 4 |  | 0 |  | 24904694 (COMPLETED) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone selected |
| dinov3-vits16_W | SUBMITTED | 6850 | 0 |  |  | 0 |  | 24843717 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone step 6850 |
| dinov3-vitb16_W | RUNNING | 19200 | 0 |  |  | 0 |  | 24904693 (RUNNING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone step 19200 |
| dinov3-vit7b16_W | SUBMITTED |  | 0 |  |  | 0 |  | 24843721 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| c-radiov3-b_W | RUNNING | 12700 | 0 |  |  | 0 |  | 24940531 (RUNNING) | 3710051 (CANCELLED+) | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone step 12700 |
| c-radiov3-l_W | RUNNING | 10450 | 0 |  |  | 0 |  | 24940529 (RUNNING) | 3710049 (CANCELLED+) | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone step 10450 |
| c-radiov3-h_W | DONE | 40000 | 1 | 4 | 10000 | 1 | 4 | 24843729 (COMPLETED) |  () | read (section 5) |
| c-radiov3-g_W | RUNNING | 10500 | 0 |  |  | 0 |  | 24940527 (RUNNING) | 3710047 (CANCELLED+) | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone step 10500 |
| c-radiov4-so400m_W | RUNNING | 12450 | 0 |  |  | 0 |  | 24940530 (RUNNING) | 3710050 (CANCELLED+) | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone step 12450 |
| c-radiov4-h_W | RUNNING | 10550 | 0 |  |  | 0 |  | 24940528 (RUNNING) | 3710048 (CANCELLED+) | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone step 10550 |
| pe-spatial-b16-512_W | DONE | 40000 | 1 | 4 |  | 0 |  | 24904695 (COMPLETED) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone selected |
| webssl-mae300m-224_W | RUNNING | 32350 | 0 |  |  | 0 |  | 24939113 (RUNNING) | 3710046 (CANCELLED+) | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone step 32350 |
| vit-mae-large_W | DONE | 40000 | 1 | 4 |  | 0 |  | 24939112 (COMPLETED) | 3710045 (CANCELLED+) | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone selected |
| sd3-medium-diffusers_W | SUBMITTED |  | 0 |  |  | 0 |  | 24957069 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| flux1-dev_W | SUBMITTED |  | 0 |  |  | 0 |  | 24866387 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| flux1-schnell_W | SUBMITTED |  | 0 |  |  | 0 |  | 24866388 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| qwen3.5-4b_W | SUBMITTED | 0 | 0 |  |  | 0 |  | 24904696 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| INKSIGHT_W | SUBMITTED |  | 0 |  |  | 0 |  | 24843782 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| inkfm_W | QUEUED |  | 0 |  |  | 0 |  |  () |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| NOFM_S | PINNED_CLUSTER-B |  | 0 |  |  | 0 |  | 24843785 (CANCELLED+) | 3710052 (PENDING) | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| PEINIT_S | PINNED_CLUSTER-B |  | 0 |  |  | 0 |  | 24843790 (CANCELLED+) | 3710053 (PENDING) | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| dinov3-vitl16_S | SUBMITTED | 2600 | 0 |  |  | 0 |  | 24843787 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone step 2600 |
| cleandift-sd21_S | SUBMITTED | 5850 | 0 |  |  | 0 |  | 24843789 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone step 5850 |
| dinov3-vits16_S | SUBMITTED |  | 0 |  |  | 0 |  | 24843797 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| dinov3-vitb16_S | SUBMITTED |  | 0 |  |  | 0 |  | 24843799 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| dinov3-vit7b16_S | SUBMITTED |  | 0 |  |  | 0 |  | 24843801 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| c-radiov3-b_S | PINNED_CLUSTER-B |  | 0 |  |  | 0 |  | 24843807 (CANCELLED+) | 3710061 (PENDING) | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| c-radiov3-l_S | PINNED_CLUSTER-B |  | 0 |  |  | 0 |  | 24843808 (CANCELLED+) | 3710059 (PENDING) | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| c-radiov3-h_S | PINNED_CLUSTER-B |  | 0 |  |  | 0 |  | 24843809 (CANCELLED+) | 3710056 (PENDING) | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| c-radiov3-g_S | PINNED_CLUSTER-B |  | 0 |  |  | 0 |  | 24843810 (CANCELLED+) | 3710057 (PENDING) | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| c-radiov4-so400m_S | PINNED_CLUSTER-B |  | 0 |  |  | 0 |  | 24843811 (CANCELLED+) | 3710060 (PENDING) | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| c-radiov4-h_S | PINNED_CLUSTER-B |  | 0 |  |  | 0 |  | 24843812 (CANCELLED+) | 3710058 (PENDING) | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| pe-spatial-b16-512_S | SUBMITTED |  | 0 |  |  | 0 |  | 24843815 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| webssl-mae300m-224_S | PINNED_CLUSTER-B |  | 0 |  |  | 0 |  | 24843825 (CANCELLED+) | 3710055 (PENDING) | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| vit-mae-large_S | PINNED_CLUSTER-B |  | 0 |  |  | 0 |  | 24843826 (CANCELLED+) | 3710054 (PENDING) | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| sd3-medium-diffusers_S | SUBMITTED |  | 0 |  |  | 0 |  | 24866389 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| flux1-dev_S | SUBMITTED |  | 0 |  |  | 0 |  | 24866390 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| flux1-schnell_S | SUBMITTED |  | 0 |  |  | 0 |  | 24866391 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| qwen3.5-4b_S | SUBMITTED |  | 0 |  |  | 0 |  | 24843836 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| INKSIGHT_S | SUBMITTED |  | 0 |  |  | 0 |  | 24843847 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| inkfm_S | QUEUED |  | 0 |  |  | 0 |  |  () |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| sd3.5-large-diffusers-layout_W | SUBMITTED |  | 0 |  |  | 0 |  | 24878945 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| sd3.5-large-turbo-diffusers-layout_W | SUBMITTED |  | 0 |  |  | 0 |  | 24878946 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| sd3.5-medium-diffusers-layout_W | SUBMITTED |  | 0 |  |  | 0 |  | 24878944 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| sd3.5-large-diffusers-layout_S | SUBMITTED |  | 0 |  |  | 0 |  | 24878948 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| sd3.5-large-turbo-diffusers-layout_S | SUBMITTED |  | 0 |  |  | 0 |  | 24878949 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| sd3.5-medium-diffusers-layout_S | SUBMITTED |  | 0 |  |  | 0 |  | 24878947 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| webssl-mae1b-224_W | RUNNING | 30850 | 0 |  |  | 0 |  | 24914901 (RUNNING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone step 30850 |
| webssl-mae1b-224_S | SUBMITTED |  | 0 |  |  | 0 |  | 24914904 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| dinov3-vitl16-sat493m_W | DONE | 40000 | 1 | 4 |  | 0 |  | 24914902 (COMPLETED) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: backbone selected |
| dinov3-vitl16-sat493m_S | SUBMITTED |  | 0 |  |  | 0 |  | 24914905 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| chandra-ocr-2_W | SUBMITTED |  | 0 |  |  | 0 |  | 24914903 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |
| chandra-ocr-2_S | SUBMITTED |  | 0 |  |  | 0 |  | 24914906 (PENDING) |  () | not assessed by 2026-11-07T15:01:45Z; last completed stage: not started |

## 7. Blocked or failed attempts and items not assessed

### inkfm screen and arms [reason: no public weights]
- verdicts/blocked/blocked_inkfm.json; status PENDING; arms inkfm_W/S QUEUED, never dispatched.

### sam3.1 screen [reason: no extractor before 2026-11-05T12:00Z (A3.a.2/A3.b)]
- verdict status PENDING, status_reason "not assessed: no extractor"; no arm rows.

### StarVector SVG zero-shot rows [reason: starcoderbase gated at first attempt; starvector-8b route moved to CLUSTER-B]
- verdicts/blocked/blocked_starvector_svg_zeroshot.json. starvector-1b/8b statuses come from vision-probe rows (both FAIL: SHAPE failed on all 5 cross-script sets).

### PEINIT_W selection sampling [reason: VAL_SAMPLING_FAILED exit 8 on <node>]
- training complete; milestone 4 selected by monitor sel_job 24877089. PEINIT_W FT and composed rows are present in the warm table; all its cells are read in section 5.

### Screen cut-off ids [reason: A3.c.3 cut-off]
- ids without complete gate rows on all 6 sets at the cut-off: webssl-dino3b-224, webssl-dino5b-224, webssl-dino7b-224, webssl-mae3b-224; status PENDING, partial rows enter no verdict; no arms.

### Probe attentive / info rows [reason: screen tracks finished linear gate rows first]
- info rows do not enter status (Section 3).

### priority_score, m_ord, m_t0 [reason: not recomputed by this read]
- they order the queue and do not set status.

### Not assessed in this version

- All arms in section 6 marked "not assessed by 2026-11-07T15:01:45Z".
