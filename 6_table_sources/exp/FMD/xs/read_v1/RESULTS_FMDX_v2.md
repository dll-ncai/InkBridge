# RESULTS_FMDX v2

Screen read: **FINAL (cut-off 2026-11-06T23:59:59Z; snapshot 2026-11-07T00:47Z; unchanged since v1.1)**. Warm tables snapshot: FMDX_warm_contrasts.csv from CLUSTER-A table job 25049213 (warm monitor HANDOFF 2026-11-10T17:13Z; 5352 rows). Arm status captured: see section 6. Prereg exp/FMD/plans/FMD_PREREG_2026-11-03.md, sha256 be563cdfa038d76a216e14a0a251a94d1a894c25bf38b5b58f159b993d5d37ab. The screen gate read was computed under sha256 7a71c715...; the later amendment A9.4 is operational only (sampling device). Written by the FMDX XS gate read track, <agent>. All backbone and FT results are EXPLORATORY (single training seed). Facts only.

Artifacts: [GATE_READ.md]({{artifact:8422a091-0250-424e-97d5-9584609ba1bb}}), [fmdx_screen_table.csv]({{artifact:bcefeaef-2be6-457f-a049-b883c47dbe4e}}), [fmdx_gate_read.csv]({{artifact:4531786f-7d6c-4d25-8bab-f337fd87f737}}), [fmdx_gate_read_sets.csv]({{artifact:8fa95c9b-420d-4611-a64c-18972d05a1be}}), [fmdx_screen_heatmap.png]({{artifact:b28900d4-2f6b-449c-ac75-66e6758f3c59}}), [fmdx_warm_read.csv]({{artifact:46bb3681-f47d-4590-a375-aa2c85dc49ef}}), [fmdx_warm_read_clauses.csv]({{artifact:23280023-2300-4588-9f39-9680a39118c1}}), [fmdx_warm_forest.png]({{artifact:16908625-f9e0-45f3-a37f-650c7007bca3}}), [FMDX_warm_contrasts.csv (warm monitor, job 25049213)]({{artifact:aae330d5-c86f-4185-b720-0ff9de3ff6f8}}), [FMDX_not_assessed_A43f.csv]({{artifact:66730411-655e-43fd-b73f-cf406f5add7e}}).

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

Current file sha256 be563cdfa038d76a216e14a0a251a94d1a894c25bf38b5b58f159b993d5d37ab contains A1-A9.4. Header of each FMDX amendment:

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
- A9.4 (2026-11-08T08:48:42Z): the A9.1 'first CLUSTER-A GPU type' becomes 'the first GPU type on CLUSTER-A or CLUSTER-B'. CLUSTER-B-sampled preds are copied to CLUSTER-A with a sha256 check and scored on CLUSTER-A with the same sc...

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

### Stale CLUSTER-B controller cancelled and held CLUSTER-B arms (6 Nov) [reason: lost CLUSTER-B arm copies]
- CLUSTER-B controller v1.1 job 3703956 (submitted 2026-11-04T14:04:32 CET, before the VPS shutdown, never cancelled) started 2026-11-06 14:57:40 CET. Its race step read stale owner data and scancelled 3710045 (vit-mae-large_W) and 3710046 (webssl-mae300m-224_W) as 'race lost'; both CANCELLED 2026-11-06T13:58:13Z, never started. Its own CLUSTER-B cap of 8 GPUs held 3710051 (c-radiov3-b_W) and the 10 CLUSTER-B SCRATCH segments 3710052-61. Action: 3703956 and continuation 3716777 cancelled; the 11 held jobs released; no CLUSTER-B controller runs any more.
- Source: runs/DECISION_RECORD_backbone_addendum_9.md.b64 (K1)

### CLUSTER-A SSH outage (8 Nov) [reason: lost access for transfers]
- CLUSTER-A SSH stopped answering 10:40-11:41Z and again from ~12:20Z on 8 Nov, during agent sftp transfers of 50-200 MB; downloads of a few hundred kB work.
- Source: runs/DECISION_RECORD_backbone_addendum_10.md; runs/HANDOFF_FMDX_monitor (10 Nov) section 5

### A9.4 checkpoint relay CLUSTER-A -> CLUSTER-B failed (8 Nov) [reason: CLUSTER-B sampling not possible]
- Checkpoint relay (~389 MB each) by agent download from CLUSTER-A and upload to CLUSTER-B: 200 MB parts (08:55Z), 100 MB parts (09:31Z) and 50 MB parts (11:42Z) all failed on CLUSTER-A sftp get ('Connection closed'). Nothing relayed; no CLUSTER-B sampling job submitted. CLUSTER-B SCRATCH arms all PENDING at 15:3xZ 10 Nov; no RELAY.json under CLUSTER-B runs/.
- Source: runs/DECISION_RECORD_backbone_addendum_10.md; HANDOFF section 5

### A9 race-cancel bug killed running sampling tasks (8 Nov) [reason: lost sampling seeds]
- The A9 race-cancel used squeue %A (array id) and scancelled whole arrays including running tasks: cleandift-sd21_W bb seed0 (24958052_0) and webssl-mae1b-224_W bb seed0 (24969669_0) were killed; stale locks removed; seed0 resubmitted as A9 copies (logged 2026-11-08T05:41:32Z). The 10 Nov autopilot cancels pending tasks only.
- Source: exp/CAND/queue/cand_log.md; runs/fmdx_monitor/ACTIONS_LOG.md

## 5. Finished warm arms, A4.5 cells [reason: A4.5 reads; EXPLORATORY (single training seed)]

Source table FMDX_warm_contrasts.csv (snapshot FMDX_warm_contrasts.csv from CLUSTER-A table job 25049213 (warm monitor HANDOFF 2026-11-10T17:13Z; 5352 rows)); device types per arm are listed below; 188 of 382 clause rows are DEVICE_MIXED (A9.2: the two sides were sampled on different GPU types; DDIM sampling is not bitwise reproducible across GPU types). U-WIN: AIoU CI lo > 0, pen-up F1 v2 > -0.0141, strict > -0.01065, direction > -0.01245 (test_ouhdl, FT vs pooled COMP-p01). X-NI: composed AIoU CI lo > -0.010 on all six non-Urdu test sets vs LUO ms6. FM-ATTRIBUTED per comparator: cell U AIoU > -0.01035 and the three order NI clauses AND at least one of the four CI lo > 0. Promotion: FM status PASS_XS AND U-WIN AND X-NI; controls never promoted.

| arm | FM status at read | U-WIN | X-NI | FM-ATTRIBUTED vs NOFM_W | FM-ATTRIBUTED vs PEINIT_W | promotion | milestones | devices | DEVICE_MIXED clauses |
|---|---|---|---|---|---|---|---|---|---|
| NOFM_W | control | True | True | n/a (control) | n/a (control) | not promotable (control) | bb:3;cmp:1;ft:1 | RTX A4000 | 0 of 10 |
| PEINIT_W | control | True | True | n/a (control) | n/a (control) | not promotable (control) | bb:4;cmp:2;ft:2 | RTX A4000 | 0 of 10 |
| c-radiov3-b_W | PASS_XS | True | True | False | False | PROMOTED | bb:4;cmp:3;ft:3 | A30;A30|RTX A4000;RTX A4000 | 18 of 38 |
| c-radiov3-h_W | PASS_XS | True | True | False | False | PROMOTED | bb:4;cmp:4;ft:4 | RTX A4000 | 0 of 38 |
| c-radiov4-so400m_W | PASS_XS | True | True | True | False | PROMOTED | bb:4;cmp:3;ft:3 | A30;A30|RTX A4000;RTX A4000 | 18 of 38 |
| cleandift-sd21_W | PASS_URDU_ONLY | True | True | True | False | not promotable (PASS_URDU_ONLY) | bb:4;cmp:3;ft:3 | B200|RTX A4000;RTX A4000 | 18 of 38 |
| dinov3-vitb16_W | PASS_XS | True | True | False | False | PROMOTED | bb:4;cmp:2;ft:2 | RTX A4000 | 0 of 38 |
| dinov3-vitl16-sat493m_W | PASS_XS | True | True | False | False | PROMOTED | bb:4;cmp:5;ft:5 | B200;B200|RTX A4000;RTX A4000 | 38 of 38 |
| pe-spatial-b16-512_W | PASS_URDU_ONLY | True | True | True | False | not promotable (PASS_URDU_ONLY) | bb:4;cmp:2;ft:2 | B200;B200|RTX A4000;RTX A4000 | 38 of 38 |
| vit-mae-large_W | PASS_XS | True | True | True | False | PROMOTED | bb:4;cmp:5;ft:5 | B200;B200|RTX A4000;RTX A4000 | 38 of 38 |
| webssl-mae1b-224_W | PASS_XS | True | not assessed | True | False | not assessed (cells incomplete) | ft:3 | A30;RTX A4000 | 20 of 20 |
| webssl-mae300m-224_W | PASS_XS | True | True | True | False | PROMOTED | bb:4;cmp:2;ft:2 | RTX A4000 | 0 of 38 |

Promoted-arm order (A4.5: larger min over the three cell-U order NI CI lo values, then larger cell-U AIoU CI lo):

| rank | arm | min order NI CI lo (cell U) | AIoU CI lo (cell U) | DEVICE_MIXED clauses |
|---|---|---|---|---|
| 1 | c-radiov3-h_W | -0.00141 | 0.03248 | 0 |
| 2 | c-radiov4-so400m_W | -0.00185 | 0.04674 | 18 |
| 3 | c-radiov3-b_W | -0.00205 | 0.03747 | 18 |
| 4 | vit-mae-large_W | -0.00337 | 0.04503 | 38 |
| 5 | dinov3-vitb16_W | -0.00419 | 0.03658 | 0 |
| 6 | dinov3-vitl16-sat493m_W | -0.00468 | 0.03771 | 38 |
| 7 | webssl-mae300m-224_W | -0.00793 | 0.05231 | 0 |

### NOFM_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices | DEVICE_MIXED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.04834 | 0.04623 | 0.05061 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.07476 | 0.06608 | 0.08305 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.02668 | 0.01882 | 0.03497 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | -0.00576 | -0.01058 | -0.00101 | -0.01245 | True | 4386 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.03794 | 0.03515 | 0.04092 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.00875 | 0.00683 | 0.01069 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.02056 | 0.01815 | 0.02355 | -0.01 | True | 500 | 52 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.01536 | 0.01299 | 0.01773 | -0.01 | True | 500 | 63 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.01945 | 0.01335 | 0.02499 | -0.01 | True | 827 | 7 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.02690 | 0.02615 | 0.02764 | -0.01 | True | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 | False |

### PEINIT_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices | DEVICE_MIXED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.05952 | 0.05727 | 0.06196 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.11259 | 0.10251 | 0.12226 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.04885 | 0.04128 | 0.05702 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | -0.00223 | -0.00774 | 0.00274 | -0.01245 | True | 4389 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.05704 | 0.05399 | 0.06033 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.01306 | 0.01135 | 0.01483 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.02946 | 0.02687 | 0.03243 | -0.01 | True | 500 | 52 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.02364 | 0.02077 | 0.02685 | -0.01 | True | 500 | 63 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.02723 | 0.01632 | 0.03632 | -0.01 | True | 827 | 7 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.04570 | 0.04487 | 0.04653 | -0.01 | True | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 | False |

### c-radiov3-b_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices | DEVICE_MIXED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.03897 | 0.03747 | 0.04069 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.06875 | 0.06281 | 0.07432 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.03084 | 0.02351 | 0.03806 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.00172 | -0.00205 | 0.00546 | -0.01245 | True | 4385 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.03434 | 0.03116 | 0.03778 | -0.01 | True | 500 | 60 (writer) | A30|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.00654 | 0.00472 | 0.00831 | -0.01 | True | 500 | 60 (writer) | A30|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.01759 | 0.01531 | 0.02026 | -0.01 | True | 500 | 52 (writer) | A30|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.01236 | 0.00970 | 0.01510 | -0.01 | True | 500 | 63 (writer) | A30|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.01198 | 0.00141 | 0.01989 | -0.01 | True | 827 | 7 (writer) | A30|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.02381 | 0.02304 | 0.02455 | -0.01 | True | 5925 | 5925 (unit) | A30|RTX A4000 / RTX A4000 | True |
| F_FT_vs_control | F U AIoU NI | NOFM_W | test_ouhdl | -0.00937 | -0.01106 | -0.00767 | -0.01035 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 NI | NOFM_W | test_ouhdl | -0.00601 | -0.01378 | 0.00142 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc NI | NOFM_W | test_ouhdl | 0.00416 | -0.00293 | 0.01204 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc NI | NOFM_W | test_ouhdl | 0.00775 | 0.00341 | 0.01224 | -0.01245 | True | 4395 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U aiou win | NOFM_W | test_ouhdl | -0.00937 | -0.01106 | -0.00767 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 win | NOFM_W | test_ouhdl | -0.00601 | -0.01378 | 0.00142 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc win | NOFM_W | test_ouhdl | 0.00416 | -0.00293 | 0.01204 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc win | NOFM_W | test_ouhdl | 0.00775 | 0.00341 | 0.01224 | 0.0 | True | 4395 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_zh | -0.00359 | -0.00598 | -0.00108 | -0.01 | True | 500 | 60 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_en | -0.00221 | -0.00387 | -0.00055 | -0.01 | True | 500 | 60 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_hpl_tamil | -0.00297 | -0.00464 | -0.00138 | -0.01 | True | 500 | 52 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_iam500 | -0.00300 | -0.00495 | -0.00095 | -0.01 | True | 500 | 63 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_unipen_de_test | -0.00747 | -0.01233 | -0.00437 | -0.01 | False | 827 | 7 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_onhw_all | -0.00309 | -0.00376 | -0.00245 | -0.01 | True | 5925 | 5925 (unit) | A30|RTX A4000 / RTX A4000 | True |
| F_FT_vs_control | F U AIoU NI | PEINIT_W | test_ouhdl | -0.02055 | -0.02221 | -0.01875 | -0.01035 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 NI | PEINIT_W | test_ouhdl | -0.04384 | -0.05206 | -0.03557 | -0.0141 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc NI | PEINIT_W | test_ouhdl | -0.01801 | -0.02579 | -0.01046 | -0.01065 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc NI | PEINIT_W | test_ouhdl | 0.00425 | 0.00075 | 0.00805 | -0.01245 | True | 4397 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U aiou win | PEINIT_W | test_ouhdl | -0.02055 | -0.02221 | -0.01875 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 win | PEINIT_W | test_ouhdl | -0.04384 | -0.05206 | -0.03557 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc win | PEINIT_W | test_ouhdl | -0.01801 | -0.02579 | -0.01046 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc win | PEINIT_W | test_ouhdl | 0.00425 | 0.00075 | 0.00805 | 0.0 | True | 4397 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_zh | -0.02269 | -0.02530 | -0.02012 | -0.01 | False | 500 | 60 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_en | -0.00652 | -0.00828 | -0.00480 | -0.01 | True | 500 | 60 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_hpl_tamil | -0.01187 | -0.01371 | -0.01011 | -0.01 | False | 500 | 52 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_iam500 | -0.01128 | -0.01356 | -0.00913 | -0.01 | False | 500 | 63 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_unipen_de_test | -0.01524 | -0.01746 | -0.01375 | -0.01 | False | 827 | 7 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_onhw_all | -0.02189 | -0.02257 | -0.02119 | -0.01 | False | 5925 | 5925 (unit) | A30|RTX A4000 / RTX A4000 | True |

### c-radiov3-h_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices | DEVICE_MIXED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.03411 | 0.03248 | 0.03577 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.06357 | 0.05773 | 0.06918 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.03311 | 0.02773 | 0.03923 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.00288 | -0.00141 | 0.00726 | -0.01245 | True | 4385 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.03596 | 0.03327 | 0.03884 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.00623 | 0.00424 | 0.00847 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.01815 | 0.01611 | 0.02087 | -0.01 | True | 500 | 52 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.01309 | 0.01076 | 0.01555 | -0.01 | True | 500 | 63 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.01048 | -0.00196 | 0.01928 | -0.01 | True | 827 | 7 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.02664 | 0.02589 | 0.02738 | -0.01 | True | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U AIoU NI | NOFM_W | test_ouhdl | -0.01422 | -0.01597 | -0.01255 | -0.01035 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 NI | NOFM_W | test_ouhdl | -0.01120 | -0.01800 | -0.00413 | -0.0141 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc NI | NOFM_W | test_ouhdl | 0.00643 | -0.00038 | 0.01401 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc NI | NOFM_W | test_ouhdl | 0.00873 | 0.00513 | 0.01242 | -0.01245 | True | 4397 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U aiou win | NOFM_W | test_ouhdl | -0.01422 | -0.01597 | -0.01255 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 win | NOFM_W | test_ouhdl | -0.01120 | -0.01800 | -0.00413 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc win | NOFM_W | test_ouhdl | 0.00643 | -0.00038 | 0.01401 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc win | NOFM_W | test_ouhdl | 0.00873 | 0.00513 | 0.01242 | 0.0 | True | 4397 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_zh | -0.00197 | -0.00394 | 0.00008 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_en | -0.00252 | -0.00378 | -0.00132 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_hpl_tamil | -0.00242 | -0.00393 | -0.00090 | -0.01 | True | 500 | 52 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_iam500 | -0.00228 | -0.00373 | -0.00079 | -0.01 | True | 500 | 63 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_unipen_de_test | -0.00897 | -0.01559 | -0.00493 | -0.01 | False | 827 | 7 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_onhw_all | -0.00026 | -0.00075 | 0.00028 | -0.01 | True | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U AIoU NI | PEINIT_W | test_ouhdl | -0.02540 | -0.02717 | -0.02354 | -0.01035 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 NI | PEINIT_W | test_ouhdl | -0.04902 | -0.05702 | -0.04081 | -0.0141 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc NI | PEINIT_W | test_ouhdl | -0.01574 | -0.02191 | -0.01003 | -0.01065 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc NI | PEINIT_W | test_ouhdl | 0.00529 | 0.00078 | 0.00963 | -0.01245 | True | 4398 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U aiou win | PEINIT_W | test_ouhdl | -0.02540 | -0.02717 | -0.02354 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 win | PEINIT_W | test_ouhdl | -0.04902 | -0.05702 | -0.04081 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc win | PEINIT_W | test_ouhdl | -0.01574 | -0.02191 | -0.01003 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc win | PEINIT_W | test_ouhdl | 0.00529 | 0.00078 | 0.00963 | 0.0 | True | 4398 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_zh | -0.02107 | -0.02336 | -0.01869 | -0.01 | False | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_en | -0.00683 | -0.00807 | -0.00560 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_hpl_tamil | -0.01131 | -0.01280 | -0.00955 | -0.01 | False | 500 | 52 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_iam500 | -0.01055 | -0.01260 | -0.00869 | -0.01 | False | 500 | 63 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_unipen_de_test | -0.01675 | -0.02006 | -0.01523 | -0.01 | False | 827 | 7 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_onhw_all | -0.01906 | -0.01963 | -0.01849 | -0.01 | False | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 | False |

### c-radiov4-so400m_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices | DEVICE_MIXED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.04847 | 0.04674 | 0.05028 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.08364 | 0.07661 | 0.09057 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.04030 | 0.03382 | 0.04783 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.00327 | -0.00185 | 0.00809 | -0.01245 | True | 4388 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.04471 | 0.04155 | 0.04799 | -0.01 | True | 500 | 60 (writer) | A30|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.00984 | 0.00790 | 0.01188 | -0.01 | True | 500 | 60 (writer) | A30|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.02353 | 0.02112 | 0.02637 | -0.01 | True | 500 | 52 (writer) | A30|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.01664 | 0.01394 | 0.01945 | -0.01 | True | 500 | 63 (writer) | A30|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.01611 | 0.00465 | 0.02434 | -0.01 | True | 827 | 7 (writer) | A30|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.03190 | 0.03114 | 0.03268 | -0.01 | True | 5925 | 5925 (unit) | A30|RTX A4000 / RTX A4000 | True |
| F_FT_vs_control | F U AIoU NI | NOFM_W | test_ouhdl | 0.00013 | -0.00117 | 0.00152 | -0.01035 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 NI | NOFM_W | test_ouhdl | 0.00888 | 0.00371 | 0.01449 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc NI | NOFM_W | test_ouhdl | 0.01362 | 0.00684 | 0.02081 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc NI | NOFM_W | test_ouhdl | 0.00905 | 0.00401 | 0.01389 | -0.01245 | True | 4398 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U aiou win | NOFM_W | test_ouhdl | 0.00013 | -0.00117 | 0.00152 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 win | NOFM_W | test_ouhdl | 0.00888 | 0.00371 | 0.01449 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc win | NOFM_W | test_ouhdl | 0.01362 | 0.00684 | 0.02081 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc win | NOFM_W | test_ouhdl | 0.00905 | 0.00401 | 0.01389 | 0.0 | True | 4398 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_zh | 0.00678 | 0.00465 | 0.00865 | -0.01 | True | 500 | 60 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_en | 0.00109 | -0.00036 | 0.00245 | -0.01 | True | 500 | 60 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_hpl_tamil | 0.00297 | 0.00125 | 0.00471 | -0.01 | True | 500 | 52 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_iam500 | 0.00128 | -0.00045 | 0.00313 | -0.01 | True | 500 | 63 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_unipen_de_test | -0.00334 | -0.00914 | 0.00045 | -0.01 | True | 827 | 7 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_onhw_all | 0.00500 | 0.00434 | 0.00565 | -0.01 | True | 5925 | 5925 (unit) | A30|RTX A4000 / RTX A4000 | True |
| F_FT_vs_control | F U AIoU NI | PEINIT_W | test_ouhdl | -0.01105 | -0.01268 | -0.00925 | -0.01035 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 NI | PEINIT_W | test_ouhdl | -0.02895 | -0.03467 | -0.02310 | -0.0141 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc NI | PEINIT_W | test_ouhdl | -0.00855 | -0.01419 | -0.00275 | -0.01065 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc NI | PEINIT_W | test_ouhdl | 0.00572 | 0.00156 | 0.00992 | -0.01245 | True | 4400 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U aiou win | PEINIT_W | test_ouhdl | -0.01105 | -0.01268 | -0.00925 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 win | PEINIT_W | test_ouhdl | -0.02895 | -0.03467 | -0.02310 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc win | PEINIT_W | test_ouhdl | -0.00855 | -0.01419 | -0.00275 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc win | PEINIT_W | test_ouhdl | 0.00572 | 0.00156 | 0.00992 | 0.0 | True | 4400 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_zh | -0.01232 | -0.01469 | -0.01007 | -0.01 | False | 500 | 60 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_en | -0.00322 | -0.00498 | -0.00156 | -0.01 | True | 500 | 60 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_hpl_tamil | -0.00593 | -0.00771 | -0.00414 | -0.01 | True | 500 | 52 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_iam500 | -0.00700 | -0.00918 | -0.00488 | -0.01 | True | 500 | 63 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_unipen_de_test | -0.01112 | -0.01491 | -0.00924 | -0.01 | False | 827 | 7 (writer) | A30|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_onhw_all | -0.01380 | -0.01445 | -0.01314 | -0.01 | False | 5925 | 5925 (unit) | A30|RTX A4000 / RTX A4000 | True |

### cleandift-sd21_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices | DEVICE_MIXED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.04942 | 0.04748 | 0.05140 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.08430 | 0.07611 | 0.09204 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.03561 | 0.02788 | 0.04442 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | -0.00091 | -0.00582 | 0.00360 | -0.01245 | True | 4387 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.04789 | 0.04488 | 0.05107 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.01111 | 0.00904 | 0.01318 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.02595 | 0.02368 | 0.02887 | -0.01 | True | 500 | 52 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.01965 | 0.01717 | 0.02231 | -0.01 | True | 500 | 63 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.02009 | 0.00942 | 0.02835 | -0.01 | True | 827 | 7 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.03809 | 0.03727 | 0.03885 | -0.01 | True | 5925 | 5925 (unit) | B200|RTX A4000 / RTX A4000 | True |
| F_FT_vs_control | F U AIoU NI | NOFM_W | test_ouhdl | 0.00108 | -0.00014 | 0.00234 | -0.01035 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 NI | NOFM_W | test_ouhdl | 0.00953 | 0.00509 | 0.01406 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc NI | NOFM_W | test_ouhdl | 0.00893 | 0.00452 | 0.01361 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc NI | NOFM_W | test_ouhdl | 0.00474 | 0.00023 | 0.00921 | -0.01245 | True | 4399 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U aiou win | NOFM_W | test_ouhdl | 0.00108 | -0.00014 | 0.00234 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 win | NOFM_W | test_ouhdl | 0.00953 | 0.00509 | 0.01406 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc win | NOFM_W | test_ouhdl | 0.00893 | 0.00452 | 0.01361 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc win | NOFM_W | test_ouhdl | 0.00474 | 0.00023 | 0.00921 | 0.0 | True | 4399 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_zh | 0.00996 | 0.00751 | 0.01228 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_en | 0.00236 | 0.00092 | 0.00370 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_hpl_tamil | 0.00538 | 0.00426 | 0.00659 | -0.01 | True | 500 | 52 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_iam500 | 0.00429 | 0.00293 | 0.00572 | -0.01 | True | 500 | 63 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_unipen_de_test | 0.00064 | -0.00419 | 0.00395 | -0.01 | True | 827 | 7 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_onhw_all | 0.01118 | 0.01061 | 0.01172 | -0.01 | True | 5925 | 5925 (unit) | B200|RTX A4000 / RTX A4000 | True |
| F_FT_vs_control | F U AIoU NI | PEINIT_W | test_ouhdl | -0.01010 | -0.01153 | -0.00870 | -0.01035 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 NI | PEINIT_W | test_ouhdl | -0.02829 | -0.03293 | -0.02397 | -0.0141 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc NI | PEINIT_W | test_ouhdl | -0.01324 | -0.01954 | -0.00732 | -0.01065 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc NI | PEINIT_W | test_ouhdl | 0.00137 | -0.00279 | 0.00542 | -0.01245 | True | 4400 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U aiou win | PEINIT_W | test_ouhdl | -0.01010 | -0.01153 | -0.00870 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 win | PEINIT_W | test_ouhdl | -0.02829 | -0.03293 | -0.02397 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc win | PEINIT_W | test_ouhdl | -0.01324 | -0.01954 | -0.00732 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc win | PEINIT_W | test_ouhdl | 0.00137 | -0.00279 | 0.00542 | 0.0 | False | 4400 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_zh | -0.00914 | -0.01170 | -0.00668 | -0.01 | False | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_en | -0.00195 | -0.00347 | -0.00040 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_hpl_tamil | -0.00351 | -0.00482 | -0.00201 | -0.01 | True | 500 | 52 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_iam500 | -0.00399 | -0.00544 | -0.00255 | -0.01 | True | 500 | 63 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_unipen_de_test | -0.00713 | -0.00890 | -0.00515 | -0.01 | True | 827 | 7 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_onhw_all | -0.00762 | -0.00819 | -0.00706 | -0.01 | True | 5925 | 5925 (unit) | B200|RTX A4000 / RTX A4000 | True |

### dinov3-vitb16_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices | DEVICE_MIXED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.03845 | 0.03658 | 0.04048 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.05006 | 0.04213 | 0.05726 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.02388 | 0.01663 | 0.03142 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | -0.00012 | -0.00419 | 0.00380 | -0.01245 | True | 4384 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.02981 | 0.02693 | 0.03299 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.00477 | 0.00303 | 0.00662 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.01341 | 0.01093 | 0.01632 | -0.01 | True | 500 | 52 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.00955 | 0.00707 | 0.01216 | -0.01 | True | 500 | 63 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.00987 | 0.00182 | 0.01586 | -0.01 | True | 827 | 7 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.02027 | 0.01954 | 0.02100 | -0.01 | True | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U AIoU NI | NOFM_W | test_ouhdl | -0.00989 | -0.01123 | -0.00842 | -0.01035 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 NI | NOFM_W | test_ouhdl | -0.02470 | -0.03222 | -0.01741 | -0.0141 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc NI | NOFM_W | test_ouhdl | -0.00280 | -0.00834 | 0.00332 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc NI | NOFM_W | test_ouhdl | 0.00580 | 0.00238 | 0.00958 | -0.01245 | True | 4395 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U aiou win | NOFM_W | test_ouhdl | -0.00989 | -0.01123 | -0.00842 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 win | NOFM_W | test_ouhdl | -0.02470 | -0.03222 | -0.01741 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc win | NOFM_W | test_ouhdl | -0.00280 | -0.00834 | 0.00332 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc win | NOFM_W | test_ouhdl | 0.00580 | 0.00238 | 0.00958 | 0.0 | True | 4395 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_zh | -0.00813 | -0.00993 | -0.00624 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_en | -0.00398 | -0.00525 | -0.00262 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_hpl_tamil | -0.00716 | -0.00874 | -0.00554 | -0.01 | True | 500 | 52 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_iam500 | -0.00581 | -0.00740 | -0.00403 | -0.01 | True | 500 | 63 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_unipen_de_test | -0.00958 | -0.01195 | -0.00815 | -0.01 | False | 827 | 7 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_onhw_all | -0.00664 | -0.00716 | -0.00610 | -0.01 | True | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U AIoU NI | PEINIT_W | test_ouhdl | -0.02107 | -0.02261 | -0.01943 | -0.01035 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 NI | PEINIT_W | test_ouhdl | -0.06253 | -0.07032 | -0.05513 | -0.0141 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc NI | PEINIT_W | test_ouhdl | -0.02497 | -0.03203 | -0.01779 | -0.01065 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc NI | PEINIT_W | test_ouhdl | 0.00237 | -0.00211 | 0.00724 | -0.01245 | True | 4398 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U aiou win | PEINIT_W | test_ouhdl | -0.02107 | -0.02261 | -0.01943 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 win | PEINIT_W | test_ouhdl | -0.06253 | -0.07032 | -0.05513 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc win | PEINIT_W | test_ouhdl | -0.02497 | -0.03203 | -0.01779 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc win | PEINIT_W | test_ouhdl | 0.00237 | -0.00211 | 0.00724 | 0.0 | False | 4398 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_zh | -0.02723 | -0.02963 | -0.02486 | -0.01 | False | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_en | -0.00829 | -0.00937 | -0.00729 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_hpl_tamil | -0.01605 | -0.01780 | -0.01421 | -0.01 | False | 500 | 52 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_iam500 | -0.01409 | -0.01620 | -0.01202 | -0.01 | False | 500 | 63 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_unipen_de_test | -0.01736 | -0.02111 | -0.01378 | -0.01 | False | 827 | 7 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_onhw_all | -0.02544 | -0.02604 | -0.02485 | -0.01 | False | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 | False |

### dinov3-vitl16-sat493m_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices | DEVICE_MIXED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.03984 | 0.03771 | 0.04196 | 0.0 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.06155 | 0.05312 | 0.07012 | -0.0141 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.01730 | 0.00815 | 0.02688 | -0.01065 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | -0.00081 | -0.00468 | 0.00331 | -0.01245 | True | 4387 | 32 (writer) | B200 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.04339 | 0.04055 | 0.04655 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.01040 | 0.00840 | 0.01242 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.02378 | 0.02119 | 0.02692 | -0.01 | True | 500 | 52 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.01879 | 0.01606 | 0.02163 | -0.01 | True | 500 | 63 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.01668 | 0.00469 | 0.02533 | -0.01 | True | 827 | 7 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.03320 | 0.03242 | 0.03396 | -0.01 | True | 5925 | 5925 (unit) | B200|RTX A4000 / RTX A4000 | True |
| F_FT_vs_control | F U AIoU NI | NOFM_W | test_ouhdl | -0.00850 | -0.01023 | -0.00665 | -0.01035 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 NI | NOFM_W | test_ouhdl | -0.01322 | -0.02115 | -0.00549 | -0.0141 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc NI | NOFM_W | test_ouhdl | -0.00938 | -0.01602 | -0.00157 | -0.01065 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc NI | NOFM_W | test_ouhdl | 0.00444 | 0.00080 | 0.00866 | -0.01245 | True | 4400 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U aiou win | NOFM_W | test_ouhdl | -0.00850 | -0.01023 | -0.00665 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 win | NOFM_W | test_ouhdl | -0.01322 | -0.02115 | -0.00549 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc win | NOFM_W | test_ouhdl | -0.00938 | -0.01602 | -0.00157 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc win | NOFM_W | test_ouhdl | 0.00444 | 0.00080 | 0.00866 | 0.0 | True | 4400 | 32 (writer) | B200 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_zh | 0.00546 | 0.00338 | 0.00743 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_en | 0.00165 | 0.00062 | 0.00261 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_hpl_tamil | 0.00322 | 0.00177 | 0.00464 | -0.01 | True | 500 | 52 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_iam500 | 0.00343 | 0.00213 | 0.00481 | -0.01 | True | 500 | 63 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_unipen_de_test | -0.00277 | -0.00919 | 0.00094 | -0.01 | True | 827 | 7 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_onhw_all | 0.00630 | 0.00582 | 0.00677 | -0.01 | True | 5925 | 5925 (unit) | B200|RTX A4000 / RTX A4000 | True |
| F_FT_vs_control | F U AIoU NI | PEINIT_W | test_ouhdl | -0.01968 | -0.02170 | -0.01756 | -0.01035 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 NI | PEINIT_W | test_ouhdl | -0.05104 | -0.05983 | -0.04290 | -0.0141 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc NI | PEINIT_W | test_ouhdl | -0.03155 | -0.03991 | -0.02392 | -0.01065 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc NI | PEINIT_W | test_ouhdl | 0.00135 | -0.00376 | 0.00652 | -0.01245 | True | 4401 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U aiou win | PEINIT_W | test_ouhdl | -0.01968 | -0.02170 | -0.01756 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 win | PEINIT_W | test_ouhdl | -0.05104 | -0.05983 | -0.04290 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc win | PEINIT_W | test_ouhdl | -0.03155 | -0.03991 | -0.02392 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc win | PEINIT_W | test_ouhdl | 0.00135 | -0.00376 | 0.00652 | 0.0 | False | 4401 | 32 (writer) | B200 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_zh | -0.01364 | -0.01592 | -0.01135 | -0.01 | False | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_en | -0.00266 | -0.00378 | -0.00152 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_hpl_tamil | -0.00568 | -0.00680 | -0.00440 | -0.01 | True | 500 | 52 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_iam500 | -0.00485 | -0.00614 | -0.00351 | -0.01 | True | 500 | 63 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_unipen_de_test | -0.01055 | -0.01320 | -0.00901 | -0.01 | False | 827 | 7 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_onhw_all | -0.01250 | -0.01305 | -0.01194 | -0.01 | False | 5925 | 5925 (unit) | B200|RTX A4000 / RTX A4000 | True |

### pe-spatial-b16-512_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices | DEVICE_MIXED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.04175 | 0.03969 | 0.04395 | 0.0 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.07169 | 0.06486 | 0.07864 | -0.0141 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.03213 | 0.02514 | 0.03967 | -0.01065 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | -0.00024 | -0.00408 | 0.00370 | -0.01245 | True | 4384 | 32 (writer) | B200 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.03533 | 0.03241 | 0.03849 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.00742 | 0.00549 | 0.00945 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.01898 | 0.01661 | 0.02177 | -0.01 | True | 500 | 52 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.01396 | 0.01170 | 0.01644 | -0.01 | True | 500 | 63 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.01467 | 0.00576 | 0.02206 | -0.01 | True | 827 | 7 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.02777 | 0.02701 | 0.02853 | -0.01 | True | 5925 | 5925 (unit) | B200|RTX A4000 / RTX A4000 | True |
| F_FT_vs_control | F U AIoU NI | NOFM_W | test_ouhdl | -0.00659 | -0.00803 | -0.00514 | -0.01035 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 NI | NOFM_W | test_ouhdl | -0.00307 | -0.00871 | 0.00253 | -0.0141 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc NI | NOFM_W | test_ouhdl | 0.00545 | 0.00053 | 0.01103 | -0.01065 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc NI | NOFM_W | test_ouhdl | 0.00549 | 0.00100 | 0.01033 | -0.01245 | True | 4396 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U aiou win | NOFM_W | test_ouhdl | -0.00659 | -0.00803 | -0.00514 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 win | NOFM_W | test_ouhdl | -0.00307 | -0.00871 | 0.00253 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc win | NOFM_W | test_ouhdl | 0.00545 | 0.00053 | 0.01103 | 0.0 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc win | NOFM_W | test_ouhdl | 0.00549 | 0.00100 | 0.01033 | 0.0 | True | 4396 | 32 (writer) | B200 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_zh | -0.00261 | -0.00450 | -0.00069 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_en | -0.00132 | -0.00238 | -0.00034 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_hpl_tamil | -0.00159 | -0.00280 | -0.00029 | -0.01 | True | 500 | 52 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_iam500 | -0.00140 | -0.00263 | -0.00004 | -0.01 | True | 500 | 63 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_unipen_de_test | -0.00477 | -0.00757 | -0.00261 | -0.01 | True | 827 | 7 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_onhw_all | 0.00087 | 0.00036 | 0.00139 | -0.01 | True | 5925 | 5925 (unit) | B200|RTX A4000 / RTX A4000 | True |
| F_FT_vs_control | F U AIoU NI | PEINIT_W | test_ouhdl | -0.01777 | -0.01948 | -0.01596 | -0.01035 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 NI | PEINIT_W | test_ouhdl | -0.04090 | -0.04762 | -0.03384 | -0.0141 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc NI | PEINIT_W | test_ouhdl | -0.01672 | -0.02246 | -0.01097 | -0.01065 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc NI | PEINIT_W | test_ouhdl | 0.00195 | -0.00327 | 0.00768 | -0.01245 | True | 4398 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U aiou win | PEINIT_W | test_ouhdl | -0.01777 | -0.01948 | -0.01596 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 win | PEINIT_W | test_ouhdl | -0.04090 | -0.04762 | -0.03384 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc win | PEINIT_W | test_ouhdl | -0.01672 | -0.02246 | -0.01097 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc win | PEINIT_W | test_ouhdl | 0.00195 | -0.00327 | 0.00768 | 0.0 | False | 4398 | 32 (writer) | B200 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_zh | -0.02171 | -0.02412 | -0.01939 | -0.01 | False | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_en | -0.00563 | -0.00691 | -0.00432 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_hpl_tamil | -0.01048 | -0.01203 | -0.00892 | -0.01 | False | 500 | 52 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_iam500 | -0.00968 | -0.01133 | -0.00811 | -0.01 | False | 500 | 63 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_unipen_de_test | -0.01255 | -0.01455 | -0.00962 | -0.01 | False | 827 | 7 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_onhw_all | -0.01793 | -0.01850 | -0.01738 | -0.01 | False | 5925 | 5925 (unit) | B200|RTX A4000 / RTX A4000 | True |

### vit-mae-large_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices | DEVICE_MIXED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.04730 | 0.04503 | 0.04945 | 0.0 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.08273 | 0.07474 | 0.09094 | -0.0141 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.03016 | 0.02161 | 0.03927 | -0.01065 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.00002 | -0.00337 | 0.00382 | -0.01245 | True | 4387 | 32 (writer) | B200 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.04845 | 0.04532 | 0.05184 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.01231 | 0.01047 | 0.01426 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.02679 | 0.02479 | 0.02942 | -0.01 | True | 500 | 52 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.02053 | 0.01804 | 0.02311 | -0.01 | True | 500 | 63 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.01986 | 0.00613 | 0.02965 | -0.01 | True | 827 | 7 (writer) | B200|RTX A4000 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.03728 | 0.03653 | 0.03803 | -0.01 | True | 5925 | 5925 (unit) | B200|RTX A4000 / RTX A4000 | True |
| F_FT_vs_control | F U AIoU NI | NOFM_W | test_ouhdl | -0.00103 | -0.00252 | 0.00049 | -0.01035 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 NI | NOFM_W | test_ouhdl | 0.00797 | 0.00258 | 0.01405 | -0.0141 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc NI | NOFM_W | test_ouhdl | 0.00348 | -0.00237 | 0.01042 | -0.01065 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc NI | NOFM_W | test_ouhdl | 0.00602 | 0.00208 | 0.01005 | -0.01245 | True | 4400 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U aiou win | NOFM_W | test_ouhdl | -0.00103 | -0.00252 | 0.00049 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 win | NOFM_W | test_ouhdl | 0.00797 | 0.00258 | 0.01405 | 0.0 | True | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc win | NOFM_W | test_ouhdl | 0.00348 | -0.00237 | 0.01042 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc win | NOFM_W | test_ouhdl | 0.00602 | 0.00208 | 0.01005 | 0.0 | True | 4400 | 32 (writer) | B200 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_zh | 0.01051 | 0.00876 | 0.01237 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_en | 0.00356 | 0.00249 | 0.00467 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_hpl_tamil | 0.00622 | 0.00502 | 0.00747 | -0.01 | True | 500 | 52 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_iam500 | 0.00517 | 0.00401 | 0.00639 | -0.01 | True | 500 | 63 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_unipen_de_test | 0.00041 | -0.00785 | 0.00531 | -0.01 | True | 827 | 7 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_onhw_all | 0.01038 | 0.00987 | 0.01085 | -0.01 | True | 5925 | 5925 (unit) | B200|RTX A4000 / RTX A4000 | True |
| F_FT_vs_control | F U AIoU NI | PEINIT_W | test_ouhdl | -0.01221 | -0.01433 | -0.01020 | -0.01035 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 NI | PEINIT_W | test_ouhdl | -0.02986 | -0.03591 | -0.02363 | -0.0141 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc NI | PEINIT_W | test_ouhdl | -0.01869 | -0.02527 | -0.01255 | -0.01065 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc NI | PEINIT_W | test_ouhdl | 0.00267 | -0.00179 | 0.00748 | -0.01245 | True | 4400 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U aiou win | PEINIT_W | test_ouhdl | -0.01221 | -0.01433 | -0.01020 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 win | PEINIT_W | test_ouhdl | -0.02986 | -0.03591 | -0.02363 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc win | PEINIT_W | test_ouhdl | -0.01869 | -0.02527 | -0.01255 | 0.0 | False | 4406 | 32 (writer) | B200 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc win | PEINIT_W | test_ouhdl | 0.00267 | -0.00179 | 0.00748 | 0.0 | False | 4400 | 32 (writer) | B200 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_zh | -0.00859 | -0.01041 | -0.00685 | -0.01 | False | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_en | -0.00075 | -0.00173 | 0.00031 | -0.01 | True | 500 | 60 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_hpl_tamil | -0.00267 | -0.00409 | -0.00122 | -0.01 | True | 500 | 52 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_iam500 | -0.00311 | -0.00436 | -0.00193 | -0.01 | True | 500 | 63 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_unipen_de_test | -0.00737 | -0.01127 | -0.00562 | -0.01 | False | 827 | 7 (writer) | B200|RTX A4000 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_onhw_all | -0.00842 | -0.00893 | -0.00793 | -0.01 | True | 5925 | 5925 (unit) | B200|RTX A4000 / RTX A4000 | True |

### webssl-mae1b-224_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices | DEVICE_MIXED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.04845 | 0.04640 | 0.05062 | 0.0 | True | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.08231 | 0.07450 | 0.09011 | -0.0141 | True | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.02895 | 0.02145 | 0.03584 | -0.01065 | True | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | -0.00251 | -0.00634 | 0.00124 | -0.01245 | True | 4387 | 32 (writer) | A30 / RTX A4000 | True |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | | | | -0.01 | not assessed (no row) | | | |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | | | | -0.01 | not assessed (no row) | | | |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | | | | -0.01 | not assessed (no row) | | | |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | | | | -0.01 | not assessed (no row) | | | |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | | | | -0.01 | not assessed (no row) | | | |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | | | | -0.01 | not assessed (no row) | | | |
| F_FT_vs_control | F U AIoU NI | NOFM_W | test_ouhdl | 0.00011 | -0.00111 | 0.00149 | -0.01035 | True | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 NI | NOFM_W | test_ouhdl | 0.00755 | 0.00142 | 0.01404 | -0.0141 | True | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc NI | NOFM_W | test_ouhdl | 0.00227 | -0.00474 | 0.00987 | -0.01065 | True | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc NI | NOFM_W | test_ouhdl | 0.00315 | -0.00184 | 0.00780 | -0.01245 | True | 4399 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U aiou win | NOFM_W | test_ouhdl | 0.00011 | -0.00111 | 0.00149 | 0.0 | False | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 win | NOFM_W | test_ouhdl | 0.00755 | 0.00142 | 0.01404 | 0.0 | True | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc win | NOFM_W | test_ouhdl | 0.00227 | -0.00474 | 0.00987 | 0.0 | False | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc win | NOFM_W | test_ouhdl | 0.00315 | -0.00184 | 0.00780 | 0.0 | False | 4399 | 32 (writer) | A30 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_zh | | | | -0.01 | not assessed (no row) | | | |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_en | | | | -0.01 | not assessed (no row) | | | |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_hpl_tamil | | | | -0.01 | not assessed (no row) | | | |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_iam500 | | | | -0.01 | not assessed (no row) | | | |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_unipen_de_test | | | | -0.01 | not assessed (no row) | | | |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_onhw_all | | | | -0.01 | not assessed (no row) | | | |
| F_FT_vs_control | F U AIoU NI | PEINIT_W | test_ouhdl | -0.01107 | -0.01251 | -0.00956 | -0.01035 | False | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 NI | PEINIT_W | test_ouhdl | -0.03028 | -0.03684 | -0.02356 | -0.0141 | False | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc NI | PEINIT_W | test_ouhdl | -0.01990 | -0.02755 | -0.01311 | -0.01065 | False | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc NI | PEINIT_W | test_ouhdl | -0.00020 | -0.00526 | 0.00495 | -0.01245 | True | 4400 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U aiou win | PEINIT_W | test_ouhdl | -0.01107 | -0.01251 | -0.00956 | 0.0 | False | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U penup_f1_v2 win | PEINIT_W | test_ouhdl | -0.03028 | -0.03684 | -0.02356 | 0.0 | False | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U strict_order_acc win | PEINIT_W | test_ouhdl | -0.01990 | -0.02755 | -0.01311 | 0.0 | False | 4406 | 32 (writer) | A30 / RTX A4000 | True |
| F_FT_vs_control | F U direction_acc win | PEINIT_W | test_ouhdl | -0.00020 | -0.00526 | 0.00495 | 0.0 | False | 4400 | 32 (writer) | A30 / RTX A4000 | True |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_zh | | | | -0.01 | not assessed (no row) | | | |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_en | | | | -0.01 | not assessed (no row) | | | |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_hpl_tamil | | | | -0.01 | not assessed (no row) | | | |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_iam500 | | | | -0.01 | not assessed (no row) | | | |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_unipen_de_test | | | | -0.01 | not assessed (no row) | | | |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_onhw_all | | | | -0.01 | not assessed (no row) | | | |

### webssl-mae300m-224_W clause table

| cell | clause | comparator | set | delta | CI lo | CI hi | threshold | pass | n units | clusters (type) | devices | DEVICE_MIXED |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U_FT_vs_pooledCOMPp01 | U-WIN AIoU | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.05448 | 0.05231 | 0.05664 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN penup_f1_v2 NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.08776 | 0.07943 | 0.09616 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN strict_order_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | 0.03591 | 0.02736 | 0.04451 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| U_FT_vs_pooledCOMPp01 | U-WIN direction_acc NI | pooled_COMP_p01_seeds_3 | test_ouhdl | -0.00313 | -0.00793 | 0.00144 | -0.01245 | True | 4386 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_zh | 0.04927 | 0.04636 | 0.05255 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_casia_en | 0.01196 | 0.01002 | 0.01401 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_hpl_tamil | 0.02769 | 0.02526 | 0.03071 | -0.01 | True | 500 | 52 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | test_iam500 | 0.02086 | 0.01822 | 0.02384 | -0.01 | True | 500 | 63 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_unipen_de_test | 0.02470 | 0.01570 | 0.03205 | -0.01 | True | 827 | 7 (writer) | RTX A4000 / RTX A4000 | False |
| X_composed_vs_LUOms6 | X-NI AIoU | LUO_ms6 | german_onhw_all | 0.03696 | 0.03616 | 0.03772 | -0.01 | True | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U AIoU NI | NOFM_W | test_ouhdl | 0.00614 | 0.00499 | 0.00721 | -0.01035 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 NI | NOFM_W | test_ouhdl | 0.01300 | 0.00749 | 0.01835 | -0.0141 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc NI | NOFM_W | test_ouhdl | 0.00923 | 0.00309 | 0.01589 | -0.01065 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc NI | NOFM_W | test_ouhdl | 0.00235 | -0.00188 | 0.00678 | -0.01245 | True | 4398 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U aiou win | NOFM_W | test_ouhdl | 0.00614 | 0.00499 | 0.00721 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 win | NOFM_W | test_ouhdl | 0.01300 | 0.00749 | 0.01835 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc win | NOFM_W | test_ouhdl | 0.00923 | 0.00309 | 0.01589 | 0.0 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc win | NOFM_W | test_ouhdl | 0.00235 | -0.00188 | 0.00678 | 0.0 | False | 4398 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_zh | 0.01134 | 0.00953 | 0.01326 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_casia_en | 0.00321 | 0.00221 | 0.00424 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_hpl_tamil | 0.00713 | 0.00595 | 0.00829 | -0.01 | True | 500 | 52 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | test_iam500 | 0.00550 | 0.00374 | 0.00719 | -0.01 | True | 500 | 63 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_unipen_de_test | 0.00525 | 0.00242 | 0.00749 | -0.01 | True | 827 | 7 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | NOFM_W | german_onhw_all | 0.01006 | 0.00956 | 0.01055 | -0.01 | True | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U AIoU NI | PEINIT_W | test_ouhdl | -0.00504 | -0.00670 | -0.00351 | -0.01035 | True | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 NI | PEINIT_W | test_ouhdl | -0.02483 | -0.03082 | -0.01919 | -0.0141 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc NI | PEINIT_W | test_ouhdl | -0.01294 | -0.02044 | -0.00665 | -0.01065 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc NI | PEINIT_W | test_ouhdl | -0.00121 | -0.00563 | 0.00346 | -0.01245 | True | 4401 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U aiou win | PEINIT_W | test_ouhdl | -0.00504 | -0.00670 | -0.00351 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U penup_f1_v2 win | PEINIT_W | test_ouhdl | -0.02483 | -0.03082 | -0.01919 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U strict_order_acc win | PEINIT_W | test_ouhdl | -0.01294 | -0.02044 | -0.00665 | 0.0 | False | 4406 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_FT_vs_control | F U direction_acc win | PEINIT_W | test_ouhdl | -0.00121 | -0.00563 | 0.00346 | 0.0 | False | 4401 | 32 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_zh | -0.00776 | -0.00949 | -0.00610 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_casia_en | -0.00110 | -0.00217 | -0.00000 | -0.01 | True | 500 | 60 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_hpl_tamil | -0.00176 | -0.00304 | -0.00048 | -0.01 | True | 500 | 52 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | test_iam500 | -0.00278 | -0.00446 | -0.00124 | -0.01 | True | 500 | 63 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_unipen_de_test | -0.00253 | -0.00433 | -0.00001 | -0.01 | True | 827 | 7 (writer) | RTX A4000 / RTX A4000 | False |
| F_composed_vs_control | F X AIoU (reported) | PEINIT_W | german_onhw_all | -0.00874 | -0.00925 | -0.00823 | -0.01 | True | 5925 | 5925 (unit) | RTX A4000 / RTX A4000 | False |

## 6. Arms not assessed by 2026-11-10T18:00Z [reason: A4.3(f) deadline reporting] (list from the warm monitor, FMDX_not_assessed_A43f.csv)

| arm | regime | fm | queue state | label | last completed stage |
|---|---|---|---|---|---|
| dinov3-vitl16_W | warm | dinov3-vitl16 | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training not started on CLUSTER-A (queue state SUBMITTED) |
| dinov3-vits16_W | warm | dinov3-vits16 | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step 6850 of 40000 (slurm PENDING(JobHeldUser)) |
| dinov3-vit7b16_W | warm | dinov3-vit7b16 | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| c-radiov3-l_W | warm | c-radiov3-l | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step 36200 of 40000 (slurm PENDING(Priority)) |
| c-radiov3-g_W | warm | c-radiov3-g | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step 36100 of 40000 (slurm PENDING(Priority)) |
| c-radiov4-h_W | warm | c-radiov4-h | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step 36050 of 40000 (slurm PENDING(Priority)) |
| sd3-medium-diffusers_W | warm | sd3-medium-diffusers | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| flux1-dev_W | warm | flux1-dev | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| flux1-schnell_W | warm | flux1-schnell | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| qwen3.5-4b_W | warm | qwen3.5-4b | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step 0 of 40000 (slurm PENDING(JobHeldUser)) |
| INKSIGHT_W | warm | inksight_small_p | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| inkfm_W | warm | inkfm | QUEUED | not assessed by 2026-11-10T18:00Z (A4.3f) | training not started on CLUSTER-A (queue state QUEUED) |
| NOFM_S | scratch | - | PINNED_CLUSTER-B | not assessed by 2026-11-10T18:00Z (A4.3f) | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| PEINIT_S | scratch | - | PINNED_CLUSTER-B | not assessed by 2026-11-10T18:00Z (A4.3f) | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| dinov3-vitl16_S | scratch | dinov3-vitl16 | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step 2600 of 100000 (slurm PENDING(JobHeldUser)) |
| cleandift-sd21_S | scratch | cleandift-sd21 | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step 5850 of 100000 (slurm PENDING(JobHeldUser)) |
| dinov3-vits16_S | scratch | dinov3-vits16 | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| dinov3-vitb16_S | scratch | dinov3-vitb16 | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| dinov3-vit7b16_S | scratch | dinov3-vit7b16 | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| c-radiov3-b_S | scratch | c-radiov3-b | PINNED_CLUSTER-B | not assessed by 2026-11-10T18:00Z (A4.3f) | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| c-radiov3-l_S | scratch | c-radiov3-l | PINNED_CLUSTER-B | not assessed by 2026-11-10T18:00Z (A4.3f) | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| c-radiov3-h_S | scratch | c-radiov3-h | PINNED_CLUSTER-B | not assessed by 2026-11-10T18:00Z (A4.3f) | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| c-radiov3-g_S | scratch | c-radiov3-g | PINNED_CLUSTER-B | not assessed by 2026-11-10T18:00Z (A4.3f) | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| c-radiov4-so400m_S | scratch | c-radiov4-so400m | PINNED_CLUSTER-B | not assessed by 2026-11-10T18:00Z (A4.3f) | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| c-radiov4-h_S | scratch | c-radiov4-h | PINNED_CLUSTER-B | not assessed by 2026-11-10T18:00Z (A4.3f) | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| pe-spatial-b16-512_S | scratch | pe-spatial-b16-512 | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| webssl-mae300m-224_S | scratch | webssl-mae300m-224 | PINNED_CLUSTER-B | not assessed by 2026-11-10T18:00Z (A4.3f) | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| vit-mae-large_S | scratch | vit-mae-large | PINNED_CLUSTER-B | not assessed by 2026-11-10T18:00Z (A4.3f) | training not started on CLUSTER-A (queue state PINNED_CLUSTER-B) |
| sd3-medium-diffusers_S | scratch | sd3-medium-diffusers | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| flux1-dev_S | scratch | flux1-dev | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| flux1-schnell_S | scratch | flux1-schnell | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| qwen3.5-4b_S | scratch | qwen3.5-4b | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| INKSIGHT_S | scratch | inksight_small_p | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| inkfm_S | scratch | inkfm | QUEUED | not assessed by 2026-11-10T18:00Z (A4.3f) | training not started on CLUSTER-A (queue state QUEUED) |
| sd3.5-large-diffusers-layout_W | warm | sd3.5-large-diffusers-layout | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| sd3.5-large-turbo-diffusers-layout_W | warm | sd3.5-large-turbo-diffusers-layout | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| sd3.5-medium-diffusers-layout_W | warm | sd3.5-medium-diffusers-layout | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| sd3.5-large-diffusers-layout_S | scratch | sd3.5-large-diffusers-layout | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| sd3.5-large-turbo-diffusers-layout_S | scratch | sd3.5-large-turbo-diffusers-layout | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| sd3.5-medium-diffusers-layout_S | scratch | sd3.5-medium-diffusers-layout | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| webssl-mae1b-224_W | warm | webssl-mae1b-224 | DONE | not assessed by 2026-11-10T18:00Z (A4.3f) | FT selected ms3; incomplete: bb rows 0/18; cmp rows 0/18 (sampling PENDING(Priority)) |
| webssl-mae1b-224_S | scratch | webssl-mae1b-224 | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| dinov3-vitl16-sat493m_S | scratch | dinov3-vitl16-sat493m | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |
| chandra-ocr-2_W | warm | chandra-ocr-2 | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 40000 (slurm PENDING(JobHeldUser)) |
| chandra-ocr-2_S | scratch | chandra-ocr-2 | SUBMITTED | not assessed by 2026-11-10T18:00Z (A4.3f) | training step None of 100000 (slurm PENDING(JobHeldUser)) |

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

- webssl-mae1b-224_W: clauses without rows: F X AIoU (reported); X-NI AIoU.
- All arms listed in section 6 (45 rows).
