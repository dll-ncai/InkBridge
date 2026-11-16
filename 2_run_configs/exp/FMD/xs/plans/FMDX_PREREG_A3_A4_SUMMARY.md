# FMDX pre-registration amendments A3 and A4: summary (facts only)

The amendments are appended to `exp/FMD/plans/FMD_PREREG_2026-11-03.md`, both at UTC 2026-11-03T13:05:33Z.

| file state | sha256 |
|---|---|
| before A3 (= after A2), kept as `FMD_PREREG_2026-11-03.pre_A3.md` | c5bb785c597ec77862d85e5f40ef5fa12b598776226eba96fe201a0cacb451d9 |
| after A3, kept as `FMD_PREREG_2026-11-03.pre_A4.md`, recorded in `.A3.sha256` | 53150041ed0f1ea214649d3073f01b3a757fce9d55d1198ed352a0dd5e517afb |
| after A4 (FROZEN), recorded in `.A4.sha256`; kept as `FMD_PREREG_2026-11-03.pre_A5.md` | eef828663d75295980f25b775542f5077a9d034f64a3b6025046db5d8011f779 |
| after A5 (CURRENT; clarification appended 2026-11-03T13:08:47Z), recorded in `.A5.sha256` and in `exp/FMD/xs/admin/PREREG_FROZEN.flag` | d5137178cf0c3beb5791fc7c45e2c9e986f38d30cc4bc42f939a3194cc4f2adb |

Byte-identical copies of the current file (sha256 d5137178..., checked) are at: CLUSTER-A `exp/FMD/plans/`, `exp/FMD/xs/plans/` and `$U/plans/`; the VPS `$VPS_ROOT/uhwr/stage/exp/FMD/xs/plans/`; CLUSTER-B `$U_B/exp/FMD/xs/plans/`. The earlier `$U/plans/` copy (sha256 c5bb785c…, the A2 state) was renamed `FMD_PREREG_2026-11-03.superseded_c5bb785c.md`; nothing was deleted.
State before the freeze: at 13:05Z, `exp/FMD/xs/{screen,features,targets,verdicts,runs}` were empty. Running at the time were only the XS staging downloads and the render job (Slurm 24841106-24841114, 24841128, 24841130).

## A3: cross-script screen of all FMs
- The 6 screen sets are: ouhdl_val1000 (1,000 units / 31 writers; Urdu gate, Section 5 unchanged); casia_zh_val250 (250/24); casia_en_val250 (250/24); hpl_tamil_val250 (250/6); iam_val250 (250 units / 62 writer keys, 42 of them 'nowriter' line keys); unipen_de_tr3927 (all 3,927 units of UNIPEN-de train / 12 writers).
- Folds are writer-disjoint in every set (5 folds; the fallback to unit-level folds applies only below 5 clusters and triggers for no set).
- Comparators: on Urdu, the best of {nomL1, nomL2} x {mid16, cat16}. On cross-script sets, the best of {LUO ms6 (ab6b178d…), M5 joint ms9 (b9592f22…)} x {mid16, cat16}. Zero-shot trajectories are compared with E0 ms8 on Urdu and LUO ms6 (seeds 0-2) on cross-script sets. VLM V2 is compared with the majority label on cross-script sets; V1-V3 run on Urdu.
- Roster: every phase-1 FM plus every id the XS staging track is fetching (SigLIP2 base/giant, DINOv3 ViT/ConvNeXt variants, DINOv2-reg, C-RADIOv3 B/L/H/g, C-RADIOv4 SO400M/H, PE-Spatial T/S/B/L/G, PE-Core B/L, TIPSv2 B/L/so400m/g, Web-SSL, MAE, AIMv2, Depth-Anything v2/3, SAM3, TrOCR, Florence-2, SD1.5 DIFT, CleanDIFT SD1.5/SD2.1, SD3-medium, FLUX.1-dev/schnell, Qwen2-VL-2B, Qaari, DeepSeek-OCR(-2) + Urdu LoRAs, GOT-OCR2, PaddleOCR-VL, LightOnOCR-2, StarVector-1B, InkSight, InkFM). Each has a pre-named input, grid and gate layer/cell. Any further candidate needs a lettered A3.a, A3.b, … amendment before its first output.
- Rules: SHAPE(s) holds when T0 CI lo > -0.01035 vs the comparator. ORDER(s) holds when at least one order target or zero-shot order task has CI lo > 0. xs_pass requires SHAPE on all 5 cross-script sets AND ORDER on at least 3 of 5.
- Status: PASS_XS = xs_pass. PASS_URDU_ONLY = urdu_pass and not xs_pass. FAIL = neither (early FAIL is allowed once xs_pass is impossible). PENDING = otherwise.
- priority_score = status_rank·1e9 + sets_passed·1e7 + order-strength term + T0 term.
- verdict_<fm>.json schema 'fmdx_verdict_v1' (A3.7).

## A4: queued gated backbone arms
- WARM: LUO ms6 non-encoder weights + 224-px ViT-S/14 student encoder, 40k steps, LUO recipe. The encoder group uses lr 2e-4 with a 200-step warm-up; the rest uses lr 8e-5.
- SCRATCH: random init, 100k steps, LUO recipe, seed 20261016.
- One WARM and one SCRATCH arm per FM with a token map (REPA, lambda 0.5). InkSight gets a WARM and a SCRATCH pseudo-label arm: 20k units per LTR source, IoU >= 0.80 filter.
- Controls: NOFM-WARM/SCRATCH and PEINIT-WARM/SCRATCH (pe-spatial-s16-512 weights, patch embedding resized 16 to 14).
- The encoder input is the 224-px render at fixed thickness 7 (no thickness augmentation on that path).
- Queue: all arms enter `exp/FMD/xs/admin/arm_queue.csv` up front. Cancel when status is FAIL; PASS_URDU_ONLY stays alive as non-promotable. Initial order: NOFM, DINOv3, CleanDIFT-SD2.1 (or SD2.1), PEINIT; after that, by priority_score. Each job checks its verdict at start and at every 10k save, writing GATE_FAIL and exiting 0 on FAIL.
- Milestone selection follows the LUO rule (4 val250 sets, pooled LDTW).
- Urdu p01 FT follows the COMP-p01 recipe exactly: 10k steps with val1000 E0-rule milestone selection, identity frame, seed 20261034. Primary sampling is on the VPS 4090, seeds 0-2.
- Cells:
  - U: test_ouhdl FT vs pooled COMP-p01. Win = AIoU CI lo > 0, pen-up > -0.0141, strict > -0.01065, direction > -0.01245.
  - X: composed (router 0.5) vs LUO ms6 on the six non-Urdu test sets, AIoU CI lo > -0.010 on every set.
  - F: guided minus NOFM and guided minus PEINIT, within regime.
- Promotion: PASS_XS AND U-WIN AND X-NI. Single training seed, so every result is labelled EXPLORATORY.
- Arms not finished by 2026-11-10T18:00Z are reported 'not assessed'.

## Inspection facts used (3 Nov 2026)
- `m5_luo_train.npz` has 80,000 units (20,000 each of zh, en, Tamil, IAM) and no GT-ink arrays (seq only).
- LUO ms6 is step 60,000, selected on pooled native LDTW (0.27783) over the 4 val250 sets.
- COMP-p01 seeds 20261034/35/36 selected ms3/ms3/ms4 of 10k steps. Their 4090 preds have seeds 0-2 on test_ouhdl and gold, and seed 0 only on the six non-Urdu test sets.
- Route A has LUO ms6 4090 preds with seeds 0-2 on all 7 test sets (checkpoint sha ab6b178d…).

## A5: clarifications appended 2026-11-03T13:08:47Z, before any FMDX output
- A5.1: any rows-file name is accepted (e.g. `rows_xs_enc_<set id>.csv`), provided it holds one set, uses schema v1, carries the set id in its name and is listed in the verdict's rows_sha256.
- A5.2: the capacity note may fix a shared REPA training subset S (seeded, result-independent, with sha256). If it does, every guided arm uses the same S, REPA is applied to the batch members in S, and the denoising loss uses all units. S is logged in the decision record before the first guided arm. Without such a note, targets cover all 80,000 units.
- Reason: alignment with the approved plan text (version fe054fca-4871-4713-a867-6b5e07abc25c), which names `rows_xs_enc_<set>.csv` and a 'fixed shared training subset for REPA targets'.
