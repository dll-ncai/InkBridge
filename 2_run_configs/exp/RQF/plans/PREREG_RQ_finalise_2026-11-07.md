# PREREG — RQ finalisation round (RQF), frozen 2026-11-07

Author: lead session (<agent>), frame d94cf545. Approved plan: artifact b3da4ee6 v0ccb171b.
Status of every result in this round: CONFIRMATORY only where a rule below says so; everything else EXPLORATORY.
Parent documents: exp/FMD/plans/FMD_PREREG_2026-11-03.md (sha256 7a71c715dc5ca1cf146f829b8c7f498811e1330e1eefa2bf1b3b60fddbc5d154), ORDER_LOOP_RULES.md, claims ledger 2026-11-07 (v 7ba071f8).

## 0. Common protocol (all RQs)
- Scorer: uhwr_eval 0.3.1. CI: writer-cluster percentile bootstrap, B = 2,000, 95 %; unit bootstrap on OnHW (2 writers). Paired differences on the unit intersection, per-unit mean over training and sampling seeds.
- Margins (1.5 x MDE80, as logged): AIoU 0.01035; pen-up F1 v2 -0.0141; strict order -0.01065; direction -0.01245. X-NI: six-set composed AIoU CI lo > -0.010.
- Rules reused verbatim: U-WIN and X-NI (FMD A4.5), FM-ATTRIBUTED (FMD cell F), ORDER-WIN (ORDER_LOOP_RULES), K1 (E137).
- Sampling device: any CLUSTER-A/CLUSTER-B/VPS GPU with cc >= 8 (FMD A9.1). Each preds file records device, node, sidecar. When method and comparator differ in device, the row is tagged (M); for every new comparator family one same-device check row (comparator on both devices) is produced.
- Sampling seeds 0-2 on every test set used in a rule. Training seeds: 20261034/35/36 for fine-tunes (fresh), 20261017 for second backbone seeds.
- Lanes: CLUSTER-A (gpuidle cc>=8, b200idle, CPU partitions), CLUSTER-B, VPS 4090 runner. NO previously queued or running job is cancelled or pre-empted (<author>, 7 Nov). Two-lane copies are raced (first to START wins; loser cancelled; both ids logged in exp/RQF/queue/rqf_log.md). VPS cmd band for this round: 500-599.
- Dependents: nothing is submitted that waits on another experiment's verdict. Chains inside one experiment (train -> select -> sample -> score) are allowed (afterok).
- Every new data split gets PROVENANCE json (source files + sha256, transform, seed, unit/writer counts).
- Test sets are sealed until a run's own milestone selection is done on val (val1000 for Urdu; each set's val split otherwise).

## 1. RQ1 — zero-target-data direction frame
R1a (W1, mirrored-LTR control; CONFIRMATORY). Sets: IAM500, CASIA-en, HPL Tamil test (500 units each) mirrored x -> -x, render bit-exact (shifted flip as in E137). Bases: LUO ms6, joint M5 ms9.
  - COLLAPSE: direction(identity, mirrored set) - direction(identity, original set) CI hi < -0.10, on >= 2 of 3 sets per base.
  - REPAIR: direction(mirror_x, mirrored set) - direction(identity, mirrored set) CI lo > 0 AND AIoU difference CI lo > -0.01035, on >= 2 of 3 sets.
  - Note fixed in advance: mirror_x on a mirrored LTR set restores the original orientation, so REPAIR is partly a consistency check; COLLAPSE is the informative clause (direction-bound vs Urdu-bound failure).
R1b (reverse direction; CONFIRMATORY). E0 ms8 with identity vs mirror_x on UNIPEN-de, IAM500, CASIA-en, OnHW. REPAIR clause as R1a on >= 3 of 4 sets.
R1c (Urdu zero-shot, paired CIs; descriptive + K1 restated). Bases with no Urdu in training: LUO ms6, NOFM_W backbone (bb ms selected), PEINIT_W backbone, TRACE (test253 lines). identity vs mirror_x, 3 sampling seeds, paired CIs on AIoU, tau, direction, strict, pen-up. K1 (tau CI lo >= 0.65) reported per base; not re-scored as a new claim for LUO.

## 2. RQ2 — adaptation with 1 % target data
Definitions (FMD A4.2): WARM = LUO ms6 EMA non-encoder modules + new ViT-S/14 224-px student encoder (render thickness 7), all parameters, +40,000 steps on m5_luo_train; NOFM = no FM loss; PEINIT = encoder initialised from pe-spatial-s16-512. Then the A4.3(e) p01 fine-tune (10,000 steps, val1000 E0-rule selection).
R2a (seeds; CONFIRMATORY). NOFM_W and PEINIT_W: 3 fresh p01 FT seeds each on the existing selected backbones. Read: U-WIN vs COMP-p01 pooled (3 seeds), Holm over the 2 arms. Second backbone seed (20261017) per arm + 1 FT seed: EXPLORATORY variance row.
R2b (compute-matched control LUOC40; CONFIRMATORY for attribution). LUO ms6 (model + EMA, fresh optimizer), ORIGINAL 64-px encoder and LUO recipe (64-px thickness 1-3 aug), +40,000 steps on m5_luo_train, seed 20261016, select by the WARM backbone selection rule; then p01 FT 3 seeds.
  - STEPS effect = LUOC40 - COMP-p01. ENCODER-ATTRIBUTED (per arm) = arm - LUOC40 on cell U meets: AIoU CI lo > 0 AND the three order NI clauses. Composed cell X reported alongside.
R2c (data-efficiency curve; CONFIRMATORY at p10, EXPLORATORY at p30). Fractions p10, p30 (subset files m5_ft_ouhdl_p10/p30; if p30 is absent build it writer-stratified with seed 20261107 and PROVENANCE). Arms: NOFM_W, PEINIT_W, LUOC40, plain LUO-FT (p10 ms10 reused; p30 run if absent), 1 FT seed each (3 at p10 for NOFM_W/PEINIT_W/LUO-FT). Rule at each fraction: U-WIN vs plain LUO-FT at the same fraction (<author>'s bar).
R2d (second held-out target script: HPL Tamil; CONFIRMATORY design, likely after 10 Nov). Leave-Tamil-out backbone LTO = LUO recipe on m5 train minus Tamil (Urdu INCLUDED), from scratch, seed 20261016; LTO_W = WARM stage (NOFM) on LTO, 40k steps. Tamil 1 % subset: writer-stratified from HPL Tamil train, seed 20261107, PROVENANCE. FT: plain LTO-FT p01 and LTO_W-FT p01, 3 seeds each; selection on HPL Tamil val. Rule: U-WIN analogue on test_hpl_tamil (LTO_W-FT vs LTO-FT) with the same margins; X-NI on the other 6 sets incl. OUHD-L. Zero-shot LTO on Tamil +- mirror_x reported (descriptive).
R2e (German retention; descriptive). Per-writer direction and tau on UNIPEN-de (7 writers) for all RQ2 arms vs LUO ms6.

## 3. RQ3 — ground-truth-free test-time selection
Verifier: endpoint-agreement MBR, file 579c9b22, k = 3, frozen; plus random-of-3 control.
R3a (other base models, Urdu; CONFIRMATORY). Bases: E0 ms8, joint M5 ms9, LUO-FT p10 ms10, NOFM_W FT, PEINIT_W FT. Each vs its own pooled 3-seed base on test_ouhdl. ORDER-WIN per base, Holm over 5 bases. Random-of-3 control must have all CIs containing 0.
R3b (other scripts; CONFIRMATORY). Bases: LUO ms6, joint M5 ms9, COMP-p01, NOFM_W composed; sets: CASIA-zh, CASIA-en, HPL Tamil, IAM500, UNIPEN-de, OnHW, seeds 0-2. ORDER-WIN per (base, set); report count; Holm within base.

## 4. FM arms (existing rules, no change)
Remaining WARM arms run under fmdx_controller and FMD A4.5 / cell F. FT for vit-mae-large_W and dinov3-vitl16-sat493m_W submitted now. Every FM arm is also read against LUOC40 (R2b) as an extra descriptive row.

## 5. Verdict messaging
Each verdict above is sent to <author> as a facts-only check-in when its read lands; launches continue meanwhile.
