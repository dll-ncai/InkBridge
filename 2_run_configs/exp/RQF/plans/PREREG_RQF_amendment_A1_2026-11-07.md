# PREREG_RQ_finalise — Amendment A1 (7 Nov 2026)
Parent: PREREG_RQ_finalise_2026-11-07.md, sha256 04766e973badf7ff469d468e25bab17cebabf38f2c9537d731a4d91ea960c624. No existing rule, margin, arm or seed is changed.
A1.1 RQ mapping. RQ1 Shortcut = R1a + M1 + M3. RQ2 Label-free removal = R1b + R1c + R3a + R3b. RQ3 Transfer = R2a-R2e + FM arms (FMD A4.5 / cell F) + M2. Document: RQ_uhwr_2026-11-07.md.
A1.2 M1 (descriptive). E1 selectivity check (existing E1 perturbation code, off-factor leak <= 0.10) for AIoU, tau, direction, pen-up F1 v2, strict order on all 7 test sets (OUHD-L, CASIA-zh, CASIA-en, HPL Tamil, IAM500, UNIPEN-de, OnHW). Reported as a metric x set pass/fail table; no claim of a clean factor-separated protocol unless every cell passes.
A1.3 M3 (descriptive). From existing per-unit rows: share of paired units with |dAIoU| <= 0.01 and direction disagreement >= 0.5 of strokes, for (LUO identity, LUO mirror_x), (InkSight, E0), (E0 on UNIPEN-de, LUO on UNIPEN-de).
A1.4 M2 (descriptive). Across every FM WARM arm with a cell-U read: Spearman rho between screen Urdu T0 CI lo and (arm - NOFM_W) AIoU and pen-up F1 v2; 95 % CI by bootstrap over arms (B = 2,000). Reported with n arms; no inference if n < 6.
A1.5 Nothing in this amendment waits on or alters a running job.
