# FMDX warm contrasts (no verdicts, no promotion labels)
Read label: PRELIMINARY: NOFM_W backbone only (A4000), comparators A4000

Per-unit = mean over sampling seeds 0-2 (native protocol, uhwr_eval 0.3.1); paired difference arm - comparator; 95 % percentile CI of B=2000 writer-cluster bootstrap, default_rng(20261103); german_onhw_all unit-cluster. Comparator rows: A4000 re-sampled (prereg A8.1). Every row: EXPLORATORY (single training seed). Arms read: 1 (NOFM_W). No multiplicity correction.

## X_backbone_only_vs_LUOms6 / test_casia_zh
| arm | comparator | n | aiou d [lo, hi] | penup_f1_v2 d [lo, hi] | strict_order_acc d [lo, hi] | direction_acc d [lo, hi] | tau_cp d [lo, hi] | ldtw d [lo, hi] |
|---|---|---|---|---|---|---|---|---|
| NOFM_W | LUO_ms6 | 472-500 | +0.0379 [+0.0352, +0.0409] | +0.0648 [+0.0482, +0.0803] | +0.0413 [+0.0144, +0.0675] | +0.0027 [-0.0037, +0.0089] | -0.0036 [-0.0113, +0.0039] | +0.0002 [-0.0051, +0.0070] |

## X_backbone_only_vs_LUOms6 / test_casia_en
| arm | comparator | n | aiou d [lo, hi] | penup_f1_v2 d [lo, hi] | strict_order_acc d [lo, hi] | direction_acc d [lo, hi] | tau_cp d [lo, hi] | ldtw d [lo, hi] |
|---|---|---|---|---|---|---|---|---|
| NOFM_W | LUO_ms6 | 177-500 | +0.0087 [+0.0068, +0.0107] | +0.0104 [-0.0054, +0.0269] | +0.0013 [-0.0136, +0.0161] | -0.0012 [-0.0094, +0.0069] | -0.0163 [-0.0519, +0.0181] | +0.0152 [+0.0030, +0.0281] |

## X_backbone_only_vs_LUOms6 / test_hpl_tamil
| arm | comparator | n | aiou d [lo, hi] | penup_f1_v2 d [lo, hi] | strict_order_acc d [lo, hi] | direction_acc d [lo, hi] | tau_cp d [lo, hi] | ldtw d [lo, hi] |
|---|---|---|---|---|---|---|---|---|
| NOFM_W | LUO_ms6 | 217-500 | +0.0205 [+0.0181, +0.0235] | +0.0288 [+0.0103, +0.0492] | +0.0080 [-0.0052, +0.0225] | -0.0022 [-0.0095, +0.0044] | -0.0143 [-0.0368, +0.0052] | +0.0024 [-0.0096, +0.0156] |

## X_backbone_only_vs_LUOms6 / test_iam500
| arm | comparator | n | aiou d [lo, hi] | penup_f1_v2 d [lo, hi] | strict_order_acc d [lo, hi] | direction_acc d [lo, hi] | tau_cp d [lo, hi] | ldtw d [lo, hi] |
|---|---|---|---|---|---|---|---|---|
| NOFM_W | LUO_ms6 | 216-500 | +0.0154 [+0.0130, +0.0177] | +0.0123 [-0.0011, +0.0256] | -0.0087 [-0.0208, +0.0028] | -0.0085 [-0.0170, -0.0001] | -0.0207 [-0.0434, -0.0026] | +0.0208 [+0.0078, +0.0344] |

## X_backbone_only_vs_LUOms6 / german_unipen_de_test
| arm | comparator | n | aiou d [lo, hi] | penup_f1_v2 d [lo, hi] | strict_order_acc d [lo, hi] | direction_acc d [lo, hi] | tau_cp d [lo, hi] | ldtw d [lo, hi] |
|---|---|---|---|---|---|---|---|---|
| NOFM_W | LUO_ms6 | 166-827 | +0.0227 [+0.0203, +0.0272] | +0.0680 [+0.0472, +0.1116] | +0.0218 [+0.0044, +0.0490] | +0.0003 [-0.0153, +0.0263] | -0.0092 [-0.0409, +0.0635] | -0.0279 [-0.0857, -0.0045] |

## X_backbone_only_vs_LUOms6 / german_onhw_all
| arm | comparator | n | aiou d [lo, hi] | penup_f1_v2 d [lo, hi] | strict_order_acc d [lo, hi] | direction_acc d [lo, hi] | tau_cp d [lo, hi] | ldtw d [lo, hi] |
|---|---|---|---|---|---|---|---|---|
| NOFM_W | LUO_ms6 | 3479-5925 | +0.0269 [+0.0262, +0.0277] | +0.0252 [+0.0203, +0.0304] | +0.0094 [+0.0041, +0.0150] | +0.0033 [+0.0002, +0.0064] | -0.0327 [-0.0438, -0.0222] | -0.0028 [-0.0070, +0.0009] |
