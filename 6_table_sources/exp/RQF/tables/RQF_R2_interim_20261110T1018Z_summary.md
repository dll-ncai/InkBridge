# RQF R2 tables (facts only)
No multiplicity correction beyond the prereg (Holm over 2 arms for R2a is applied by the reader). Rule flags: U_WIN, ORDER_NI3, ENCODER_ATTRIBUTED as defined above.
X-NI (six-set composed AIoU CI lo > -0.010, R2a): {"NOFM_W": true, "PEINIT_W": true}

| evidence_label | cell | arm | comparator | set | n_train_seeds_arm | n_units | aiou_d | aiou_lo | penup_f1_v2_lo | strict_order_acc_lo | direction_acc_lo | U_WIN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_U | NOFM_W | pooled_COMP_p01 | test_ouhdl | 3 | 4406 | 0.0469 | 0.0449 | 0.0675 | 0.0211 | -0.0072 | True |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_X_composed | NOFM_W | LUO_ms6 | test_casia_zh | 3 | 500 | 0.0379 | 0.0352 | 0.0482 | 0.0144 | -0.0037 | True |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_X_composed | NOFM_W | LUO_ms6 | test_casia_en | 3 | 500 | 0.0087 | 0.0068 | -0.0054 | -0.0136 | -0.0094 | False |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_X_composed | NOFM_W | LUO_ms6 | test_hpl_tamil | 3 | 500 | 0.0206 | 0.0181 | 0.0103 | -0.0052 | -0.0095 | True |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_X_composed | NOFM_W | LUO_ms6 | test_iam500 | 3 | 500 | 0.0154 | 0.013 | -0.0011 | -0.0208 | -0.017 | False |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_X_composed | NOFM_W | LUO_ms6 | german_unipen_de_test | 3 | 827 | 0.0186 | 0.0114 | -0.0531 | -0.0905 | -0.1754 | False |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_X_composed | NOFM_W | LUO_ms6 | german_onhw_all | 3 | 5925 | 0.0269 | 0.0262 | 0.0203 | 0.0041 | 0.0001 | True |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_U | PEINIT_W | pooled_COMP_p01 | test_ouhdl | 3 | 4406 | 0.0595 | 0.0573 | 0.1036 | 0.0421 | -0.0065 | True |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_X_composed | PEINIT_W | LUO_ms6 | test_casia_zh | 3 | 500 | 0.057 | 0.054 | 0.1987 | 0.1605 | 0.0029 | True |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_X_composed | PEINIT_W | LUO_ms6 | test_casia_en | 3 | 500 | 0.0131 | 0.0113 | 0.0171 | -0.0014 | -0.0042 | True |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_X_composed | PEINIT_W | LUO_ms6 | test_hpl_tamil | 3 | 500 | 0.0295 | 0.0269 | 0.0585 | 0.0073 | -0.0046 | True |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_X_composed | PEINIT_W | LUO_ms6 | test_iam500 | 3 | 500 | 0.0236 | 0.0208 | 0.0234 | 0.0014 | -0.009 | True |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_X_composed | PEINIT_W | LUO_ms6 | german_unipen_de_test | 3 | 827 | 0.0268 | 0.0156 | 0.0182 | -0.0281 | -0.1695 | False |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2a_X_composed | PEINIT_W | LUO_ms6 | german_onhw_all | 3 | 5925 | 0.0457 | 0.0449 | 0.0993 | 0.0438 | 0.0083 | True |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2c_p10_U_vs_LUOFT | NOFM_W | LUOFT_p10 | test_ouhdl | 3 | 4406 | 0.0428 | 0.0412 | 0.061 | 0.0241 | -0.0072 | True |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2c_p10_U_vs_pooled_COMP_p01 | NOFM_W | pooled_COMP_p01 | test_ouhdl | 3 | 4406 | 0.0721 | 0.0697 | 0.1405 | 0.0893 | 0.0067 | True |
| EXPLORATORY | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_casia_zh | 3 | 500 | 0.0791 | 0.0743 | 0.1052 | 0.0144 | -0.0247 | False |
| EXPLORATORY | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_casia_en | 3 | 500 | 0.0745 | 0.0684 | 0.2521 | 0.1464 | 0.0297 | True |
| EXPLORATORY | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_hpl_tamil | 3 | 500 | 0.0842 | 0.0772 | 0.1763 | 0.0906 | -0.0594 | False |
| EXPLORATORY | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_iam500 | 3 | 500 | 0.0769 | 0.071 | 0.2166 | 0.106 | 0.0011 | True |
| EXPLORATORY | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | german_unipen_de_test | 3 | 827 | 0.0806 | 0.0666 | 0.1386 | 0.0849 | -0.0461 | False |
| EXPLORATORY | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | german_onhw_all | 3 | 5925 | 0.0383 | 0.0372 | 0.0783 | 0.0337 | 0.0239 | True |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2c_p10_U_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_ouhdl | 3 | 4406 | 0.0507 | 0.0491 | 0.0784 | 0.0313 | -0.0069 | True |
| CONFIRMATORY (CLUSTER-A; R2b = CLUSTER-A copy 24964380 chain, <author> 10 Nov) | R2c_p10_U_vs_pooled_COMP_p01 | PEINIT_W | pooled_COMP_p01 | test_ouhdl | 3 | 4406 | 0.08 | 0.0775 | 0.1582 | 0.0973 | 0.0063 | True |
| EXPLORATORY | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_casia_zh | 3 | 500 | 0.1166 | 0.1097 | 0.2495 | 0.0547 | 0.0649 | True |
| EXPLORATORY | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_casia_en | 3 | 500 | 0.0776 | 0.0713 | 0.299 | 0.168 | 0.0394 | True |
| EXPLORATORY | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_hpl_tamil | 3 | 500 | 0.1185 | 0.1096 | 0.3721 | 0.2329 | 0.0669 | True |
| EXPLORATORY | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_iam500 | 3 | 500 | 0.0847 | 0.0772 | 0.3001 | 0.1398 | -0.0079 | True |
| EXPLORATORY | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | german_unipen_de_test | 3 | 827 | 0.0862 | 0.0712 | 0.2234 | 0.1105 | -0.0244 | False |
| EXPLORATORY | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | german_onhw_all | 3 | 5925 | 0.0462 | 0.0451 | 0.1181 | 0.0255 | 0.0025 | True |
| EXPLORATORY | R2c_p30_U_vs_pooled_COMP_p01 | NOFM_W | pooled_COMP_p01 | test_ouhdl | 1 | 4406 | 0.0811 | 0.0785 | 0.1672 | 0.1164 | 0.009 | True |

## Missing inputs
- LUO-FT p30 comparator: existing Route A M5 ft/p30 ms10 (4090 preds, one training seed) - read separately, not pooled here
- LUOC40/p01/german_unipen_de_test
- LUOC40/p01/test_ouhdl
- LUOC40/p10/german_onhw_all
- LUOC40/p10/german_unipen_de_test
- LUOC40/p10/test_casia_en
- LUOC40/p10/test_casia_zh
- LUOC40/p10/test_hpl_tamil
- LUOC40/p10/test_iam500
- LUOC40/p10/test_ouhdl
- LUOC40/p30/test_ouhdl
- LUOC40VPS/p01/german_unipen_de_test
- LUOC40VPS/p01/test_ouhdl
- NOFM_W2 FT
- PEINIT_W/p30/ft/test_ouhdl
- PEINIT_W2 FT
- R2b_ENCODER_ATTRIBUTED:NOFM_W vs LUOC40:test_ouhdl
- R2b_ENCODER_ATTRIBUTED:PEINIT_W vs LUOC40:test_ouhdl
- R2b_ENCODER_ATTRIBUTED_VPS:NOFM_W vs LUOC40VPS:test_ouhdl
- R2b_ENCODER_ATTRIBUTED_VPS:PEINIT_W vs LUOC40VPS:test_ouhdl
- R2b_STEPS_effect:LUOC40 vs pooled_COMP_p01:test_ouhdl
- R2b_STEPS_effect_VPS:LUOC40VPS vs pooled_COMP_p01:test_ouhdl
- R2b_VPS_vs_CLUSTER-A_LUOC40:LUOC40VPS vs LUOC40:test_ouhdl
- R2c_p10_U_vs_LUOFT:LUOC40 vs LUOFT_p10:test_ouhdl
- R2c_p10_U_vs_pooled_COMP_p01:LUOC40 vs pooled_COMP_p01:test_ouhdl
- R2c_p10_X_FTnoRouter_vs_LUOFT:LUOC40 vs LUOFT_p10:german_onhw_all
- R2c_p10_X_FTnoRouter_vs_LUOFT:LUOC40 vs LUOFT_p10:german_unipen_de_test
- R2c_p10_X_FTnoRouter_vs_LUOFT:LUOC40 vs LUOFT_p10:test_casia_en
- R2c_p10_X_FTnoRouter_vs_LUOFT:LUOC40 vs LUOFT_p10:test_casia_zh
- R2c_p10_X_FTnoRouter_vs_LUOFT:LUOC40 vs LUOFT_p10:test_hpl_tamil
- R2c_p10_X_FTnoRouter_vs_LUOFT:LUOC40 vs LUOFT_p10:test_iam500
- R2c_p30_U_vs_pooled_COMP_p01:LUOC40 vs pooled_COMP_p01:test_ouhdl
- R2c_p30_U_vs_pooled_COMP_p01:PEINIT_W vs pooled_COMP_p01:test_ouhdl
