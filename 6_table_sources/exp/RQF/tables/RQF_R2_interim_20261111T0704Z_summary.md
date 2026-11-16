# RQF R2 tables (facts only)
No multiplicity correction beyond the prereg (Holm over 2 arms for R2a is applied by the reader). Rule flags: U_WIN, ORDER_NI3, ENCODER_ATTRIBUTED as defined above.
X-NI (six-set composed AIoU CI lo > -0.010, R2a): {"seed1": {"NOFM_W": true, "PEINIT_W": true}, "3seed": {"NOFM_W": true, "PEINIT_W": true}}

| evidence_label | block | cell | arm | comparator | set | n_train_seeds_arm | n_units | aiou_d | aiou_lo | penup_f1_v2_lo | strict_order_acc_lo | direction_acc_lo | U_WIN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SEED1_PLACEHOLDER | seed1 | R2a_U | NOFM_W | pooled_COMP_p01 | test_ouhdl | 1 | 4406 | 0.0468 | 0.0445 | 0.0609 | 0.0156 | -0.0078 | True |
| SEED1_PLACEHOLDER | seed1 | R2a_X_composed | NOFM_W | LUO_ms6 | test_casia_zh | 1 | 500 | 0.0379 | 0.0352 | 0.0482 | 0.0144 | -0.0037 | True |
| SEED1_PLACEHOLDER | seed1 | R2a_X_composed | NOFM_W | LUO_ms6 | test_casia_en | 1 | 500 | 0.0087 | 0.0068 | -0.0054 | -0.0136 | -0.0094 | False |
| SEED1_PLACEHOLDER | seed1 | R2a_X_composed | NOFM_W | LUO_ms6 | test_hpl_tamil | 1 | 500 | 0.0206 | 0.0182 | 0.0103 | -0.0052 | -0.0095 | True |
| SEED1_PLACEHOLDER | seed1 | R2a_X_composed | NOFM_W | LUO_ms6 | test_iam500 | 1 | 500 | 0.0154 | 0.013 | -0.0011 | -0.0208 | -0.017 | False |
| SEED1_PLACEHOLDER | seed1 | R2a_X_composed | NOFM_W | LUO_ms6 | german_unipen_de_test | 1 | 827 | 0.0194 | 0.0134 | -0.0431 | -0.0846 | -0.1629 | False |
| SEED1_PLACEHOLDER | seed1 | R2a_X_composed | NOFM_W | LUO_ms6 | german_onhw_all | 1 | 5925 | 0.0269 | 0.0261 | 0.0203 | 0.0041 | -0.0 | True |
| SEED1_PLACEHOLDER | seed1 | R2a_U | PEINIT_W | pooled_COMP_p01 | test_ouhdl | 1 | 4406 | 0.058 | 0.0557 | 0.0972 | 0.0382 | -0.0048 | True |
| SEED1_PLACEHOLDER | seed1 | R2a_X_composed | PEINIT_W | LUO_ms6 | test_casia_zh | 1 | 500 | 0.057 | 0.054 | 0.1987 | 0.1605 | 0.0029 | True |
| SEED1_PLACEHOLDER | seed1 | R2a_X_composed | PEINIT_W | LUO_ms6 | test_casia_en | 1 | 500 | 0.0131 | 0.0113 | 0.0171 | -0.0014 | -0.0042 | True |
| SEED1_PLACEHOLDER | seed1 | R2a_X_composed | PEINIT_W | LUO_ms6 | test_hpl_tamil | 1 | 500 | 0.0295 | 0.0269 | 0.0585 | 0.0073 | -0.0046 | True |
| SEED1_PLACEHOLDER | seed1 | R2a_X_composed | PEINIT_W | LUO_ms6 | test_iam500 | 1 | 500 | 0.0236 | 0.0208 | 0.0234 | 0.0014 | -0.009 | True |
| SEED1_PLACEHOLDER | seed1 | R2a_X_composed | PEINIT_W | LUO_ms6 | german_unipen_de_test | 1 | 827 | 0.0272 | 0.0163 | 0.0079 | -0.0388 | -0.1595 | False |
| SEED1_PLACEHOLDER | seed1 | R2a_X_composed | PEINIT_W | LUO_ms6 | german_onhw_all | 1 | 5925 | 0.0457 | 0.0449 | 0.0993 | 0.0438 | 0.0081 | True |
| SEED1_PLACEHOLDER | seed1 | R2b_STEPS_effect | LUOC40 | pooled_COMP_p01 | test_ouhdl | 1 | 4406 | 0.01 | 0.0087 | 0.0086 | -0.0027 | 0.0006 | True |
| SEED1_PLACEHOLDER | seed1 | R2b_ENCODER_ATTRIBUTED | NOFM_W | LUOC40 | test_ouhdl | 1 | 4406 | 0.0368 | 0.0349 | 0.0473 | 0.0128 | -0.0108 | True |
| SEED1_PLACEHOLDER | seed1 | R2b_ENCODER_ATTRIBUTED | PEINIT_W | LUOC40 | test_ouhdl | 1 | 4406 | 0.0479 | 0.046 | 0.084 | 0.0352 | -0.0082 | True |
| SEED1_PLACEHOLDER | seed1 | R2b_STEPS_effect_VPS | LUOC40VPS | pooled_COMP_p01 | test_ouhdl | 1 | 4406 | -0.0011 | -0.0024 | -0.0062 | -0.0048 | 0.0028 | False |
| SEED1_PLACEHOLDER | seed1 | R2b_VPS_vs_CLUSTER-A_LUOC40 | LUOC40VPS | LUOC40 | test_ouhdl | 1 | 4406 | -0.0112 | -0.0125 | -0.02 | -0.0074 | -0.0003 | False |
| SEED1_PLACEHOLDER | seed1 | R2b_ENCODER_ATTRIBUTED_VPS | NOFM_W | LUOC40VPS | test_ouhdl | 1 | 4406 | 0.048 | 0.0457 | 0.063 | 0.015 | -0.0137 | False |
| SEED1_PLACEHOLDER | seed1 | R2b_ENCODER_ATTRIBUTED_VPS | PEINIT_W | LUOC40VPS | test_ouhdl | 1 | 4406 | 0.0591 | 0.0568 | 0.0998 | 0.0381 | -0.0113 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_U_vs_LUOFT | NOFM_W | LUOFT_p10 | test_ouhdl | 1 | 4406 | 0.043 | 0.0411 | 0.0616 | 0.0244 | -0.0068 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_U_vs_pooled_COMP_p01 | NOFM_W | pooled_COMP_p01 | test_ouhdl | 1 | 4406 | 0.0711 | 0.0686 | 0.1369 | 0.0859 | 0.0094 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_casia_zh | 1 | 500 | 0.0932 | 0.0874 | 0.1222 | 0.0202 | -0.0089 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_casia_en | 1 | 500 | 0.0874 | 0.08 | 0.2566 | 0.1368 | 0.0383 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_hpl_tamil | 1 | 500 | 0.0962 | 0.0887 | 0.182 | 0.0747 | -0.0556 | False |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_iam500 | 1 | 500 | 0.0898 | 0.0834 | 0.242 | 0.115 | 0.013 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | german_unipen_de_test | 1 | 827 | 0.0895 | 0.0754 | 0.163 | 0.1033 | -0.0244 | False |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | german_onhw_all | 1 | 5925 | 0.0386 | 0.0374 | 0.0842 | 0.0331 | 0.0457 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_U_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_ouhdl | 1 | 4406 | 0.05 | 0.0481 | 0.0762 | 0.0262 | -0.0065 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_U_vs_pooled_COMP_p01 | PEINIT_W | pooled_COMP_p01 | test_ouhdl | 1 | 4406 | 0.0781 | 0.0757 | 0.1513 | 0.0895 | 0.0104 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_casia_zh | 1 | 500 | 0.1233 | 0.1154 | 0.2528 | 0.0579 | 0.0749 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_casia_en | 1 | 500 | 0.09 | 0.0818 | 0.2926 | 0.1632 | 0.0588 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_hpl_tamil | 1 | 500 | 0.1288 | 0.1192 | 0.3806 | 0.2155 | 0.093 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_iam500 | 1 | 500 | 0.0972 | 0.09 | 0.3343 | 0.145 | 0.0202 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | german_unipen_de_test | 1 | 827 | 0.0936 | 0.0789 | 0.25 | 0.1215 | -0.0012 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | german_onhw_all | 1 | 5925 | 0.0469 | 0.0456 | 0.1272 | 0.0221 | 0.0101 | True |
| SEED1_PLACEHOLDER | seed1 | R2c_p30_U_vs_pooled_COMP_p01 | NOFM_W | pooled_COMP_p01 | test_ouhdl | 1 | 4406 | 0.0796 | 0.0768 | 0.162 | 0.1132 | 0.0118 | True |
| CONFIRMATORY_3SEED | 3seed | R2a_U | NOFM_W | pooled_COMP_p01 | test_ouhdl | 3 | 4406 | 0.0469 | 0.0449 | 0.0675 | 0.0211 | -0.0072 | True |
| CONFIRMATORY_3SEED | 3seed | R2a_X_composed | NOFM_W | LUO_ms6 | test_casia_zh | 3 | 500 | 0.0379 | 0.0352 | 0.0482 | 0.0144 | -0.0037 | True |
| CONFIRMATORY_3SEED | 3seed | R2a_X_composed | NOFM_W | LUO_ms6 | test_casia_en | 3 | 500 | 0.0087 | 0.0068 | -0.0054 | -0.0136 | -0.0094 | False |
| CONFIRMATORY_3SEED | 3seed | R2a_X_composed | NOFM_W | LUO_ms6 | test_hpl_tamil | 3 | 500 | 0.0206 | 0.0181 | 0.0103 | -0.0052 | -0.0095 | True |
| CONFIRMATORY_3SEED | 3seed | R2a_X_composed | NOFM_W | LUO_ms6 | test_iam500 | 3 | 500 | 0.0154 | 0.013 | -0.0011 | -0.0208 | -0.017 | False |
| CONFIRMATORY_3SEED | 3seed | R2a_X_composed | NOFM_W | LUO_ms6 | german_unipen_de_test | 3 | 827 | 0.0186 | 0.0114 | -0.0531 | -0.0905 | -0.1754 | False |
| CONFIRMATORY_3SEED | 3seed | R2a_X_composed | NOFM_W | LUO_ms6 | german_onhw_all | 3 | 5925 | 0.0269 | 0.0262 | 0.0203 | 0.0041 | 0.0001 | True |
| CONFIRMATORY_3SEED | 3seed | R2a_U | PEINIT_W | pooled_COMP_p01 | test_ouhdl | 3 | 4406 | 0.0595 | 0.0573 | 0.1036 | 0.0421 | -0.0065 | True |
| CONFIRMATORY_3SEED | 3seed | R2a_X_composed | PEINIT_W | LUO_ms6 | test_casia_zh | 3 | 500 | 0.057 | 0.054 | 0.1987 | 0.1605 | 0.0029 | True |
| CONFIRMATORY_3SEED | 3seed | R2a_X_composed | PEINIT_W | LUO_ms6 | test_casia_en | 3 | 500 | 0.0131 | 0.0113 | 0.0171 | -0.0014 | -0.0042 | True |
| CONFIRMATORY_3SEED | 3seed | R2a_X_composed | PEINIT_W | LUO_ms6 | test_hpl_tamil | 3 | 500 | 0.0295 | 0.0269 | 0.0585 | 0.0073 | -0.0046 | True |
| CONFIRMATORY_3SEED | 3seed | R2a_X_composed | PEINIT_W | LUO_ms6 | test_iam500 | 3 | 500 | 0.0236 | 0.0208 | 0.0234 | 0.0014 | -0.009 | True |
| CONFIRMATORY_3SEED | 3seed | R2a_X_composed | PEINIT_W | LUO_ms6 | german_unipen_de_test | 3 | 827 | 0.0268 | 0.0156 | 0.0182 | -0.0281 | -0.1695 | False |
| CONFIRMATORY_3SEED | 3seed | R2a_X_composed | PEINIT_W | LUO_ms6 | german_onhw_all | 3 | 5925 | 0.0457 | 0.0449 | 0.0993 | 0.0438 | 0.0083 | True |
| NOT_ASSESSED | 3seed | R2b_STEPS_effect | LUOC40 | pooled_COMP_p01 | test_ouhdl | 2 | 4406 | 0.0141 | 0.0129 | 0.019 | 0.0016 | -0.0032 | True |
| NOT_ASSESSED | 3seed | R2b_ENCODER_ATTRIBUTED | NOFM_W | LUOC40 | test_ouhdl | 3 | 4406 | 0.0328 | 0.0312 | 0.0452 | 0.0152 | -0.0069 | True |
| NOT_ASSESSED | 3seed | R2b_ENCODER_ATTRIBUTED | PEINIT_W | LUOC40 | test_ouhdl | 3 | 4406 | 0.0453 | 0.0437 | 0.0815 | 0.0366 | -0.0057 | True |
| NOT_ASSESSED | 3seed | R2b_STEPS_effect_VPS | LUOC40VPS | pooled_COMP_p01 | test_ouhdl | 3 | 4406 | 0.0037 | 0.0028 | 0.0035 | 0.0006 | -0.002 | True |
| NOT_ASSESSED | 3seed | R2b_VPS_vs_CLUSTER-A_LUOC40 | LUOC40VPS | LUOC40 | test_ouhdl | 3 | 4406 | -0.0104 | -0.0115 | -0.0195 | -0.006 | -0.0012 | False |
| NOT_ASSESSED | 3seed | R2b_ENCODER_ATTRIBUTED_VPS | NOFM_W | LUOC40VPS | test_ouhdl | 3 | 4406 | 0.0432 | 0.0412 | 0.0603 | 0.0177 | -0.0082 | True |
| NOT_ASSESSED | 3seed | R2b_ENCODER_ATTRIBUTED_VPS | PEINIT_W | LUOC40VPS | test_ouhdl | 3 | 4406 | 0.0557 | 0.0537 | 0.0972 | 0.0391 | -0.0072 | True |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_U_vs_LUOFT | NOFM_W | LUOFT_p10 | test_ouhdl | 3 | 4406 | 0.0428 | 0.0412 | 0.061 | 0.0241 | -0.0072 | True |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_U_vs_pooled_COMP_p01 | NOFM_W | pooled_COMP_p01 | test_ouhdl | 3 | 4406 | 0.0721 | 0.0697 | 0.1405 | 0.0893 | 0.0067 | True |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_casia_zh | 3 | 500 | 0.0791 | 0.0743 | 0.1052 | 0.0144 | -0.0247 | False |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_casia_en | 3 | 500 | 0.0745 | 0.0684 | 0.2521 | 0.1464 | 0.0297 | True |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_hpl_tamil | 3 | 500 | 0.0842 | 0.0772 | 0.1763 | 0.0906 | -0.0594 | False |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_iam500 | 3 | 500 | 0.0769 | 0.071 | 0.2166 | 0.106 | 0.0011 | True |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | german_unipen_de_test | 3 | 827 | 0.0806 | 0.0666 | 0.1386 | 0.0849 | -0.0461 | False |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | german_onhw_all | 3 | 5925 | 0.0383 | 0.0372 | 0.0783 | 0.0337 | 0.0239 | True |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_U_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_ouhdl | 3 | 4406 | 0.0505 | 0.0489 | 0.078 | 0.0303 | -0.0072 | True |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_U_vs_pooled_COMP_p01 | PEINIT_W | pooled_COMP_p01 | test_ouhdl | 3 | 4406 | 0.0798 | 0.0775 | 0.1576 | 0.0963 | 0.0062 | True |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_casia_zh | 3 | 500 | 0.116 | 0.109 | 0.2489 | 0.0545 | 0.0649 | True |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_casia_en | 3 | 500 | 0.0769 | 0.0706 | 0.3002 | 0.1688 | 0.0414 | True |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_hpl_tamil | 3 | 500 | 0.1186 | 0.1099 | 0.3704 | 0.2324 | 0.0646 | True |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_iam500 | 3 | 500 | 0.085 | 0.0777 | 0.3011 | 0.1405 | -0.006 | True |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | german_unipen_de_test | 3 | 827 | 0.0862 | 0.0711 | 0.2194 | 0.1093 | -0.0251 | False |
| CONFIRMATORY_3SEED | 3seed | R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | german_onhw_all | 3 | 5925 | 0.0462 | 0.0451 | 0.1176 | 0.0253 | 0.0029 | True |
| NOT_ASSESSED | 3seed | R2c_p30_U_vs_pooled_COMP_p01 | NOFM_W | pooled_COMP_p01 | test_ouhdl | 1 | 4406 | 0.0811 | 0.0785 | 0.1672 | 0.1164 | 0.009 | True |

## Missing inputs
- LUO-FT p30 comparator: existing Route A M5 ft/p30 ms10 (4090 preds, one training seed) - read separately, not pooled here
- LUOC40/p10/german_onhw_all
- LUOC40/p10/german_unipen_de_test
- LUOC40/p10/test_casia_en
- LUOC40/p10/test_casia_zh
- LUOC40/p10/test_hpl_tamil
- LUOC40/p10/test_iam500
- LUOC40/p10/test_ouhdl
- LUOC40/p30/test_ouhdl
- NOFM_W2 FT
- PEINIT_W/p30/ft/test_ouhdl
- PEINIT_W2 FT
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
