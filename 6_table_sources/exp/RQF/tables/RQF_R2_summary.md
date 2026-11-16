# RQF R2 tables (facts only)
No multiplicity correction beyond the prereg (Holm over 2 arms for R2a is applied by the reader). Rule flags: U_WIN, ORDER_NI3, ENCODER_ATTRIBUTED as defined above.
X-NI (six-set composed AIoU CI lo > -0.010, R2a): {"NOFM_W": true, "PEINIT_W": true}

| cell | arm | comparator | set | n_train_seeds_arm | n_units | aiou_d | aiou_lo | penup_f1_v2_lo | strict_order_acc_lo | direction_acc_lo | U_WIN |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R2a_U | NOFM_W | pooled_COMP_p01 | test_ouhdl | 2 | 4406 | 0.0481 | 0.0461 | 0.0682 | 0.0223 | -0.0089 | True |
| R2a_X_composed | NOFM_W | LUO_ms6 | test_casia_zh | 2 | 500 | 0.0379 | 0.0352 | 0.0482 | 0.0144 | -0.0037 | True |
| R2a_X_composed | NOFM_W | LUO_ms6 | test_casia_en | 2 | 500 | 0.0087 | 0.0068 | -0.0054 | -0.0136 | -0.0094 | False |
| R2a_X_composed | NOFM_W | LUO_ms6 | test_hpl_tamil | 2 | 500 | 0.0205 | 0.0181 | 0.0103 | -0.0052 | -0.0095 | True |
| R2a_X_composed | NOFM_W | LUO_ms6 | test_iam500 | 2 | 500 | 0.0154 | 0.013 | -0.0011 | -0.0208 | -0.017 | False |
| R2a_X_composed | NOFM_W | LUO_ms6 | german_unipen_de_test | 2 | 827 | 0.0193 | 0.0129 | -0.0443 | -0.0852 | -0.1694 | False |
| R2a_X_composed | NOFM_W | LUO_ms6 | german_onhw_all | 2 | 5925 | 0.0269 | 0.0262 | 0.0203 | 0.0041 | 0.0001 | True |
| R2a_U | PEINIT_W | pooled_COMP_p01 | test_ouhdl | 3 | 4406 | 0.0595 | 0.0573 | 0.1036 | 0.0421 | -0.0065 | True |
| R2a_X_composed | PEINIT_W | LUO_ms6 | test_casia_zh | 3 | 500 | 0.057 | 0.054 | 0.1987 | 0.1605 | 0.0029 | True |
| R2a_X_composed | PEINIT_W | LUO_ms6 | test_casia_en | 3 | 500 | 0.0131 | 0.0113 | 0.0171 | -0.0014 | -0.0042 | True |
| R2a_X_composed | PEINIT_W | LUO_ms6 | test_hpl_tamil | 3 | 500 | 0.0295 | 0.0269 | 0.0585 | 0.0073 | -0.0046 | True |
| R2a_X_composed | PEINIT_W | LUO_ms6 | test_iam500 | 3 | 500 | 0.0236 | 0.0208 | 0.0234 | 0.0014 | -0.009 | True |
| R2a_X_composed | PEINIT_W | LUO_ms6 | german_unipen_de_test | 3 | 827 | 0.0268 | 0.0156 | 0.0182 | -0.0281 | -0.1695 | False |
| R2a_X_composed | PEINIT_W | LUO_ms6 | german_onhw_all | 3 | 5925 | 0.0457 | 0.0449 | 0.0993 | 0.0438 | 0.0083 | True |
| R2c_p10_U_vs_LUOFT | NOFM_W | LUOFT_p10 | test_ouhdl | 1 | 4406 | 0.0434 | 0.0417 | 0.062 | 0.0242 | -0.0072 | True |
| R2c_p10_U_vs_pooled_COMP_p01 | NOFM_W | pooled_COMP_p01 | test_ouhdl | 1 | 4406 | 0.0727 | 0.0702 | 0.1422 | 0.0892 | 0.0066 | True |
| R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_casia_zh | 1 | 500 | 0.0795 | 0.0742 | 0.1045 | 0.0143 | -0.0334 | False |
| R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_casia_en | 1 | 500 | 0.0743 | 0.0678 | 0.2481 | 0.1443 | 0.0173 | True |
| R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_hpl_tamil | 1 | 500 | 0.0816 | 0.0739 | 0.1644 | 0.0885 | -0.083 | False |
| R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | test_iam500 | 1 | 500 | 0.0772 | 0.0713 | 0.2185 | 0.1088 | -0.0092 | True |
| R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | german_unipen_de_test | 1 | 827 | 0.0811 | 0.0673 | 0.154 | 0.0917 | -0.0443 | False |
| R2c_p10_X_FTnoRouter_vs_LUOFT | NOFM_W | LUOFT_p10 | german_onhw_all | 1 | 5925 | 0.0385 | 0.0375 | 0.0793 | 0.034 | 0.0327 | True |
| R2c_p10_U_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_ouhdl | 2 | 4406 | 0.0506 | 0.049 | 0.0785 | 0.0321 | -0.0077 | True |
| R2c_p10_U_vs_pooled_COMP_p01 | PEINIT_W | pooled_COMP_p01 | test_ouhdl | 2 | 4406 | 0.0799 | 0.0775 | 0.1579 | 0.0979 | 0.0057 | True |
| R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_casia_zh | 2 | 500 | 0.1192 | 0.112 | 0.2538 | 0.0554 | 0.0672 | True |
| R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_casia_en | 2 | 500 | 0.0769 | 0.0704 | 0.3052 | 0.1677 | 0.0445 | True |
| R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_hpl_tamil | 2 | 500 | 0.1208 | 0.1118 | 0.3723 | 0.2335 | 0.0649 | True |
| R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | test_iam500 | 2 | 500 | 0.0852 | 0.0776 | 0.2953 | 0.1393 | -0.0096 | True |
| R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | german_unipen_de_test | 2 | 827 | 0.0866 | 0.0711 | 0.2112 | 0.1035 | -0.027 | False |
| R2c_p10_X_FTnoRouter_vs_LUOFT | PEINIT_W | LUOFT_p10 | german_onhw_all | 2 | 5925 | 0.0459 | 0.0448 | 0.1152 | 0.0262 | 0.0071 | True |
| R2c_p30_U_vs_pooled_COMP_p01 | NOFM_W | pooled_COMP_p01 | test_ouhdl | 1 | 4406 | 0.0811 | 0.0785 | 0.1672 | 0.1164 | 0.009 | True |

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
- NOFM_W2 FT
- PEINIT_W/p30/ft/test_ouhdl
- PEINIT_W2 FT
- R2b_ENCODER_ATTRIBUTED:NOFM_W vs LUOC40:test_ouhdl
- R2b_ENCODER_ATTRIBUTED:PEINIT_W vs LUOC40:test_ouhdl
- R2b_STEPS_effect:LUOC40 vs pooled_COMP_p01:test_ouhdl
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
