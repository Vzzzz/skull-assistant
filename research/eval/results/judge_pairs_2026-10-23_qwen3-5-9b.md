# Парный слепой судья (qwen3.5:9b) · 2026-10-23

| пара | A | B | ходов | A | B | ничья | перевороты | нераспознано | вердикт | ожидание ядра | совпало |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b_base_vs_thinkall_s42 | ctx_probe_gemma4_e4b-it-qat_8192.json | ctx_probe_gemma4_e4b-it-qat_think-all_8192.json | 13 | 2 | 1 | 0 | 10 (0.77) | 0 | A | A | да |
| e4b_base_vs_thinkall_s7 | ctx_probe_gemma4_e4b-it-qat_s7_8192.json | ctx_probe_gemma4_e4b-it-qat_s7_think-all_8192.json | 13 | 1 | 4 | 0 | 8 (0.62) | 0 | B | A | нет |
| e2b_base_vs_thinkall_s42 | ctx_probe_gemma4_e2b-it-qat_8192.json | ctx_probe_gemma4_e2b-it-qat_think-all_8192.json | 13 | 0 | 1 | 1 | 11 (0.85) | 0 | B | B | да |
| e2b_base_vs_thinkall_s7 | ctx_probe_gemma4_e2b-it-qat_s7_8192.json | ctx_probe_gemma4_e2b-it-qat_s7_think-all_8192.json | 13 | 1 | 1 | 0 | 11 (0.85) | 0 | tie | tie | да |
| qat_vs_q4km_noimx | ctx_probe_gemma4_e4b-it-qat_8192.json | ctx_probe_kostyan-e4b-q4km_8192.json | 13 | 2 | 3 | 0 | 8 (0.62) | 0 | B | A | нет |
| qat_vs_iq4xs_imx | ctx_probe_gemma4_e4b-it-qat_8192.json | ctx_probe_kostyan-e4b-iq4xs-imx_8192.json | 13 | 2 | 2 | 0 | 9 (0.69) | 0 | tie | A | нет |
| base2k_vs_mem_v1 | ctx_probe_gemma4_e4b-it-qat_trim_2048.json | ctx_probe_gemma4_e4b-it-qat_mem-dropped-hybrid-k3_trim_2048.json | 13 | 2 | 2 | 5 | 4 (0.31) | 0 | tie | A | нет |
| base2k_vs_mem_router_v5 | ctx_probe_gemma4_e4b-it-qat_trim_2048.json | ctx_probe_gemma4_e4b-it-qat_v5_mem-dropped-hybrid-k3_rt-apply_trim_2048.json | 13 | 2 | 2 | 0 | 9 (0.69) | 0 | tie | B | нет |
| e4b_vs_e2b_v3 | ctx_probe_gemma4_e4b-it-qat_8192.json | ctx_probe_gemma4_e2b-it-qat_8192.json | 13 | 4 | 2 | 0 | 7 (0.54) | 0 | A | A | да |

Совпадение с вердиктами ядра: 4 / 9; средняя доля переворотов (позиционный сдвиг): 0.66
