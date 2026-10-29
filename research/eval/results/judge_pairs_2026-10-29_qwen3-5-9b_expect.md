# Парный слепой судья (qwen3.5:9b, матожидание по логпробам A–T) · 2026-10-29_qwen3-5-9b_expect

| пара | A | B | ходов | A | B | ничья | перевороты | нераспознано | вердикт | ожидание ядра | совпало |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b_base_vs_thinkall_s42 | ctx_probe_gemma4_e4b-it-qat_8192.json | ctx_probe_gemma4_e4b-it-qat_think-all_8192.json | 13 | 2 | 1 | 10 | 0 (0.0) | 0 | A | A | да |
| e4b_base_vs_thinkall_s7 | ctx_probe_gemma4_e4b-it-qat_s7_8192.json | ctx_probe_gemma4_e4b-it-qat_s7_think-all_8192.json | 13 | 2 | 1 | 9 | 1 (0.08) | 0 | A | A | да |
| e2b_base_vs_thinkall_s42 | ctx_probe_gemma4_e2b-it-qat_8192.json | ctx_probe_gemma4_e2b-it-qat_think-all_8192.json | 13 | 1 | 2 | 9 | 1 (0.08) | 0 | B | B | да |
| e2b_base_vs_thinkall_s7 | ctx_probe_gemma4_e2b-it-qat_s7_8192.json | ctx_probe_gemma4_e2b-it-qat_s7_think-all_8192.json | 13 | 2 | 0 | 9 | 2 (0.15) | 0 | A | tie | нет |
| qat_vs_q4km_noimx | ctx_probe_gemma4_e4b-it-qat_8192.json | ctx_probe_kostyan-e4b-q4km_8192.json | 13 | 4 | 1 | 6 | 2 (0.15) | 0 | A | A | да |
| qat_vs_iq4xs_imx | ctx_probe_gemma4_e4b-it-qat_8192.json | ctx_probe_kostyan-e4b-iq4xs-imx_8192.json | 13 | 2 | 0 | 10 | 1 (0.08) | 0 | A | A | да |
| base2k_vs_mem_v1 | ctx_probe_gemma4_e4b-it-qat_trim_2048.json | ctx_probe_gemma4_e4b-it-qat_mem-dropped-hybrid-k3_trim_2048.json | 13 | 2 | 0 | 11 | 0 (0.0) | 0 | A | A | да |
| base2k_vs_mem_router_v5 | ctx_probe_gemma4_e4b-it-qat_trim_2048.json | ctx_probe_gemma4_e4b-it-qat_v5_mem-dropped-hybrid-k3_rt-apply_trim_2048.json | 13 | 1 | 1 | 11 | 0 (0.0) | 0 | tie | B | нет |
| e4b_vs_e2b_v3 | ctx_probe_gemma4_e4b-it-qat_8192.json | ctx_probe_gemma4_e2b-it-qat_8192.json | 13 | 4 | 1 | 6 | 2 (0.15) | 0 | A | A | да |

Совпадение с вердиктами ядра: 7 / 9; средняя доля переворотов (позиционный сдвиг): 0.08

Средняя оценка E по паре (> 0 — лучше B): e4b_base_vs_thinkall_s42 -0.004, e4b_base_vs_thinkall_s7 0.001, e2b_base_vs_thinkall_s42 0.014, e2b_base_vs_thinkall_s7 -0.013, qat_vs_q4km_noimx -0.048, qat_vs_iq4xs_imx -0.026, base2k_vs_mem_v1 -0.001, base2k_vs_mem_router_v5 0.0, e4b_vs_e2b_v3 -0.066
Время: 146 с на 234 запросов
