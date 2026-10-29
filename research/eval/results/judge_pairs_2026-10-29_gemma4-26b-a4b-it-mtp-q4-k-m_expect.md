# Парный слепой судья (gemma4:26b-a4b-it-mtp-q4_K_M, матожидание по логпробам A–T) · 2026-10-29_gemma4-26b-a4b-it-mtp-q4-k-m_expect

| пара | A | B | ходов | A | B | ничья | перевороты | нераспознано | вердикт | ожидание ядра | совпало |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e4b_base_vs_thinkall_s42 | ctx_probe_gemma4_e4b-it-qat_8192.json | ctx_probe_gemma4_e4b-it-qat_think-all_8192.json | 13 | 5 | 4 | 0 | 4 (0.31) | 0 | A | A | да |
| e4b_base_vs_thinkall_s7 | ctx_probe_gemma4_e4b-it-qat_s7_8192.json | ctx_probe_gemma4_e4b-it-qat_s7_think-all_8192.json | 13 | 6 | 5 | 0 | 2 (0.15) | 0 | A | A | да |
| e2b_base_vs_thinkall_s42 | ctx_probe_gemma4_e2b-it-qat_8192.json | ctx_probe_gemma4_e2b-it-qat_think-all_8192.json | 13 | 2 | 5 | 0 | 6 (0.46) | 0 | B | B | да |
| e2b_base_vs_thinkall_s7 | ctx_probe_gemma4_e2b-it-qat_s7_8192.json | ctx_probe_gemma4_e2b-it-qat_s7_think-all_8192.json | 13 | 5 | 2 | 0 | 6 (0.46) | 0 | A | tie | нет |
| qat_vs_q4km_noimx | ctx_probe_gemma4_e4b-it-qat_8192.json | ctx_probe_kostyan-e4b-q4km_8192.json | 13 | 7 | 1 | 0 | 5 (0.38) | 0 | A | A | да |
| qat_vs_iq4xs_imx | ctx_probe_gemma4_e4b-it-qat_8192.json | ctx_probe_kostyan-e4b-iq4xs-imx_8192.json | 13 | 6 | 4 | 0 | 3 (0.23) | 0 | A | A | да |
| base2k_vs_mem_v1 | ctx_probe_gemma4_e4b-it-qat_trim_2048.json | ctx_probe_gemma4_e4b-it-qat_mem-dropped-hybrid-k3_trim_2048.json | 13 | 4 | 2 | 5 | 2 (0.15) | 0 | A | A | да |
| base2k_vs_mem_router_v5 | ctx_probe_gemma4_e4b-it-qat_trim_2048.json | ctx_probe_gemma4_e4b-it-qat_v5_mem-dropped-hybrid-k3_rt-apply_trim_2048.json | 13 | 3 | 6 | 0 | 4 (0.31) | 0 | B | B | да |
| e4b_vs_e2b_v3 | ctx_probe_gemma4_e4b-it-qat_8192.json | ctx_probe_gemma4_e2b-it-qat_8192.json | 13 | 9 | 1 | 0 | 3 (0.23) | 0 | A | A | да |

Совпадение с вердиктами ядра: 8 / 9; средняя доля переворотов (позиционный сдвиг): 0.3

Средняя оценка E по паре (> 0 — лучше B): e4b_base_vs_thinkall_s42 -0.119, e4b_base_vs_thinkall_s7 -0.034, e2b_base_vs_thinkall_s42 0.215, e2b_base_vs_thinkall_s7 -0.275, qat_vs_q4km_noimx -0.438, qat_vs_iq4xs_imx -0.176, base2k_vs_mem_v1 -0.173, base2k_vs_mem_router_v5 0.218, e4b_vs_e2b_v3 -0.572
Время: 824 с на 234 запросов
