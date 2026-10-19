# CTX-1 · openbmb/minicpm-v4.6:latest · 2026-10-19

| num_ctx | пустых ответов | recall (все) | recall (≥30) | math верно | math с выкладкой | L1 токенов | L2 оговорка | обрезка на ходах | персона (судья, 1–5) | согласованность L1/L3 | L2 выдумал | TTFT посл.10, с | tok/s | токенов в контексте к концу |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8192 | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 352 | False | — | 1 | 5 | True | 0.37 | 181.9 | 4121 |

Recall по расстоянию (ход/проба@расстояние): ✓ попал, ✗ нет; для M-проб ✓ = повторил свой же ранний ответ.

- ctx 8192: t19/A1@17=✗, t20/A5@9=✗, t34/A2@30=✗, t35/A4@26=✗, t36/M2@14=✗, t49/A6@35=✗, t50/A3u@25=✗, t51/A5@40=✗, t52/A1@50=✗, t53/L1@37=✗, t54/T1@53=✗, t55/M3@32=✗, t56/T41@15=✗, t57/A4@48=✗, t58/L2@41=✗

Математика (ответ модели, первые 120 символов):

- ctx 8192: M1: 'When we multiply seventeen by twenty-three, the result is four hundred nineteen.' · M2: 'If the item costs two thousand four hundred and then we apply a fifteen percent discount, the amount you need to pay is ' · M3: 'If you leave at eight forty-five and travel for an additional thirty-five minutes, you will arrive at ten twenty.'

Оценки персоны по ходам:

- ctx 8192: t1=1, t5=1, t8=1, t10=1, t13=1, t16=1, t24=1, t28=1, t31=1, t40=1, t46=1, t59=1, t60=1
