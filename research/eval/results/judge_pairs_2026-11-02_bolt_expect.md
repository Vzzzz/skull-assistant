# Парный слепой судья (gemma4:26b-a4b-it-q4_K_M, матожидание по логпробам A–T) · 2026-11-02 · T-055, основные пары

A — текущий прод (`kostyan-e4b-iq4xs-imx`), B — кандидат; E > 0 — лучше B. Восстановлено из лога запуска (файл того же имени перезаписан вторым запуском — `judge_pairs_2026-11-02_bolt-conv_expect.md`).

| пара | итог по устойчивым вердиктам |
|---|---|
| bolt_iq4_s1 | A 0 / B 3 / tie 2 / flip 8 / unparsed 0 -> B (expect None) |
| bolt_iq4_s2 | A 1 / B 5 / tie 2 / flip 5 / unparsed 0 -> B (expect None) |
| bolt_q4km_s1 | A 0 / B 4 / tie 1 / flip 8 / unparsed 0 -> B (expect None) |
| bolt_q4km_s2 | A 1 / B 4 / tie 2 / flip 6 / unparsed 0 -> B (expect None) |
| prod_s1_s2 | A 4 / B 0 / tie 2 / flip 7 / unparsed 0 -> A (expect None) |

Средняя оценка E по паре (> 0 — лучше B): bolt_iq4_s1 0.23, bolt_iq4_s2 0.34, bolt_q4km_s1 0.307, bolt_q4km_s2 0.218, prod_s1_s2 -0.34

Пары: `bolt_iq4_s1/s2` — прод против Bolt IQ4_XS (своя), сиды 1/2; `bolt_q4km_s1/s2` — прод против Bolt Q4_K_M (их); `prod_s1_s2` — прод против прода (шум судьи).

## Ходы

```
[bolt_iq4_s1] t01 0.464/1.0 -> B mass [1.0, 1.0] top ['T', 'A']
[bolt_iq4_s1] t05 -0.949/0.616 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_iq4_s1] t08 0.333/-0.999 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_iq4_s1] t10 -0.991/0.869 -> flip mass [1.0, 1.0] top ['A', 'B']
[bolt_iq4_s1] t13 0.971/-0.978 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_iq4_s1] t16 1.0/0.989 -> B mass [1.0, 1.0] top ['T', 'A']
[bolt_iq4_s1] t24 -0.776/1.0 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_iq4_s1] t28 0.0/-0.0 -> tie mass [1.0, 1.0] top ['K', 'K']
[bolt_iq4_s1] t31 -0.408/0.999 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_iq4_s1] t40 -0.227/0.984 -> flip mass [1.0, 1.0] top ['K', 'A']
[bolt_iq4_s1] t46 -0.827/0.991 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_iq4_s1] t59 0.907/1.0 -> B mass [1.0, 1.0] top ['T', 'A']
[bolt_iq4_s1] t60 -0.0/0.0 -> tie mass [1.0, 1.0] top ['K', 'K']
[bolt_iq4_s2] t01 0.998/-0.669 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_iq4_s2] t05 -0.599/0.998 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_iq4_s2] t08 -0.917/0.999 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_iq4_s2] t10 -0.385/-0.992 -> A mass [1.0, 1.0] top ['T', 'A']
[bolt_iq4_s2] t13 1.0/0.806 -> B mass [1.0, 1.0] top ['A', 'T']
[bolt_iq4_s2] t16 0.788/1.0 -> B mass [1.0, 1.0] top ['A', 'T']
[bolt_iq4_s2] t24 -0.0/0.0 -> tie mass [1.0, 1.0] top ['K', 'K']
[bolt_iq4_s2] t28 0.0/-0.0 -> tie mass [1.0, 1.0] top ['K', 'K']
[bolt_iq4_s2] t31 0.92/1.0 -> B mass [1.0, 1.0] top ['T', 'A']
[bolt_iq4_s2] t40 1.0/-0.997 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_iq4_s2] t46 0.993/1.0 -> B mass [1.0, 1.0] top ['T', 'A']
[bolt_iq4_s2] t59 0.998/1.0 -> B mass [1.0, 1.0] top ['T', 'A']
[bolt_iq4_s2] t60 -0.995/0.897 -> flip mass [1.0, 1.0] top ['A', 'B']
[bolt_q4km_s1] t01 0.418/1.0 -> B mass [1.0, 1.0] top ['T', 'A']
[bolt_q4km_s1] t05 0.999/-0.747 -> flip mass [1.0, 1.0] top ['A', 'B']
[bolt_q4km_s1] t08 -0.752/1.0 -> flip mass [1.0, 1.0] top ['B', 'A']
[bolt_q4km_s1] t10 0.993/-0.939 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_q4km_s1] t13 0.997/-0.999 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_q4km_s1] t16 1.0/1.0 -> B mass [1.0, 1.0] top ['T', 'A']
[bolt_q4km_s1] t24 1.0/-0.768 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_q4km_s1] t28 -0.0/0.0 -> tie mass [1.0, 1.0] top ['K', 'K']
[bolt_q4km_s1] t31 -0.946/0.999 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_q4km_s1] t40 -0.995/0.7 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_q4km_s1] t46 0.994/0.89 -> B mass [1.0, 1.0] top ['A', 'T']
[bolt_q4km_s1] t59 0.998/1.0 -> B mass [1.0, 1.0] top ['T', 'A']
[bolt_q4km_s1] t60 -0.807/0.937 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_q4km_s2] t01 0.507/-0.936 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_q4km_s2] t05 -0.995/0.661 -> flip mass [1.0, 1.0] top ['A', 'B']
[bolt_q4km_s2] t08 -0.746/1.0 -> flip mass [1.0, 1.0] top ['B', 'A']
[bolt_q4km_s2] t10 0.999/-0.84 -> flip mass [1.0, 0.999] top ['A', 'A']
[bolt_q4km_s2] t13 1.0/-0.815 -> flip mass [1.0, 1.0] top ['A', 'A']
[bolt_q4km_s2] t16 1.0/1.0 -> B mass [1.0, 1.0] top ['T', 'A']
[bolt_q4km_s2] t24 0.0/-0.0 -> tie mass [1.0, 1.0] top ['K', 'K']
[bolt_q4km_s2] t28 0.0/-0.0 -> tie mass [1.0, 1.0] top ['K', 'K']
[bolt_q4km_s2] t31 1.0/0.904 -> B mass [1.0, 1.0] top ['A', 'T']
[bolt_q4km_s2] t40 -0.947/-1.0 -> A mass [1.0, 1.0] top ['T', 'A']
[bolt_q4km_s2] t46 0.942/1.0 -> B mass [1.0, 1.0] top ['T', 'A']
[bolt_q4km_s2] t59 1.0/0.999 -> B mass [1.0, 1.0] top ['A', 'T']
[bolt_q4km_s2] t60 -0.993/0.918 -> flip mass [1.0, 1.0] top ['A', 'B']
[prod_s1_s2] t01 -0.916/1.0 -> flip mass [1.0, 1.0] top ['A', 'A']
[prod_s1_s2] t05 0.41/-0.999 -> flip mass [1.0, 1.0] top ['B', 'A']
[prod_s1_s2] t08 -1.0/-0.994 -> A mass [1.0, 1.0] top ['A', 'T']
[prod_s1_s2] t10 -0.989/0.886 -> flip mass [1.0, 1.0] top ['A', 'A']
[prod_s1_s2] t13 -1.0/-0.995 -> A mass [1.0, 1.0] top ['A', 'T']
[prod_s1_s2] t16 -0.953/0.395 -> flip mass [1.0, 1.0] top ['T', 'T']
[prod_s1_s2] t24 1.0/-0.94 -> flip mass [1.0, 1.0] top ['A', 'A']
[prod_s1_s2] t28 0.0/-0.0 -> tie mass [1.0, 1.0] top ['K', 'K']
[prod_s1_s2] t31 -0.846/-0.999 -> A mass [1.0, 1.0] top ['T', 'A']
[prod_s1_s2] t40 -1.0/0.997 -> flip mass [1.0, 1.0] top ['A', 'A']
[prod_s1_s2] t46 -0.976/-1.0 -> A mass [1.0, 1.0] top ['T', 'A']
[prod_s1_s2] t59 0.0/-0.0 -> tie mass [1.0, 1.0] top ['K', 'K']
[prod_s1_s2] t60 0.993/-0.915 -> flip mass [1.0, 1.0] top ['A', 'B']
```
