"""Human-review report for a CTX-1 run: judged turns with judge score + empty manual column,
all memory probes with the automatic verdict, math and hallucination items.

Usage: python3 research/eval/ctx_review.py research/eval/results/ctx_probe_gemma3_4b_8192.json
Writes <same name>_review.md next to the JSON (markdown is tracked by git).
"""
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
d = json.loads(src.read_text(encoding="utf-8"))
log, m, j = d["log"], d["metrics"], d.get("judge") or {}
by_turn = {e["turn"]: e for e in log}
jp = {p["turn"]: p for p in j.get("persona", [])}

out = [f"# Ручная валидация CTX-1 · {d['model']} · num_ctx {d['ctx']}", "",
       "Заполните колонку «Моя оценка» (1–5) и «Комментарий». Оценка судьи дана для сравнения, не как эталон.", "",
       "## 1. Персона (оценённые судьёй реплики)", "",
       "| Ход | Реплика пользователя | Ответ модели | Судья | Почему (судья) | Моя оценка | Комментарий |",
       "|---|---|---|---|---|---|---|"]
for e in log:
    if e.get("judge_persona"):
        p = jp.get(e["turn"], {})
        rep = e["reply"].replace("\n", " ").replace("|", "¦")
        out.append(f"| {e['turn']} | {e['text'].replace('|', '¦')} | {rep[:300]} | {p.get('score', '—')} | {p.get('why', p.get('raw', ''))} |  |  |")

out += ["", "## 2. Память: пробы и автоматический вердикт", "",
        "Вердикт «✓» = в ответе найдено ожидаемое ключевое слово (и нет запрещённого). Проверьте, не ошибся ли автомат: ответ может быть верным другими словами или неверным при совпадении слова.", "",
        "| Ход | Проба | Расстояние | Вопрос | Ответ модели | Ожидалось | Авто | Верно на самом деле? |",
        "|---|---|---|---|---|---|---|---|"]
for r in m["recall"]:
    e = by_turn[r["turn"]]
    exp = e.get("expect", {})
    expected = ", ".join(exp.get("keywords", [])) + (f" (+{', '.join(exp['keywords2'])})" if "keywords2" in exp else "") + (f" [не: {', '.join(exp['must_not'])}]" if "must_not" in exp else "")
    rep = e["reply"].replace("\n", " ").replace("|", "¦")
    out.append(f"| {r['turn']} | {r['probe_for']} | {r['distance']} | {e['text'].replace('|', '¦')} | {rep[:200]} | {expected} | {'✓' if r['hit'] else '✗'} |  |")

out += ["", "## 3. Математика", "", "| Задача | Вопрос | Ответ модели | Эталон | Авто: верно | Выкладка есть | Комментарий |", "|---|---|---|---|---|---|---|"]
for x in m["math"]:
    e = next(e for e in log if e.get("math_id") == x["math_id"])
    out.append(f"| {x['math_id']} | {e['text']} | {e['reply'].replace(chr(10), ' ')[:200]} | {e['expect']['answer']} | {'✓' if x['correct'] else '✗'} | {'да' if x['has_working'] else 'нет'} |  |")

l1 = next((e for e in log if e.get("longgen_id") == "L1"), None)
l2 = next((e for e in log if e.get("hallucination_probe")), None)
l3 = next((e for e in log if e.get("longgen_id") == "L3"), None)
out += ["", "## 4. Длинная генерация и галлюцинация", ""]
if l2:
    out += [f"**L2, вымышленный прибор.** Судья: `{j.get('hallucination')}`; авто-оговорка: {m['longgen'].get('L2_hedged')}", "", "> " + l2["reply"].replace("\n", "\n> "), "", "Моя оценка (выдумал / оговорился / отказался): ", ""]
if l1 and l3:
    out += [f"**L1 → L3, согласованность пересказа.** Судья: `{j.get('consistency')}`; L1 длина {l1['eval_count']} токенов", "",
            "L1:", "", "> " + l1["reply"].replace("\n", "\n> "), "", "L3:", "", "> " + l3["reply"].replace("\n", "\n> "), "", "Моя оценка согласованности (1–5): ", ""]

dst = src.with_name(src.stem + "_review.md")
dst.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"review: {dst}")
