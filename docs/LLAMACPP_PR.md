# PR в llama.cpp: `cache_reuse` на мультимодальном сервере для промптов без медиа (T-059)

Это **материал для автора PR** (владелец), на русском: что сделано, почему, чем доказано, как оформить. Сам текст PR, ответы ревьюерам и любые посты в треде автор пишет сам, по-английски — политика llama.cpp (`CONTRIBUTING.md`, «AI Usage Policy»: код с помощью ИИ разрешён при явном раскрытии; использовать ИИ для написания постов и описаний PR запрещено, нарушение — бан). Числа в §3 заполняет исполнитель по T-059; до этого строки помечены «(T-059)».

## 1. Проблема одним абзацем

`llama-server` с загруженным проектором (`--mmproj`) **отключает `--cache-reuse` для всех запросов**, даже если ни в одном промпте нет ни картинки, ни аудио. Для голосовых ассистентов это типичный случай: проектор нужен редко (кадр по событию), а текстовый диалог идёт постоянно, и каждый раз, когда меняется середина промпта (скользящее окно истории, вставка заметки памяти), сервер пересчитывает всё после общего префикса. На Jetson Orin Nano 8 ГБ с Gemma 4 E4B это 5 секунд на 3k токенов против 0.1 с при совпадении префикса (наш замер T-056). Issue #21468 («cache reuse is not supported for Gemma 4») закрыт PR #22288, но в треде зафиксировано, что с mmproj reuse по-прежнему выключен — и так и осталось.

## 2. Где это в коде (master, 2026-11-03)

- `tools/server/server-context.cpp`, загрузка модели (~строки 1404–1412): если `mctx` создан, `params_base.ctx_shift = false` и `params_base.n_cache_reuse = 0` с предупреждениями «… is not supported by multimodal, it will be disabled».
- там же, обработка промпта (~4123–4160): `can_cache_reuse = llama_memory_can_shift(...) && !slot.prompt.tokens.has_mtmd;` — `has_mtmd` истинно для **любого** `server_tokens`, созданного на сервере с проектором, независимо от содержимого; внутри ветки `if (mctx) GGML_ABORT("not supported by multimodal")`.
- `tools/server/server-common.h`: `server_tokens` хранит `map_idx_to_media` — карту «индекс → медиа-чанк»; медиа-позиции в `tokens` помечены `LLAMA_TOKEN_NULL`. `get_common_prefix` уже умеет сравнивать медиа-чанки по id — то есть инфраструктура для различения «есть ли медиа в промпте» в классе есть, а проверка на уровне reuse сделана по флагу сервера.
- `src/llama-kv-cache-iswa.cpp`: `get_can_shift()` для моделей с SWA требует равных размеров base и swa кэшей → для Gemma 4 reuse (как и `ctx_shift`) работает только с `--swa-full`. Это не часть правки, но важно для замера (цена по памяти) и для текста PR.

## 3. Что меняется

1. `server_tokens::has_media() const` → `!map_idx_to_media.empty()` (промпт содержит хотя бы один медиа-чанк).
2. При загрузке mmproj **не обнулять** `n_cache_reuse`; вместо предупреждения — информационная строка, что reuse будет применяться только к промптам без медиа. `ctx_shift` не трогать (другой механизм, отдельный вопрос).
3. В обработке промпта: `can_cache_reuse = llama_memory_can_shift(mem) && !slot.prompt.tokens.has_media() && !input_tokens.has_media();` при `n_cache_reuse > 0` и пропуске из-за медиа — `SLT_WRN` с причиной «prompt contains media»; `GGML_ABORT` по `mctx` убрать (теперь ветка достижима при mmproj, но только для текстовых промптов).
4. В цикле сопоставления чанков считать `LLAMA_TOKEN_NULL` несовпадением (защита: два плейсхолдера разных картинок равны по значению токена) — при условии 3 недостижимо, но делает код честным.
5. `tools/server/README.md`: в строке `--cache-reuse` — «with a multimodal projector loaded, applies only to prompts without media chunks».
6. Тест `tools/server/tests/unit/test_cache_reuse_mtmd.py` (образец — `test_vision_api.py`, `ServerPreset.tinygemma3()`): (а) текстовый промпт A, затем B = A без фрагмента в середине; при `cache_reuse=32` `timings.prompt_n` у B меньше, чем при `cache_reuse=0`, ответ при `temperature=0` совпадает; (б) промпт с картинкой, затем текстовый — ответ корректен, reuse пропущен. До правки (а) падает, после — проходит; `test_vision_api.py` и `test_ctx_shift.py` остаются зелёными.

Что **не** делается: reuse через медиа-чанки (сдвиг позиций картинки), `ctx_shift` для мультимодальных, изменения в `mtmd`.

## 4. Чем доказано (заполняет T-059)

| Сценарий | Без правки | С правкой |
|---|---|---|
| Стенд (RTX 3080 Laptop), E4B IQ4_XS + mmproj, `--cache-reuse 256 --swa-full`, промпт 3k → тот же без абзаца в середине: токенов заново / TTFT | 1187 / 0.35 с («cache_reuse is not supported by multimodal») | **1 / 0.04 с** |
| То же без mmproj (контроль: reuse уже работает) | 1 / 0.05 с | 1 / 0.04 с |
| Диалог со сдвигом окна @2048 (CTX-1, 60 ходов, 23–26 со сдвигом), TTFT на ходах со сдвигом, медиана / макс | 0.25 / 0.29 с | **0.05 / 0.05 с** (ход без сдвига — 0.03 с) |
| **Orin Nano Super 8 ГБ**, E4B IQ4_XS + mmproj, `--cache-reuse 256 --swa-full`, промпт 3k → тот же без абзаца в середине: токенов заново / TTFT | 1187 / 1.90 с (полный префилл 3k — 4.5 с; T-056: 5.1 с при резидентном голосе) | **1 / 0.21 с** |
| **Orin**: диалог со сдвигом окна @2048 (CTX-1, 60 ходов, 23–28 со сдвигом), TTFT на ходах со сдвигом, медиана / макс | 1.20 / 1.40 с | **0.25 / 0.28 с** (ход без сдвига — 0.10 с) |
| Цена `--swa-full` на Orin (пик nvmap, unified-память) | без: 4340–4349 МБ | с: 4444–4448 МБ (**+95…108 МБ**) |
| Цена `--swa-full` для Gemma 4 E4B @8k, KV q8_0 (пик VRAM, стенд) | без: 3868 МБ | с: 3956–3970 МБ (**+88…102 МБ**) |
| Корректность (тест на tinygemma3): ответ мультимодального сервера с reuse = ответу текстового сервера с тем же reuse | — | совпадают; на master тест падает (`prompt_n` 69 вместо 1) |

Сводка — `research/bench/results/llamacpp_cache_reuse_2026-11-03.md` (стенд, §1–5; §6 — числа для PR); строки Orin — `research/bench/results/llamacpp_cache_reuse_orin_2026-11-04.md` (T-059 шаг 4 на устройстве, 2026-11-04: до — прод платы 098ce52, после — форк 2d41788; там же гейт переезда прода платы на форк, шаг 6). В PR — только таблица и ссылка на репозиторий проекта (он публичный). Ветка: `Vzzzz/llama.cpp` `server-cache-reuse-mtmd`, коммит `2d41788` поверх master `9309bf1` — перед отправкой rebase.

### 4a. Таблица чисел для PR (данные из сводок; прозу вокруг неё пишет автор)

| Scenario | before | with the patch |
|---|---|---|
| Jetson Orin Nano Super 8 GB (JetPack 6.2.1, sm_87), Gemma 4 E4B IQ4_XS + mmproj, `--cache-reuse 256 --swa-full`: 3k-token text prompt, then the same prompt with one paragraph removed in the middle — tokens re-evaluated / prompt time | 1187 / 1.90 s (full 3k prefill: 4.5 s) | 1 / 0.21 s |
| Orin: 60-turn dialog with a sliding token window @2048 (23–28 turns shift the window), prompt time on the shifted turns, median / max | 1.20 / 1.40 s | 0.25 / 0.28 s (unshifted turn: 0.10 s) |
| RTX 3080 Laptop 8 GB (sm_86), same model and flags, the edit scenario | 1187 / 0.35 s | 1 / 0.04 s |
| RTX 3080 Laptop, shifted turns, median / max | 0.25 / 0.29 s | 0.05 / 0.05 s |
| Control: same server without `--mmproj` (reuse already works), the edit scenario, RTX 3080 | 1 / 0.05 s | 1 / 0.04 s |
| Cost of `--swa-full` (required for KV shifting with iSWA, Gemma 4), E4B @8k ctx, KV q8_0 | RTX 3080: +88…102 MB VRAM; Orin: +95…108 MB (unified, nvmap) | — |

Примечания для автора: «before» на стенде — master `9309bf1` (база патча); на Orin «before» — прод-сборка платы `098ce52` (master от 2026-10-2x), «with the patch» — форк `2d41788`; это стоит сказать в описании одной фразой. Лог «before»: `cache_reuse is not supported by multimodal, it will be disabled`; «with the patch»: `cache_reuse with multimodal: enabled for prompts without media`. Полные сводки: `research/bench/results/llamacpp_cache_reuse_2026-11-03.md` (стенд) и `llamacpp_cache_reuse_orin_2026-11-04.md` (Orin), в репозитории проекта.

## 5. Как оформить PR, по шагам

1. **Форк и ветка.** На GitHub: Fork `ggml-org/llama.cpp` → `Vzzzz/llama.cpp`. В WSL: `git clone git@github.com:Vzzzz/llama.cpp.git ~/llama.cpp-pr && cd ~/llama.cpp-pr && git remote add upstream https://github.com/ggml-org/llama.cpp.git && git fetch upstream && git checkout -b server-cache-reuse-mtmd upstream/master`. Исполнитель делает правку и тест в этой ветке (T-059 шаги 2–3) и коммитит; автор коммитов — вы (`git config user.name/email` как в проекте).
2. **Проверки перед отправкой:** `cmake -B build -DLLAMA_BUILD_SERVER=ON && cmake --build build -j` (CPU-сборки достаточно для тестов); `cd tools/server/tests && ./tests.sh unit/test_cache_reuse_mtmd.py unit/test_vision_api.py unit/test_ctx_shift.py`; `clang-format` (v15+) только на добавленных строках; `git rebase upstream/master` непосредственно перед push.
3. **Push:** `git push -u origin server-cache-reuse-mtmd`. На GitHub — «Compare & pull request», base `ggml-org:master`.
4. **Заголовок PR** в формате проекта: `server : enable cache_reuse for multimodal prompts without media (#21468)`.
5. **Описание PR — пишете сами, по-английски.** Что в нём должно быть (структура, не текст): проблема (с цитатой предупреждения из лога и ссылкой на #21468 и комментарий про mmproj); причина (флаг `has_mtmd` на уровне сервера вместо проверки содержимого промпта); правка (пункты §3 своими словами); тест и как его запустить; таблица §4; ограничения (`--swa-full` для SWA-моделей, медиа-промпты не затронуты, `ctx_shift` не трогается); **раскрытие ИИ** — факты, которые надо сообщить: код и тест написаны с помощью Claude Code (Anthropic) под вашим руководством в рамках вашего проекта, вы проверили каждую строку, сборку и тесты запускали сами, замеры сделаны на вашем железе; описание и обсуждение — ваши. Формулировку выберите свою.
6. **После открытия:** дождаться CI (server-тесты идут в GitHub Actions), на замечания отвечать самостоятельно; при просьбе изменить — исполнитель правит в той же ветке, вы проверяете и пушите; если PR залежался — rebase на master и короткое напоминание в треде. Ссылку на PR — в `docs/ROADMAP.md` §4.3 и README («Как это делается»).

## 6. Использовать до мержа (и если мержа не будет)

Да: прод стенда и Orin собирается из нашего форка (ветка `server-cache-reuse-mtmd` поверх свежего master, коммит фиксируется в `deploy/build_llama.sh` и D-033). Условия: (1) гейт переезда — master ушёл далеко от нашего 098ce52, поэтому CTX-1 через шину и три точки памяти `skull-llm` на стенде и Orin перемеряются и не должны стать хуже; (2) `--swa-full` для Gemma 4 — только если его цена по памяти влезает в запас Orin (945 МБ), иначе reuse на Gemma 4 ждёт upstream-решения по iSWA; (3) правило D-033: после мержа в upstream — обратно на master, патч снимается; (4) в README «Как это делается» и в описании PR — честно: патч используется в проде проекта на обоих машинах, это и есть наш «тест в поле». Так PR ещё и подкреплён эксплуатацией, а не только бенчмарком.

## 7. Почему это хороший PR для мейнтейнеров

Маленький, локальный, с тестом, который падает до и проходит после; закрывает хвост известного issue; мотивирован реальным устройством (Jetson) с цифрами; ничего не ломает для промптов с медиа — для них поведение прежнее (reuse пропущен с понятным предупреждением вместо молчаливого отключения на весь сервер).
