# Browser AI Agent

Автономный AI-агент для управления видимым браузером через OpenAI Responses API, Playwright и LangGraph.

Проект приведён к требованиям тестового задания:
- видимый браузер, управляемый агентом;
- задача передаётся текстом через терминал;
- агент работает автономно до завершения задачи или запроса данных у пользователя;
- поддерживаются persistent sessions;
- есть security layer для реально рискованных действий;
- есть sub-agent architecture: `planner`, `checkpoint planner`, `executor`, `evaluator`, `summarizer`;
- orchestration построен на `LangGraph StateGraph`, а не на одном хрупком ручном loop;
- есть управление контекстом через локальную память и compaction;
- нет хардкода под конкретные сайты, маршруты, селекторы или сценарии.

## Архитектура

`executor`
- основной агент, который видит скриншоты и управляет браузером.

`planner`
- строит краткий тактический план по задаче и текущей памяти.

`checkpoint planner`
- раскладывает задачу на внешне проверяемые checkpoints.

`evaluator`
- возвращает структурированный verdict в JSON и ловит drift, dead ends, wrong state и readiness to finish.

`summarizer`
- периодически сжимает историю выполнения, чтобы не раздувать контекст.

`langgraph state graph`
- управляет переходами `bootstrap -> plan -> execute -> evaluate -> recover -> done`.
- делает orchestration явной и проверяемой.

## Production-oriented guardrails

- запрет на guessed deep links;
- запрет на внешний web search внутри уже открытого web-app по умолчанию;
- ручной handoff на CAPTCHA, login и реально рискованные действия;
- confirm-before-done: первое `done()` всегда проходит через верификацию;
- structured evaluator + deterministic policy engine вместо хрупкого парсинга свободного текста;
- graph-based orchestration вместо разросшегося ручного while-loop;
- trace logging каждого запуска в `.agent_runs/*.jsonl`;
- динамический step budget с мягким и жёстким лимитом;
- repeat guard и soft accept для избежания бесконечных verification loops.

## Regression Evals

Есть локальный eval harness для policy layer:

```powershell
py -3 -m agent.evals
```

Он проверяет:
- парсинг structured evaluator output;
- dead-end classification;
- soft accept для address-like checkpoints.

## Основные Модули

- `agent/graph_runtime.py` — LangGraph orchestration runtime
- `agent/policy.py` — deterministic policy layer
- `agent/subagents.py` — planner/checkpoint/evaluator/summarizer roles
- `agent/browser_tools.py` — browser execution layer
- `agent/memory.py` — context memory with history compaction
- `agent/llm.py` — OpenAI Responses API client
- `agent/prompts.py` — system prompts for executor and sub-agents
- `agent/evals.py` — local regression tests for policy layer

## Запуск

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
py -3 -m playwright install
```

Настройте `.env`:

```env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.4-mini
HEADLESS=0
SLOW_MO_MS=100
MAX_STEPS=60
HARD_MAX_STEPS=120
STEP_EXTENSION=20
COMPACT_EVERY_STEPS=16
KEEP_BROWSER_OPEN=1
STARTUP_HELPER_TIMEOUT=30
ALLOW_EXTERNAL_SEARCH=0
TRACE_DIR=.agent_runs
PERSIST_SESSION=0
USE_CHROME=0
CHROME_PROFILE=
CHROME_PROFILE_DIRECTORY=
```

Запуск:

```powershell
py -3 -m agent
```

## Persistent Session

Если указать `CHROME_PROFILE`, агент запустит persistent browser context и сможет продолжить работу после ручного логина пользователя в том же профиле.

Пример:

```env
CHROME_PROFILE=C:\Users\<username>\AppData\Local\Google\Chrome\User Data
CHROME_PROFILE_DIRECTORY=Profile 1
```

## Как смотреть результаты

- браузер остаётся открытым после успешного завершения, если `KEEP_BROWSER_OPEN=1`;
- подробная трасса запуска пишется в `TRACE_DIR`;
- по trace-файлу можно разбирать, где агент drift-ил, какие tools вызывал и что видел evaluator.

## Ограничения

- CAPTCHA и ручной login требуют вмешательства пользователя;
- реальные оплаты, удаления, credential changes и security-sensitive шаги требуют подтверждения;
- без рабочего OpenAI API key рантайм не запустится;
- production-grade качество зависит от модели, стабильности целевого сайта и качества eval-сценариев.

## Что показывать в демо

- постановку задачи текстом;
- visible browser и автономный прогресс агента;
- ручной handoff только там, где это реально нужно;
- финальное состояние страницы;
- trace-файл как инженерный артефакт воспроизводимости.
