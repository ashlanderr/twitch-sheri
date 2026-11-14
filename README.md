# Шери

ИИ-кот для Twitch-чата. Модель работает через routerai.ru.

## Запуск

Скопируй `.env.example` в `.env` и укажи `ROUTERAI_API_KEY`.

Проверка без Twitch:

```bash
.venv/Scripts/python -m sheri console
```

Тесты:

```bash
.venv/Scripts/python -m pytest
```

Прогон сценария из `scripts/live_check.py` на настоящей модели, во временной папке:

```bash
.venv/Scripts/python scripts/live_check.py
```

Проверка характера на коротких сценах, каждая с чистой историей. Реплики в сценах не должны зависеть от конкретного ответа Шери, иначе контекст ломается. `heldout.json` — сцены, под которые промпт не подгонялся, `normal.json` — вежливые зрители, с которыми Шери не должен хамить:

```bash
.venv/Scripts/python scripts/eval_scenes.py evals/scenes.json --memory evals/memory
```

Прогон записи чата (JSON-список `{"t", "user", "text"}`): ответы прежнего бота выкидываются, Шери отвечает заново. Реплики людей там отвечают на старые ответы, так что годится только для грубой оценки:

```bash
.venv/Scripts/python scripts/replay_chat.py chat.json --from 0 --to 20 --memory evals/memory
```

## Подключение к Twitch

1. Создай отдельный аккаунт Twitch для бота и сделай его модератором канала: `/mod <ник_бота>`.
2. В [консоли разработчика](https://dev.twitch.tv/console/apps) создай приложение с типом клиента Public и включённым Device Code Flow. Укажи его Client ID в `TWITCH_CLIENT_ID`.
3. Войди в оба аккаунта. Команда покажет ссылку, её нужно открыть в браузере, где выполнен вход в нужный аккаунт:

```bash
.venv/Scripts/python -m sheri login bot
```

```bash
.venv/Scripts/python -m sheri login broadcaster
```

Если приложение в консоли зарегистрировать нельзя (там нужна 2FA), токены можно взять на [twitchtokengenerator.com](https://twitchtokengenerator.com): Custom Scope Token, для бота права `user:bot user:read:chat user:write:chat moderator:manage:banned_users` (вход под аккаунтом бота), для канала `channel:manage:broadcast` (вход под своим). Потом импортируй каждый, команда спросит Access Token и Refresh Token и подскажет `TWITCH_CLIENT_ID`. Такие токены обновляются через их сервер.

```bash
.venv/Scripts/python -m sheri import-token bot
```

```bash
.venv/Scripts/python -m sheri import-token broadcaster
```

4. Запусти бота:

```bash
.venv/Scripts/python -m sheri twitch
```

Если озвучка фильтрует ботов, добавь аккаунт Шери в исключения.

## Голос

Клиент на стрим-компе снимает микрофон, вырезает фразы с помощью Silero VAD и отправляет их на сервер. Сервер распознаёт их через GigaAM и передаёт агенту. Фраза с именем «Шери» запускает ход. Вместе с ней агент получает последние 90 секунд речи, а 20 секунд после обращения слушает реплики и без имени.

В `.env` на обеих машинах нужен одинаковый `VOICE_TOKEN`, любая длинная случайная строка.

Серверу нужны torch и GigaAM, около 3 ГБ. При первом запуске он скачает веса модели с Hugging Face в `models/gigaam/`, ещё около 430 МБ:

```bash
.venv/Scripts/python -m pip install -r requirements-server.txt
```

Сервер, вместе с ботом:

```bash
.venv/Scripts/python -m sheri twitch --voice
```

Клиент: сначала найди номер микрофона, потом подключись к серверу.

```bash
.venv/Scripts/python -m sheri voice-client --list-devices
```

```bash
.venv/Scripts/python -m sheri voice-client --server ws://<адрес_ноута>:8765 --device <номер>
```

Клиенту нужны только `numpy`, `sounddevice`, `onnxruntime`, `websockets` и `python-dotenv`, torch ему не нужен.

## Экран

Шери может посмотреть на стрим инструментом `look_at_screen`: сервер просит снимок у того же голосового клиента, а клиент берёт его из OBS. Снимается сцена, которая сейчас в эфире, а не мониторы, поэтому второй монитор с личными окнами Шери не видит.

В OBS включи Инструменты → Настройки сервера WebSocket, а пароль оттуда укажи на стрим-компе в `.env` как `OBS_PASSWORD`. Клиент сам подключается к `ws://127.0.0.1:4455`, другой адрес задаётся через `--obs`, ширина снимка через `--screen-width` (по умолчанию 1920), а `--no-screen` выключает снимки.

Снимок 1080p стоит около 2450 входных токенов, примерно 0,3 ₽ на `gpt-6.1-terra`. Картинка остаётся в истории только до конца хода, в `history.json` она не попадает.

Проверка распознавания на записи стрима:

```bash
.venv/Scripts/python scripts/transcribe_file.py запись.mp4 --start 10:00 --duration 600
```

## Устройство

- `prompts/system.md`: характер, правила и уровни доступа. `prompts/people.md`: кто есть кто.
- `sheri/agent.py`: агентный цикл и общая история. Она сохраняется в `data/history.json`.
- `sheri/tools.py`: инструменты и проверка прав при вызове.
- `sheri/memory.py`: заметки в `data/memory/`.
- `sheri/events.py`: события и очередь. Новый источник ввода просто кладёт `Event` в очередь.
