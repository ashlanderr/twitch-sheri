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

4. Запусти бота:

```bash
.venv/Scripts/python -m sheri twitch
```

Если озвучка фильтрует ботов, добавь аккаунт Шери в исключения.

## Устройство

- `prompts/system.md`: характер, правила и уровни доступа. `prompts/people.md`: кто есть кто.
- `sheri/agent.py`: агентный цикл и общая история. Она сохраняется в `data/history.json`.
- `sheri/tools.py`: инструменты и проверка прав при вызове.
- `sheri/memory.py`: заметки в `data/memory/`.
- `sheri/events.py`: события и очередь. Новый источник ввода просто кладёт `Event` в очередь.
