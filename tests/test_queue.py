import asyncio

from sheri.events import Event, EventQueue, Role

# Шаг больше таймера Windows (около 15 мс), чтобы тесты не зависели от его точности.
DEBOUNCE = 0.2
MAX_WAIT = 1.0


def viewer(user="vasya", text="привет"):
    return Event("test", Role.VIEWER, user, text)


def moderator(user="ashllanderr", text="команда"):
    return Event("test", Role.MODERATOR, user, text)


def broadcaster(user="Mo0nlaght", text="команда"):
    return Event("test", Role.BROADCASTER, user, text)


async def feed(queue, events, gap):
    for event in events:
        queue.put(event)
        await asyncio.sleep(gap)


def texts(batch):
    return [event.text for event in batch]


async def test_phrase_split_into_messages_becomes_one_batch():
    # Окно отсчитывается от последнего сообщения, а не от первого:
    # пять сообщений идут дольше одного окна, но без пауз длиннее окна.
    queue = EventQueue(DEBOUNCE, MAX_WAIT, 20)
    words = ["точки", "я", "не", "ставлю", "вот"]
    asyncio.create_task(feed(queue, [viewer(text=word) for word in words], DEBOUNCE / 2))
    assert texts(await queue.next_batch()) == words


async def test_pause_longer_than_debounce_closes_batch():
    queue = EventQueue(DEBOUNCE, MAX_WAIT, 20)
    asyncio.create_task(feed(queue, [viewer(text="раз"), viewer(text="два")], DEBOUNCE * 2))
    assert texts(await queue.next_batch()) == ["раз"]
    assert texts(await queue.next_batch()) == ["два"]


async def test_batch_closes_after_max_wait_even_if_chat_never_pauses():
    queue = EventQueue(DEBOUNCE, MAX_WAIT, 100)
    events = [viewer(text=str(number)) for number in range(30)]
    asyncio.create_task(feed(queue, events, DEBOUNCE / 2))
    loop = asyncio.get_running_loop()
    started = loop.time()
    first = await queue.next_batch()
    assert loop.time() - started < MAX_WAIT + DEBOUNCE
    assert 3 <= len(first) < len(events)
    second = await queue.next_batch()
    assert texts(first + second) == texts(events[: len(first) + len(second)])


async def test_batch_respects_max_size():
    queue = EventQueue(DEBOUNCE, MAX_WAIT, 3)
    for number in range(5):
        queue.put(viewer(text=str(number)))
    assert texts(await queue.next_batch()) == ["0", "1", "2"]
    assert texts(await queue.next_batch()) == ["3", "4"]


async def test_different_viewers_share_batch():
    queue = EventQueue(DEBOUNCE, MAX_WAIT, 20)
    for user in ("a", "b", "c"):
        queue.put(viewer(user=user))
    assert [event.user for event in await queue.next_batch()] == ["a", "b", "c"]


async def test_moderator_messages_in_a_row_become_one_batch():
    queue = EventQueue(DEBOUNCE, MAX_WAIT, 20)
    words = ["Шери", "забань", "васю"]
    asyncio.create_task(feed(queue, [moderator(text=word) for word in words], DEBOUNCE / 2))
    batch = await queue.next_batch()
    assert texts(batch) == words
    assert {event.user for event in batch} == {"ashllanderr"}


async def test_privileged_author_is_never_mixed_with_others():
    queue = EventQueue(DEBOUNCE, MAX_WAIT, 20)
    for event in (
        viewer(user="a"), viewer(user="b"),
        moderator(text="м1"), moderator(text="м2"),
        viewer(user="c"),
        moderator(user="other_mod", text="о1"),
        broadcaster(text="л1"), broadcaster(text="л2"),
        moderator(text="м3"),
    ):
        queue.put(event)
    batches = [[(event.user, event.text) for event in await queue.next_batch()] for _ in range(6)]
    assert batches == [
        [("a", "привет"), ("b", "привет")],
        [("ashllanderr", "м1"), ("ashllanderr", "м2")],
        [("c", "привет")],
        [("other_mod", "о1")],
        [("Mo0nlaght", "л1"), ("Mo0nlaght", "л2")],
        [("ashllanderr", "м3")],
    ]


async def test_held_event_starts_next_batch_and_keeps_order():
    queue = EventQueue(DEBOUNCE, MAX_WAIT, 20)
    queue.put(moderator(text="м1"))
    queue.put(viewer(text="з1"))
    queue.put(viewer(text="з2"))
    assert texts(await queue.next_batch()) == ["м1"]
    assert texts(await queue.next_batch()) == ["з1", "з2"]
