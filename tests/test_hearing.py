import numpy as np
import pytest

from sheri.events import Role
from sheri.hearing import WAKE_RE, Hearing
from sheri.voice.segmenter import FRAME, Segmenter


class FakeQueue:
    def __init__(self):
        self.events = []

    def put(self, event):
        self.events.append(event)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_hearing(context=90, followup=20):
    queue, clock = FakeQueue(), Clock()
    return Hearing(queue, "Mo0nlaght", context, followup, clock), queue, clock


@pytest.mark.parametrize("text", [
    "Шери, привет", "шерри поменяй название", "Шэри ты тут?", "ну Шерик", "Шеричка моя",
    "эй, ШЕРИ!", "Sherry, hi", "черри? cherry",
])
def test_wake_word_variants(text):
    assert WAKE_RE.search(text)


@pytest.mark.parametrize("text", [
    "шерсть везде", "у нас шеринг экрана", "прошерил", "привет всем", "шерифа не видели",
])
def test_not_wake_word(text):
    assert not WAKE_RE.search(text)


def test_speech_without_name_is_buffered_not_sent():
    hearing, queue, _ = make_hearing()
    hearing.hear("так, сейчас босс будет")
    hearing.hear("  ")
    assert queue.events == []


def test_name_sends_buffer_as_room_then_voice():
    hearing, queue, clock = make_hearing()
    hearing.hear("так, сейчас босс будет")
    clock.now = 5
    hearing.hear("кто-нибудь знает как его бить")
    clock.now = 10
    hearing.hear("Шери, загугли тактику")
    assert [(e.source, e.text) for e in queue.events] == [
        ("room", "так, сейчас босс будет"),
        ("room", "кто-нибудь знает как его бить"),
        ("voice", "Шери, загугли тактику"),
    ]
    assert {(e.role, e.user) for e in queue.events} == {(Role.BROADCASTER, "Mo0nlaght")}


def test_buffer_is_not_resent():
    hearing, queue, clock = make_hearing(followup=5)
    hearing.hear("раз")
    hearing.hear("Шери, привет")
    clock.now = 100
    hearing.hear("Шери, ещё раз")
    assert [e.text for e in queue.events] == ["раз", "Шери, привет", "Шери, ещё раз"]


def test_old_speech_falls_out_of_context():
    hearing, queue, clock = make_hearing(context=90)
    hearing.hear("давно сказано")
    clock.now = 60
    hearing.hear("недавно сказано")
    clock.now = 120
    hearing.hear("Шери, что я говорила?")
    assert [e.text for e in queue.events] == ["недавно сказано", "Шери, что я говорила?"]


def test_followup_without_name_is_addressed_and_window_extends():
    hearing, queue, clock = make_hearing(followup=20)
    hearing.hear("Шери, забань Васю")
    clock.now = 15
    hearing.hear("да")
    clock.now = 30
    hearing.hear("и Петю тоже")
    assert [e.source for e in queue.events] == ["voice", "voice", "voice"]


def test_followup_window_closes():
    hearing, queue, clock = make_hearing(followup=20)
    hearing.hear("Шери, привет")
    clock.now = 25
    hearing.hear("так, что там в игре")
    assert [e.text for e in queue.events] == ["Шери, привет"]


def frame(value=1000):
    return np.full(FRAME, value, dtype=np.int16)


def feed(segmenter, probabilities):
    segments = []
    for index, probability in enumerate(probabilities):
        result = segmenter.push(frame(index), probability)
        if result is not None:
            segments.append(result)
    return segments


def test_segment_has_padding_and_ends_after_silence():
    # Кадр 32 мс: тишина 600 мс = 19 кадров, запас 200 мс = 6 кадров.
    segmenter = Segmenter()
    probabilities = [0.0] * 10 + [0.9] * 20 + [0.0] * 19
    [segment] = feed(segmenter, probabilities)
    first_frame = segment[0]
    assert first_frame == 10 - 6
    assert len(segment) == (6 + 20 + 6) * FRAME


def test_short_pause_does_not_split_phrase():
    segmenter = Segmenter()
    probabilities = [0.9] * 20 + [0.0] * 10 + [0.9] * 20 + [0.0] * 19
    assert len(feed(segmenter, probabilities)) == 1


def test_hysteresis_keeps_phrase_between_thresholds():
    segmenter = Segmenter()
    probabilities = [0.9] * 10 + [0.4] * 30 + [0.0] * 19
    [segment] = feed(segmenter, probabilities)
    assert len(segment) == (10 + 30 + 6) * FRAME


def test_click_shorter_than_min_speech_is_dropped():
    segmenter = Segmenter()
    assert feed(segmenter, [0.9] * 3 + [0.0] * 30) == []


def test_long_monologue_is_cut():
    segmenter = Segmenter(max_speech_s=1)
    segments = feed(segmenter, [0.9] * 100 + [0.0] * 19)
    assert len(segments) >= 3
    assert all(len(segment) <= 32 * FRAME for segment in segments)
