from collections import deque

import numpy as np

SAMPLE_RATE = 16_000
FRAME = 512  # 32 мс: такой кадр ждёт Silero VAD при 16 кГц
FRAME_MS = FRAME * 1000 // SAMPLE_RATE


def _frames(ms: float) -> int:
    return max(1, round(ms / FRAME_MS))


class Segmenter:
    """Режет поток кадров на фразы по вероятности речи от VAD.

    Фраза начинается, когда вероятность выше threshold, и кончается после
    min_silence_ms ниже release. Небольшой запас тишины по краям сохраняется,
    чтобы не обрезать первые и последние звуки."""

    def __init__(
        self,
        threshold: float = 0.5,
        release: float = 0.35,
        min_silence_ms: float = 600,
        pad_ms: float = 200,
        min_speech_ms: float = 250,
        max_speech_s: float = 20,
    ):
        self.threshold = threshold
        self.release = release
        self.min_silence = _frames(min_silence_ms)
        self.pad = _frames(pad_ms)
        self.min_speech = _frames(min_speech_ms)
        self.max_frames = _frames(max_speech_s * 1000)
        self._preroll: deque[np.ndarray] = deque(maxlen=self.pad)
        self._frames: list[np.ndarray] = []
        self._in_speech = False
        self._silence = 0
        self._voiced = 0

    def push(self, frame: np.ndarray, probability: float) -> np.ndarray | None:
        """Возвращает готовую фразу (int16), когда она закончилась."""
        if not self._in_speech:
            if probability >= self.threshold:
                self._in_speech = True
                self._frames = [*self._preroll, frame]
                self._preroll.clear()
                self._silence = 0
                self._voiced = 1
            else:
                self._preroll.append(frame)
            return None

        self._frames.append(frame)
        if probability < self.release:
            self._silence += 1
        else:
            self._silence = 0
            self._voiced += 1

        if self._silence >= self.min_silence:
            return self._finish(drop_tail=self._silence - self.pad)
        if len(self._frames) >= self.max_frames:
            # Длинный монолог режется, чтобы распознавание не ждало конца.
            segment = self._finish(drop_tail=0)
            self._in_speech = True
            return segment
        return None

    def _finish(self, drop_tail: int) -> np.ndarray | None:
        frames = self._frames[: len(self._frames) - drop_tail] if drop_tail > 0 else self._frames
        voiced = self._voiced
        self._frames, self._in_speech, self._silence, self._voiced = [], False, 0, 0
        if voiced < self.min_speech:
            return None
        return np.concatenate(frames)
