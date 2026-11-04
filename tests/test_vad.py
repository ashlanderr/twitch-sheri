import numpy as np
import pytest

from sheri.voice.segmenter import FRAME
from sheri.voice.vad import SileroVAD


def test_silence_is_not_speech():
    vad = SileroVAD()
    probabilities = [vad(np.zeros(FRAME, dtype=np.int16)) for _ in range(30)]
    assert max(probabilities) < 0.1


def test_wrong_frame_size_is_rejected():
    with pytest.raises(ValueError):
        SileroVAD()(np.zeros(FRAME - 1, dtype=np.int16))
