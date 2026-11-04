from pathlib import Path

import numpy as np
import onnxruntime

from .segmenter import FRAME, SAMPLE_RATE

MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "silero_vad.onnx"
# Silero ждёт перед кадром 64 отсчёта конца предыдущего кадра.
CONTEXT = 64


class SileroVAD:
    """Вероятность речи для кадра 512 отсчётов, 16 кГц. Работает на процессоре без torch."""

    def __init__(self, path: Path = MODEL_PATH):
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, CONTEXT), dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def __call__(self, frame: np.ndarray) -> float:
        if len(frame) != FRAME:
            raise ValueError(f"нужен кадр {FRAME} отсчётов, пришло {len(frame)}")
        samples = (frame.astype(np.float32) / 32768.0).reshape(1, -1)
        window = np.concatenate([self._context, samples], axis=1)
        output, self._state = self.session.run(
            None, {"input": window, "state": self._state, "sr": self._sr}
        )
        self._context = window[:, -CONTEXT:]
        return float(output[0, 0])
