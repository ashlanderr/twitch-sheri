import logging

import numpy as np

log = logging.getLogger(__name__)

# GigaAM за раз распознаёт не больше 25 секунд, нарезка фраз режет на 20.
MAX_SECONDS = 25


class GigaAMTranscriber:
    def __init__(self, model_name: str):
        import gigaam
        import torch

        self.torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Загружаю GigaAM %s на %s", model_name, device)
        self.model = gigaam.load_model(model_name, device=device)

    def transcribe(self, pcm: np.ndarray) -> str:
        pcm = pcm[: MAX_SECONDS * 16_000]
        # Как gigaam.preprocess.load_audio, только без ffmpeg и временного файла.
        wav = self.torch.from_numpy(pcm.astype(np.float32) / 32768.0)
        wav = wav.to(self.model._device).to(self.model._dtype).unsqueeze(0)
        length = self.torch.full([1], wav.shape[-1], device=self.model._device)
        with self.torch.inference_mode():
            encoded, encoded_len = self.model.forward(wav, length)
            text, _ = self.model._decode(encoded, encoded_len, length)[0]
        return text.strip()
