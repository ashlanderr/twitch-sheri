import hashlib
import json
import logging
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# GigaAM за раз распознаёт не больше 25 секунд, нарезка фраз режет на 20.
MAX_SECONDS = 25

# У CDN Сбера, откуда качает пакет gigaam, истёк сертификат. Те же веса лежат
# на Hugging Face, их берём оттуда: ревизия закреплена, файлы сверяются по sha256.
HF_URL = "https://huggingface.co/ai-sage/GigaAM-v3/resolve"
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "gigaam"
MODELS = {
    "v3_e2e_rnnt": ("7655ad717f8122257385bb4b2f373db3697e8680", {
        "config.json": "02361ba9cafd6c3ec66fcdd73494c3b562a60eb2a2d1b13f3cb04ae440d93e52",
        "pytorch_model.bin": "afc6dcbae8320ea56f2cddebc0f13fbf62c9d59b6ddcad899782623c8610826a",
        "tokenizer.model": "828c12c991019eef952a960661f25a92d6ad279591e2ea466b4aeddf1d20a18a",
    }),
    "v3_e2e_ctc": ("cec030b4c4f35d928e4a9044a3bdb29ebd499fac", {
        "config.json": "3dbcd2ebe1a572cb9a9945eeb3faf8b59180d408e61a61ecda44db30421e7251",
        "pytorch_model.bin": "9801f83ef3979779982c4f83ec1c234d7571e8c1a5aa2bfc0b09ceb97a16cd75",
        "tokenizer.model": "0b9a1960898fbfdf5424ab852ea17445eb3da960fba23e977ff100eb0054fbc8",
    }),
    "v3_rnnt": ("c7f128b8accdd9624df905e5c2d7b7a48c27c0d8", {
        "config.json": "ca2998f00f064648988111c426deaa4c8bc86872a8bbe443598f2b1511dc0947",
        "pytorch_model.bin": "a89b662052ad2b8129b334ab45c171b04fc6699d69bd6917575013ff79172ec2",
    }),
    "v3_ctc": ("15ef3b5a88da78f93134b3cb7f015c70aefa8946", {
        "config.json": "5db9b9e3f8c1a6193f3e262857cbaa75f4fd7653990de846162ad7318f61bdb5",
        "pytorch_model.bin": "64d9c925df4bfd57e19f7b26d001106f9c5a35bd088a71c5bde071bf92c4a559",
    }),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(model_name: str) -> Path:
    if model_name not in MODELS:
        raise ValueError(f"нет модели {model_name}, есть: {', '.join(MODELS)}")
    revision, files = MODELS[model_name]
    folder = MODELS_DIR / model_name
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in files.items():
        path = folder / name
        if path.exists():
            continue
        log.info("Скачиваю %s/%s", model_name, name)
        partial = path.with_name(name + ".part")
        with urllib.request.urlopen(f"{HF_URL}/{revision}/{name}") as source, partial.open("wb") as output:
            while chunk := source.read(1 << 20):
                output.write(chunk)
        if sha256(partial) != expected:
            partial.unlink()
            raise RuntimeError(f"{model_name}/{name}: sha256 не совпал")
        partial.rename(path)
    return folder


def load_gigaam(folder: Path, device: str):
    import gigaam
    import torch
    from gigaam import decoder, decoding, encoder, preprocess
    from omegaconf import OmegaConf

    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))["cfg"]["model"]["cfg"]
    # Конфиг с Hugging Face ссылается на свой modeling_gigaam.py, подменяем на классы пакета.
    modules = (preprocess, encoder, decoder, decoding)
    for part in ("preprocessor", "encoder", "head", "decoding"):
        name = cfg[part]["_target_"].removeprefix("modeling_gigaam.")
        module = next(module for module in modules if hasattr(module, name))
        cfg[part]["_target_"] = f"{module.__name__}.{name}"
    if "model_path" in cfg["decoding"]:
        cfg["decoding"]["model_path"] = str(folder / "tokenizer.model")
    if device == "cpu":
        cfg["encoder"]["flash_attn"] = False

    model = gigaam.GigaAMASR(OmegaConf.create(cfg))
    # weights_only: в отличие от .ckpt из пакета, тут при загрузке не выполняется чужой код.
    state = torch.load(folder / "pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()})
    return model.eval().to(device)


class GigaAMTranscriber:
    def __init__(self, model_name: str):
        import torch

        self.torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        folder = download_model(model_name)
        log.info("Загружаю GigaAM %s на %s", model_name, device)
        self.model = load_gigaam(folder, device)

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
