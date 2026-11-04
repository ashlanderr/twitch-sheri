"""Распознаёт запись тем же путём, что и живой микрофон: VAD, нарезка фраз, GigaAM.

    .venv/Scripts/python scripts/transcribe_file.py запись.mp4 [--start 10:00] [--duration 600]

Пишет таблицу рядом с записью: время, длина фразы, прозвучало ли имя Шери, текст."""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from sheri.config import load_config
from sheri.hearing import WAKE_RE
from sheri.voice.segmenter import FRAME, SAMPLE_RATE, Segmenter
from sheri.voice.stt import GigaAMTranscriber
from sheri.voice.vad import SileroVAD


def read_audio(path: Path, start: str | None, duration: float | None) -> np.ndarray:
    command = ["ffmpeg", "-v", "error"]
    if start:
        command += ["-ss", start]
    command += ["-i", str(path)]
    if duration:
        command += ["-t", str(duration)]
    command += ["-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    return np.frombuffer(subprocess.run(command, capture_output=True, check=True).stdout, dtype=np.int16)


def timestamp(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02}:{seconds:02}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--start", help="с какого места, например 10:00")
    parser.add_argument("--duration", type=float, help="сколько секунд распознать")
    args = parser.parse_args()

    audio = read_audio(args.path, args.start, args.duration)
    print(f"Звук: {len(audio) / SAMPLE_RATE / 60:.1f} мин")
    vad, segmenter = SileroVAD(), Segmenter()
    transcriber = GigaAMTranscriber(load_config().stt_model)

    output = args.path.with_suffix(".transcript.tsv")
    started = time.perf_counter()
    speech_seconds = 0.0
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["time", "seconds", "wake", "text"])
        for offset in range(0, len(audio) - FRAME + 1, FRAME):
            frame = audio[offset : offset + FRAME]
            segment = segmenter.push(frame, vad(frame))
            if segment is None:
                continue
            seconds = len(segment) / SAMPLE_RATE
            speech_seconds += seconds
            text = transcriber.transcribe(segment)
            began = (offset + FRAME) / SAMPLE_RATE - seconds
            wake = "да" if WAKE_RE.search(text) else ""
            writer.writerow([timestamp(began), f"{seconds:.1f}", wake, text])
            print(f"[{timestamp(began)}] {'* ' if wake else ''}{text}")

    elapsed = time.perf_counter() - started
    print(f"\nРечи {speech_seconds / 60:.1f} мин, обработано за {elapsed / 60:.1f} мин. Таблица: {output}")


if __name__ == "__main__":
    main()
