"""The English text-recognition model for figure OCR: pinned, fetched once, checked (ADR 0018).

RapidOCR's wheel bundles a Chinese+English recognition model; ARIE swaps in the English
PP-OCRv4 one (95 characters: ASCII letters, digits, punctuation and space), keeping the
bundled detection and orientation models. The file is not in the wheel or in git: ``make
ocr-model`` (or ``make setup``, or the image build) downloads it once to
``ingestion.figure_ocr_model`` and checks its SHA-256. Without that file, or with another,
the runtime still starts and skips OCR with one warning (ADR 0019); each worker checks it
again before loading it.

usage: python -m arie_worker.ocr_model [path]   (default: models/ocr/en_PP-OCRv4_rec_infer.onnx)
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

REC_MODEL_URL = (
    "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.4.0/"
    "onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_infer.onnx"
)
REC_MODEL_SHA256 = "e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318"
DEFAULT_REC_MODEL = "models/ocr/en_PP-OCRv4_rec_infer.onnx"
# The pinned file is 7,653,044 bytes; anything far larger is not it.
MAX_MODEL_BYTES = 32 * 1024 * 1024

type Opener = Callable[[str], BinaryIO]


class OcrModelError(RuntimeError):
    """The OCR recognition model is missing or is not the pinned file."""


def verify_model(path: Path, sha256: str = REC_MODEL_SHA256) -> Path:
    """``path`` if it holds the pinned model; otherwise a message saying how to fix it."""
    if not path.is_file():
        raise OcrModelError(
            f"OCR model not found at {path}: run `make ocr-model` "
            "(or set ingestion.figure_ocr_model)"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != sha256:
        raise OcrModelError(
            f"OCR model at {path} has SHA-256 {digest}, expected {sha256}: "
            "delete it and run `make ocr-model`"
        )
    return path


def _open(url: str) -> BinaryIO:
    return urllib.request.urlopen(url, timeout=60)  # noqa: S310 - fixed https URL


def fetch_model(
    path: Path, url: str = REC_MODEL_URL, sha256: str = REC_MODEL_SHA256, opener: Opener = _open
) -> Path:
    """Download the model to ``path`` once; a file already there is only checked.

    The bytes are checked before anything is written, then renamed into place, so a failed
    or tampered download leaves no file behind.
    """
    if path.exists():
        return verify_model(path, sha256)
    with opener(url) as response:
        data = response.read(MAX_MODEL_BYTES + 1)
    digest = hashlib.sha256(data).hexdigest()
    if len(data) > MAX_MODEL_BYTES or digest != sha256:
        raise OcrModelError(f"download from {url} has SHA-256 {digest}, expected {sha256}")
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)
    return path


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REC_MODEL)
    try:
        fetch_model(path)
    except OcrModelError as exc:
        raise SystemExit(str(exc)) from None
    print(f"OCR model ready at {path} (SHA-256 {REC_MODEL_SHA256[:12]}…)")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_REC_MODEL",
    "REC_MODEL_SHA256",
    "REC_MODEL_URL",
    "OcrModelError",
    "fetch_model",
    "verify_model",
]
