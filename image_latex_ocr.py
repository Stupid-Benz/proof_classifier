"""Helpers for converting uploaded equation images to editable LaTeX with pix2tex.

The actual model is loaded lazily by the Streamlit app so normal text-only use
continues to work even before the optional OCR dependency/checkpoints are ready.
"""
from __future__ import annotations

from io import BytesIO
from typing import Any, Iterable

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}


class LatexOCRInputError(ValueError):
    """Raised when an uploaded image cannot safely be passed to pix2tex."""


def open_image_bytes(image_bytes: bytes) -> Image.Image:
    """Validate image bytes and return an EXIF-corrected RGB copy."""
    if not image_bytes:
        raise LatexOCRInputError("The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise LatexOCRInputError(
            f"Image is too large ({len(image_bytes) / (1024 * 1024):.1f} MB). "
            f"Please upload an image smaller than {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )

    try:
        with Image.open(BytesIO(image_bytes)) as source:
            image_format = (source.format or "").upper()
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise LatexOCRInputError(
                    f"Unsupported image format '{image_format or 'unknown'}'. "
                    "Use PNG, JPEG, WEBP, BMP or TIFF."
                )
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise LatexOCRInputError(
                    f"Image dimensions {width}×{height} are too large. "
                    f"Please use an image under {MAX_IMAGE_PIXELS:,} pixels."
                )
            corrected = ImageOps.exif_transpose(source)
            corrected.load()
            return corrected.convert("RGB")
    except LatexOCRInputError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise LatexOCRInputError(f"Could not read the uploaded image: {exc}") from exc


def clean_latex_prediction(prediction: Any) -> str:
    """Normalize pix2tex's result to a non-empty editable LaTeX string."""
    if prediction is None:
        raise LatexOCRInputError("LaTeX-OCR returned no result.")
    latex = str(prediction).strip()
    if not latex:
        raise LatexOCRInputError("LaTeX-OCR returned an empty result.")
    return latex


def recognize_latex_from_bytes(image_bytes: bytes, model: Any) -> str:
    """Run a supplied pix2tex-compatible model on one uploaded image.

    `model` is injected to keep this module testable without loading model
    weights. In the application it is an instance of `pix2tex.cli.LatexOCR`.
    """
    image = open_image_bytes(image_bytes)
    try:
        return clean_latex_prediction(model(image))
    except LatexOCRInputError:
        raise
    except Exception as exc:  # model/backend exceptions need a user-readable message
        raise RuntimeError(f"LaTeX-OCR failed to recognize this image: {exc}") from exc


def format_latex_block(latex: str) -> str:
    """Wrap a bare OCR expression as a display-math block, if needed."""
    value = clean_latex_prediction(latex)
    if (
        (value.startswith(r"\[") and value.endswith(r"\]"))
        or (value.startswith(r"\(") and value.endswith(r"\)"))
        or (value.startswith("$$") and value.endswith("$$"))
    ):
        return value
    if value.startswith("$") and value.endswith("$"):
        value = value[1:-1].strip()
    return f"\\[\n{value}\n\\]"


def combine_latex_predictions(predictions: Iterable[tuple[str, str]]) -> str:
    """Combine (filename, LaTeX) predictions into separate display-math blocks."""
    blocks = []
    for _filename, latex in predictions:
        blocks.append(format_latex_block(latex))
    return "\n\n".join(blocks)


def append_latex_to_text(existing_text: str, latex_text: str) -> str:
    """Append reviewed LaTeX without overwriting or duplicating the text."""
    current = (existing_text or "").rstrip()
    addition = (latex_text or "").strip()
    if not addition:
        raise LatexOCRInputError("There is no recognized LaTeX to insert.")
    if addition in current:
        return current
    return f"{current}\n\n{addition}" if current else addition
