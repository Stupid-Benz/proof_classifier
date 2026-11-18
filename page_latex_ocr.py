"""Page-level OCR using Pix2Text: mixed text + formulas → Markdown with LaTeX."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from image_latex_ocr import LatexOCRInputError, open_image_bytes

import re
from typing import Any

_MATH_DELIM_RE = re.compile(r"(\$\$?)(.+?)(\$\$?)", re.DOTALL)


def _tighten_math(body: str) -> str:
    # digits glued to digits: "1 3" -> "13"
    body = re.sub(r"(?<=\d)\s+(?=\d)", "", body)
    # digits glued to a following letter: "4 n" -> "4n"
    body = re.sub(r"(?<=\d)\s+(?=[A-Za-z])", "", body)
    # runs of single-letter words: "l i m" -> "lim", "L e t" -> "Let"
    body = re.sub(
        r"(?<![A-Za-z])(?:[A-Za-z]\s+)+[A-Za-z](?![A-Za-z])",
        lambda m: re.sub(r"\s+", "", m.group(0)),
        body,
    )
    # tidy spacing around braces and \operatorname*
    body = re.sub(r"\\operatorname\*\s*\{", r"\\operatorname*{", body)
    body = re.sub(r"\s+\}", "}", body)
    body = re.sub(r"\{\s+", "{", body)
    body = re.sub(r"\}\s+\{", "}{", body)
    return body


def normalize_latex_output(markdown: str) -> str:
    """Collapse Pix2Text's character-level spacing inside math spans."""
    return _MATH_DELIM_RE.sub(
        lambda m: m.group(1) + _tighten_math(m.group(2)) + m.group(3),
        markdown,
    )

class PageOCRInputError(ValueError):
    """Raised when a page/PDF cannot be passed to Pix2Text."""


def _pix2text_available() -> bool:
    try:
        import pix2text  # noqa: F401
    except ImportError:
        return False
    return True


def _upscale_if_small(image_bytes: bytes, min_width: int = 1200) -> "Image.Image":
    from PIL import Image
    from image_latex_ocr import open_image_bytes

    img = open_image_bytes(image_bytes)
    if img.width < min_width:
        scale = min_width / img.width
        img = img.resize(
            (int(img.width * scale), int(img.height * scale)),
            Image.LANCZOS,
        )
    return img


def recognize_page_image(image_bytes: bytes, model: Any) -> str:
    """Run Pix2Text on one page image and return Markdown text with LaTeX."""
    image = _upscale_if_small(image_bytes)
    try:
        page = model.recognize_page(image,resized_shape=1280)
        return _page_to_markdown_text(page)
    except PageOCRInputError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Pix2Text page recognition failed: {exc}") from exc


def recognize_pdf(pdf_bytes: bytes, model: Any, page_numbers: list[int] | None = None) -> str:
    """Run Pix2Text on a PDF file and return Markdown text."""
    if not pdf_bytes:
        raise PageOCRInputError("The uploaded PDF is empty.")
    try:
        doc = model.recognize_pdf(BytesIO(pdf_bytes), page_numbers=page_numbers)
        return _document_to_markdown_text(doc)
    except Exception as exc:
        raise RuntimeError(f"Pix2Text PDF recognition failed: {exc}") from exc


def _page_to_markdown_text(page: Any) -> str:
    """Extract Markdown-like text from a Pix2Text Page object."""
    if page is None:
        raise PageOCRInputError("Pix2Text returned no page result.")
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            page.to_markdown(tmpdir)
            md_files = list(Path(tmpdir).glob("*.md"))
            if md_files:
                return normalize_latex_output(
                    md_files[0].read_text(encoding="utf-8")
                )
    except Exception:
        pass
    for attr in ("text", "markdown", "content"):
        value = getattr(page, attr, None)
        if isinstance(value, str) and value.strip():
            return normalize_latex_output(value.strip())
    raise PageOCRInputError("Could not extract text from Pix2Text page result.")


def _document_to_markdown_text(doc: Any) -> str:
    """Extract Markdown from a Pix2Text Document object."""
    if doc is None:
        raise PageOCRInputError("Pix2Text returned no document result.")
    try:
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            doc.to_markdown(tmpdir)
            md_files = list(Path(tmpdir).glob("*.md"))
            if md_files:
                return normalize_latex_output(
                    md_files[0].read_text(encoding="utf-8")
                )
    except Exception:
        pass
    for attr in ("text", "markdown", "content"):
        value = getattr(doc, attr, None)
        if isinstance(value, str) and value.strip():
            return normalize_latex_output(value.strip())
    raise PageOCRInputError("Could not extract text from Pix2Text document result.")


def load_pix2text_model(device: str = "cpu", enable_table: bool = False):
    from pix2text import Pix2Text

    return Pix2Text.from_config(
        enable_formula=True,
        enable_table=enable_table,
        text_config={
            "languages": ["en"],
            # Larger, English-oriented recogniser.
            # Alternatives to try: "en_PP-OCRv3", "doc-densenet_lite_666-gru_large".
            "rec_model_name": "doc-densenet_lite_666-gru_large",
        },
        device=device,
    )