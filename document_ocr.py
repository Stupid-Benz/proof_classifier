"""Full-page image/PDF OCR using PaddleOCR's PP-StructureV3 document parser.

The PaddleOCR dependency is intentionally optional and imported only when the
user runs full-page OCR. Uploads are kept in a temporary file and removed after
inference; page text/Markdown is returned to the caller for review.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
from typing import Any, Iterable

from PIL import Image, UnidentifiedImageError

MAX_DOCUMENT_BYTES = 30 * 1024 * 1024
MAX_DOCUMENT_PIXELS = 60_000_000
MAX_PDF_PAGES = 25
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}
ALLOWED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


class DocumentOCRInputError(ValueError):
    """Raised when a document upload is invalid or outside configured limits."""


def _validate_image(data: bytes) -> None:
    try:
        with Image.open(BytesIO(data)) as image:
            image_format = (image.format or "").upper()
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise DocumentOCRInputError(f"Unsupported image format: {image_format or 'unknown'}.")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_DOCUMENT_PIXELS:
                raise DocumentOCRInputError(
                    f"Image dimensions {width}×{height} exceed the {MAX_DOCUMENT_PIXELS:,}-pixel limit."
                )
            image.verify()
    except DocumentOCRInputError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise DocumentOCRInputError(f"Could not read the uploaded image: {exc}") from exc


def validate_document_bytes(data: bytes, filename: str) -> str:
    """Validate an image/PDF upload and return its normalized suffix."""
    if not data:
        raise DocumentOCRInputError("The uploaded document is empty.")
    if len(data) > MAX_DOCUMENT_BYTES:
        raise DocumentOCRInputError(
            f"Document is too large ({len(data) / 1024 / 1024:.1f} MB). "
            f"The limit is {MAX_DOCUMENT_BYTES // (1024 * 1024)} MB."
        )
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise DocumentOCRInputError("Upload a PDF or a PNG/JPG/WEBP/BMP/TIFF image.")
    if suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise DocumentOCRInputError("The file extension is PDF, but its content is not a valid PDF header.")
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF page validation requires pypdf. Install the optional document OCR dependencies."
            ) from exc
        try:
            reader = PdfReader(BytesIO(data), strict=False)
            if reader.is_encrypted:
                raise DocumentOCRInputError("Encrypted PDFs are not supported. Please upload an unlocked PDF.")
            pages = len(reader.pages)
        except DocumentOCRInputError:
            raise
        except Exception as exc:
            raise DocumentOCRInputError(f"Could not inspect PDF pages: {exc}") from exc
        if pages < 1:
            raise DocumentOCRInputError("The PDF contains no pages.")
        if pages > MAX_PDF_PAGES:
            raise DocumentOCRInputError(
                f"The PDF has {pages} pages. The current limit is {MAX_PDF_PAGES} pages per OCR run."
            )
    else:
        _validate_image(data)
    return suffix


def create_pp_structure_pipeline(device: str = "cpu"):
    """Create PP-StructureV3 with formula recognition enabled."""
    try:
        from paddleocr import PPStructureV3
    except ImportError as exc:
        raise RuntimeError(
            "Full-page OCR is optional. Install PaddlePaddle for your platform, then run "
            "`pip install -r requirements-document-ocr.txt`."
        ) from exc
    return PPStructureV3(
        lang="en",
        use_formula_recognition=True,
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device,
    )


def _flatten_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(part for part in (_flatten_markdown_value(item) for item in value) if part)
    if isinstance(value, dict):
        for key in ("markdown_text", "text", "content", "markdown_texts", "block_content"):
            if key in value:
                text = _flatten_markdown_value(value[key])
                if text:
                    return text
    return ""


def _markdown_info(result: Any) -> Any:
    value = getattr(result, "markdown", None)
    if callable(value):
        value = value()
    return value


def _page_text(result: Any, markdown_info: Any) -> str:
    text = _flatten_markdown_value(markdown_info)
    if text.strip():
        return text.strip()
    payload = getattr(result, "json", None)
    if callable(payload):
        payload = payload()
    if isinstance(payload, dict):
        parsing = payload.get("parsing_res_list", [])
        if isinstance(parsing, list):
            ordered = sorted(
                (item for item in parsing if isinstance(item, dict)),
                key=lambda item: (item.get("block_order") is None, item.get("block_order", 0)),
            )
            blocks = [str(item.get("block_content", "")).strip() for item in ordered]
            return "\n\n".join(block for block in blocks if block)
    return ""


def run_document_ocr(
    document_bytes: bytes,
    filename: str,
    pipeline: Any,
) -> dict[str, Any]:
    """Run a supplied PP-StructureV3-compatible pipeline on a PDF/image upload."""
    suffix = validate_document_bytes(document_bytes, filename)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix="proof_doc_ocr_", suffix=suffix, delete=False) as handle:
            handle.write(document_bytes)
            temporary_path = Path(handle.name)

        # PP-StructureV3 documents ``predict(input=...)`` for both image and PDF paths.
        results = list(pipeline.predict(input=str(temporary_path)))
        if not results:
            raise RuntimeError("The document OCR pipeline returned no page results.")

        markdown_infos = [_markdown_info(result) for result in results]
        pages = []
        for index, (result, md_info) in enumerate(zip(results, markdown_infos), start=1):
            page_index = getattr(result, "page_index", None)
            if page_index is None:
                payload = getattr(result, "json", None)
                if callable(payload):
                    payload = payload()
                if isinstance(payload, dict):
                    page_index = payload.get("page_index")
            pages.append({
                "page_number": int(page_index) + 1 if isinstance(page_index, int) else index,
                "text": _page_text(result, md_info),
            })

        combined = ""
        concatenate = getattr(pipeline, "concatenate_markdown_pages", None)
        if callable(concatenate) and all(isinstance(info, dict) for info in markdown_infos):
            try:
                combined = str(concatenate(markdown_infos) or "").strip()
            except Exception:
                combined = ""
        if not combined:
            combined = "\n\n".join(
                f"<!-- Page {page['page_number']} -->\n{page['text']}"
                for page in pages if page["text"]
            ).strip()
        if not combined:
            raise RuntimeError(
                "The OCR pipeline ran but returned no readable Markdown/text. "
                "Check the original document or inspect the PaddleOCR output."
            )
        return {
            "filename": Path(filename).name,
            "page_count": len(results),
            "pages": pages,
            "text": combined,
            "engine": "PaddleOCR PP-StructureV3",
        }
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def append_document_text(existing_text: str, extracted_text: str) -> str:
    """Append reviewed OCR text without silently replacing or duplicating content."""
    existing = str(existing_text or "").rstrip()
    addition = str(extracted_text or "").strip()
    if not addition:
        raise DocumentOCRInputError("There is no reviewed OCR text to insert.")
    if addition in existing:
        return existing
    return f"{existing}\n\n{addition}" if existing else addition
