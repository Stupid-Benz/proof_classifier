from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

from PIL import Image

from document_ocr import (
    DocumentOCRInputError,
    append_document_text,
    run_document_ocr,
    validate_document_bytes,
)


def png_bytes() -> bytes:
    image = Image.new("RGB", (90, 50), "white")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class FakePage:
    def __init__(self, text, page_index):
        self.markdown = {"markdown_texts": text, "markdown_images": {}}
        self.page_index = page_index


class FakePipeline:
    def __init__(self):
        self.predict_path = None
        self.concatenate_called = False

    def predict(self, input):
        self.predict_path = Path(input)
        assert self.predict_path.exists()
        return [FakePage("## Page one\nText and $x^2$", 0), FakePage("## Page two\nMore proof", 1)]

    def concatenate_markdown_pages(self, markdown_list):
        self.concatenate_called = True
        return "\n\n".join(item["markdown_texts"] for item in markdown_list)


class DocumentOCRTests(unittest.TestCase):
    def test_validate_image_upload(self):
        self.assertEqual(validate_document_bytes(png_bytes(), "proof.png"), ".png")

    def test_rejects_unknown_extension_and_bad_image(self):
        with self.assertRaises(DocumentOCRInputError):
            validate_document_bytes(b"abc", "proof.txt")
        with self.assertRaises(DocumentOCRInputError):
            validate_document_bytes(b"not an image", "proof.png")

    def test_pdf_page_limit_is_checked(self):
        fake_reader = type("FakeReader", (), {"__init__": lambda self, *_a, **_k: None,
                                             "is_encrypted": False,
                                             "pages": list(range(26))})
        fake_module = types.SimpleNamespace(PdfReader=fake_reader)
        with patch.dict(sys.modules, {"pypdf": fake_module}):
            with self.assertRaisesRegex(DocumentOCRInputError, "limit is 25"):
                validate_document_bytes(b"%PDF-1.7\n", "many.pdf")

    def test_run_document_ocr_reconstructs_pages_and_cleans_temp_file(self):
        pipeline = FakePipeline()
        result = run_document_ocr(png_bytes(), "proof.png", pipeline)
        self.assertEqual(result["page_count"], 2)
        self.assertIn("$x^2$", result["text"])
        self.assertIn("More proof", result["text"])
        self.assertTrue(pipeline.concatenate_called)
        self.assertFalse(pipeline.predict_path.exists())

    def test_append_document_text_preserves_and_deduplicates(self):
        self.assertEqual(append_document_text("Question", "Proof"), "Question\n\nProof")
        self.assertEqual(append_document_text("Question\n\nProof", "Proof"), "Question\n\nProof")
        with self.assertRaises(DocumentOCRInputError):
            append_document_text("Existing", "   ")


if __name__ == "__main__":
    unittest.main()
