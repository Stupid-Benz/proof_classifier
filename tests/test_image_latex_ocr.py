from __future__ import annotations

from io import BytesIO
import unittest

from PIL import Image

from image_latex_ocr import (
    LatexOCRInputError,
    append_latex_to_text,
    clean_latex_prediction,
    combine_latex_predictions,
    format_latex_block,
    open_image_bytes,
    recognize_latex_from_bytes,
)


def make_png_bytes(mode: str = "RGB") -> bytes:
    image = Image.new(mode, (80, 32), color="white")
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


class FakeLatexOCR:
    def __init__(self, result: str):
        self.result = result
        self.called_with_rgb = False

    def __call__(self, image):
        self.called_with_rgb = image.mode == "RGB"
        return self.result


class ImageLatexOCRTests(unittest.TestCase):
    def test_open_image_bytes_returns_rgb_image(self):
        image = open_image_bytes(make_png_bytes("RGBA"))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (80, 32))

    def test_rejects_empty_or_non_image_data(self):
        with self.assertRaises(LatexOCRInputError):
            open_image_bytes(b"")
        with self.assertRaises(LatexOCRInputError):
            open_image_bytes(b"not an image")

    def test_ocr_prediction_uses_rgb_image_and_returns_latex(self):
        model = FakeLatexOCR(r"\frac{1}{2}\ln 2")
        result = recognize_latex_from_bytes(make_png_bytes("RGBA"), model)
        self.assertEqual(result, r"\frac{1}{2}\ln 2")
        self.assertTrue(model.called_with_rgb)

    def test_empty_model_result_is_rejected(self):
        with self.assertRaises(LatexOCRInputError):
            clean_latex_prediction("   ")

    def test_format_latex_wraps_bare_expression(self):
        self.assertEqual(
            format_latex_block(r"\frac{1}{2}"),
            "\\[\n\\frac{1}{2}\n\\]",
        )

    def test_format_latex_preserves_existing_display_math(self):
        expression = r"\[x^2 + y^2\]"
        self.assertEqual(format_latex_block(expression), expression)

    def test_combine_predictions_creates_separate_math_blocks(self):
        result = combine_latex_predictions([
            ("one.png", "x^2"),
            ("two.jpg", r"\frac{1}{2}"),
        ])
        self.assertIn(r"\[", result)
        self.assertIn("x^2", result)
        self.assertIn(r"\frac{1}{2}", result)
        self.assertEqual(result.count(r"\["), 2)

    def test_append_latex_preserves_existing_text(self):
        result = append_latex_to_text("Prove this theorem.", r"\[x^2\]")
        self.assertEqual(result, "Prove this theorem.\n\n\\[x^2\\]")

    def test_append_does_not_duplicate_same_latex(self):
        original = "Proof text.\n\n\\[x^2\\]"
        self.assertEqual(append_latex_to_text(original, r"\[x^2\]"), original)


if __name__ == "__main__":
    unittest.main()
