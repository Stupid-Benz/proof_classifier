from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from manual_ner import (
    count_manual_annotations,
    encode_manual_spans,
    find_exact_occurrences,
    save_manual_annotation,
    validate_manual_entities,
)
from parameter_schema import build_text


class CharTokenizer:
    """Tiny deterministic tokenizer stand-in for offset/BIO unit tests."""
    def __call__(self, text, **kwargs):
        max_length = kwargs.get("max_length", len(text) + 2)
        offsets = [(0, 0)] + [(i, i + 1) for i in range(min(len(text), max_length - 2))] + [(0, 0)]
        return {"input_ids": list(range(len(offsets))), "offset_mapping": offsets}


class ManualNerTests(unittest.TestCase):
    def test_find_exact_occurrences_returns_all_repeated_matches(self):
        self.assertEqual(find_exact_occurrences("a = 1; sum -> 1", "1"), [(4, 5), (14, 15)])

    def test_validate_manual_entities_checks_exact_offsets_and_field(self):
        text = build_text("Prove a_n = 1/n.", "The bound is 1/n.")
        start = text.rfind("1/n")
        spans = validate_manual_entities(text, "Sequence Convergence", [
            {"start": start, "end": start + 3, "field": "bound", "text": "1/n"}
        ])
        self.assertEqual(spans[0]["text"], "1/n")

    def test_validate_rejects_wrong_text_and_overlap(self):
        text = "Question: x^2"
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_manual_entities(text, "Integration and Approximation", [
                {"start": 10, "end": 13, "field": "function", "text": "x^3"}
            ])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_manual_entities(text, "Integration and Approximation", [
                {"start": 10, "end": 13, "field": "function", "text": "x^2"},
                {"start": 11, "end": 13, "field": "integral_value", "text": "^2"},
            ])

    def test_partial_manual_spans_ignore_unannotated_tokens(self):
        text = "Question: x^2 and 1/n"
        spans = [{"start": 10, "end": 13, "field": "function", "text": "x^2"}]
        encoded = encode_manual_spans(
            text, spans, CharTokenizer(), {"O": 0, "B-function": 1, "I-function": 2}
        )
        self.assertEqual(encoded["labels"][11:14], [1, 2, 2])
        self.assertEqual(encoded["labels"][14], -100)

    def test_complete_manual_spans_label_other_tokens_as_o(self):
        text = "Question: x^2 and 1/n"
        spans = [{"start": 10, "end": 13, "field": "function", "text": "x^2"}]
        encoded = encode_manual_spans(
            text, spans, CharTokenizer(), {"O": 0, "B-function": 1, "I-function": 2}, complete=True
        )
        self.assertEqual(encoded["labels"][11:14], [1, 2, 2])
        self.assertEqual(encoded["labels"][14], 0)
        self.assertEqual(encoded["labels"][0], -100)

    def test_save_manual_annotation_deduplicates(self):
        question = "Prove that a_n = 2/n converges to 0."
        proof = "The bound is 2/n."
        text = build_text(question, proof)
        start = text.rfind("2/n")
        entities = [{"start": start, "end": start + 3, "field": "bound", "text": "2/n"}]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manual.json"
            kwargs = dict(question=question, proof=proof, family="Sequence Convergence", entities=entities, path=path)
            first = save_manual_annotation(**kwargs)
            second = save_manual_annotation(**kwargs)
            self.assertTrue(first["saved"])
            self.assertTrue(second["duplicate"])
            self.assertEqual(count_manual_annotations(path), 1)

    def test_manual_spans_are_converted_to_bio_labels(self):
        text = "Question: x^2"
        spans = [{"start": 10, "end": 13, "field": "function", "text": "x^2"}]
        encoded = encode_manual_spans(text, spans, CharTokenizer(), {"O": 0, "B-function": 1, "I-function": 2})
        self.assertEqual(encoded["labels"][11:14], [1, 2, 2])

    def test_manual_spans_outside_tokenizer_length_are_rejected(self):
        text = "Question: x^2 and more text"
        spans = [{"start": 26, "end": 27, "field": "function", "text": "t"}]
        with self.assertRaisesRegex(ValueError, "outside the tokenizer's max length"):
            encode_manual_spans(text, spans, CharTokenizer(), {"O": 0, "B-function": 1}, max_length=10)


if __name__ == "__main__":
    unittest.main()
