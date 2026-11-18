from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_loader import DEFAULT_DATA_PATH, discover_json_files, load_records
from json_utils import json_safe_default
from ner_alignment import align_parameters_to_spans, parameters_for_ner
from parameter_extractor import ParameterExtractor, highest_confidence_entity
from parameter_schema import NER_FAMILIES, PARAMETER_FIELDS, SUBJECT_FAMILIES, build_text
from feedback_store import count_feedback_records, save_reviewed_feedback
from manual_ner import count_manual_annotations, save_manual_annotation
from train_ner import split_examples_with_feedback

ROOT = Path(__file__).resolve().parents[1]


def sample_record(record_id: str) -> dict:
    return {
        "id": record_id,
        "subject": "Real Analysis",
        "proof_family": "Sequence Convergence",
        "theorem": "Example limit",
        "question": r"Prove that a_n = 2/n converges to 0.",
        "proof": r"Let eps > 0 and choose N > 2/eps.",
        "parameters": {"sequence": "2/n", "limit": 0},
        "animation_notes": "",
    }


class ProjectStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_records(DEFAULT_DATA_PATH)

    def record(self, record_id):
        return next(record for record in self.records if record.record_id == record_id)

    def test_dataset_has_102_examples_and_balanced_families(self):
        self.assertEqual(len(self.records), 102)
        counts = {}
        for record in self.records:
            counts[record.proof_family] = counts.get(record.proof_family, 0) + 1
            self.assertIsInstance(record.parameters, dict)
            self.assertTrue(record.parameters)
            if record.proof_family == "Sequence Convergence":
                self.assertIn("bound", record.parameters)
        self.assertEqual(set(counts.values()), {17})
        self.assertEqual(len(PARAMETER_FIELDS), 6)  # includes the classifier-only empty family
        self.assertEqual(len(NER_FAMILIES), 5)

    def test_default_data_is_multiple_json_files(self):
        files = discover_json_files(DEFAULT_DATA_PATH)
        self.assertEqual(len(files), 6)
        self.assertTrue(all(path.suffix == ".json" for path in files))
        self.assertFalse((ROOT / "data" / "proof_data.py").exists())
        self.assertEqual(len({record.source_file for record in self.records}), 6)

    def test_loader_combines_multiple_json_files_from_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            (folder_path / "part_a.json").write_text(
                json.dumps([sample_record("CUSTOM-A")]), encoding="utf-8"
            )
            (folder_path / "part_b.json").write_text(
                json.dumps({"proofs": [sample_record("CUSTOM-B")]}), encoding="utf-8"
            )
            records = load_records(folder_path)
        self.assertEqual({record.record_id for record in records}, {"CUSTOM-A", "CUSTOM-B"})

    def test_loader_accepts_a_single_json_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "one.json"
            path.write_text(json.dumps([sample_record("SINGLE")]), encoding="utf-8")
            records = load_records(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].record_id, "SINGLE")

    def test_loader_rejects_duplicate_ids_across_json_files(self):
        record = sample_record("DUP")
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            (folder_path / "a.json").write_text(json.dumps([record]), encoding="utf-8")
            (folder_path / "b.json").write_text(json.dumps([record]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                load_records(folder_path)

    def test_loader_rejects_non_json_file_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.py"
            path.write_text("PROOF_DATA = []", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"\.json"):
                load_records(path)

    def test_subject_family_schema_is_consistent(self):
        self.assertEqual(sum(len(x) for x in SUBJECT_FAMILIES.values()), 6)
        for record in self.records:
            self.assertIn(record.proof_family, SUBJECT_FAMILIES[record.subject])


class ConfidenceDisplayTests(unittest.TestCase):
    def test_highest_confidence_entity_returns_only_the_maximum(self):
        entities = [
            {"field": "sequence", "text": "3/n", "confidence": 0.82},
            {"field": "limit", "text": "0", "confidence": 0.96},
            {"field": "bound", "text": "3/n", "confidence": 0.74},
        ]
        best = highest_confidence_entity(entities)
        self.assertEqual(best["field"], "limit")
        self.assertEqual(best["confidence"], 0.96)

    def test_highest_confidence_ignores_missing_or_non_finite_scores(self):
        entities = [
            {"field": "sequence", "text": "3/n"},
            {"field": "limit", "text": "0", "confidence": float("nan")},
            {"field": "bound", "text": "3/n", "confidence": 0.81},
        ]
        best = highest_confidence_entity(entities)
        self.assertEqual(best["field"], "bound")

    def test_highest_confidence_returns_none_when_no_valid_score_exists(self):
        self.assertIsNone(highest_confidence_entity([]))
        self.assertIsNone(highest_confidence_entity([{"field": "limit", "confidence": None}]))


class NerDataPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = load_records(DEFAULT_DATA_PATH)

    def record(self, record_id):
        return next(record for record in self.records if record.record_id == record_id)

    def test_gram_schmidt_aligns_two_input_vectors_under_one_field(self):
        record = self.record("D1")
        text = build_text(record.question, record.proof)
        targets = parameters_for_ner(record.parameters, record.proof_family)
        spans = align_parameters_to_spans(text, targets, record.proof_family)
        fields = [(span["field"], span["text"]) for span in spans]
        self.assertEqual(fields, [("vectors", "(1,1)"), ("vectors", "(1,0)")])

    def test_legacy_projection_inputs_are_combined_as_vectors(self):
        record = self.record("D7")
        targets = parameters_for_ner(record.parameters, record.proof_family)
        self.assertEqual(targets, {"vectors": [[3, 4], [1, 0]]})

    def test_only_the_requested_fields_are_ner_targets(self):
        self.assertEqual(PARAMETER_FIELDS["Sequence Convergence"], ("sequence", "limit", "bound"))
        self.assertEqual(
            PARAMETER_FIELDS["Integration and Approximation"],
            ("function", "interval", "sample_point", "integral_value"),
        )
        self.assertEqual(PARAMETER_FIELDS["Vector Operations"], ("vectors",))
        self.assertEqual(PARAMETER_FIELDS["Linear Transformations"], ("matrix",))
        self.assertEqual(
            PARAMETER_FIELDS["Curves and Moving Frames"],
            ("curve", "evaluation_point"),
        )
        self.assertEqual(PARAMETER_FIELDS["Function Properties"], ())

    def test_sequence_bound_is_distinct_from_sequence_and_N_bound(self):
        question = "Prove that a_n = 3/n converges to 0."
        proof = (
            "Take any positive eps. Choose N as a natural number greater than 3/eps. "
            "Then for any n from N onwards, 3/n is at most 3/N, which is less than eps. "
            "Hence the sequence 3/n approaches 0."
        )
        text = build_text(question, proof)
        targets = {"sequence": "3/n", "limit": 0, "bound": "3/n"}
        spans = align_parameters_to_spans(text, targets, "Sequence Convergence")
        fields = [(span["field"], span["text"]) for span in spans]
        self.assertIn(("sequence", "3/n"), fields)
        self.assertIn(("bound", "3/n"), fields)
        self.assertNotIn("N", [span["text"] for span in spans])
        self.assertEqual(len([field for field, _ in fields if field == "bound"]), 1)

    def test_integration_extracts_function_interval_sample_point_and_value(self):
        question = "Use right endpoints to prove that the integral of x^2 from 0 to 1 equals 1/3."
        proof = "For x^2 on [0, 1], the right Riemann sum tends to 1/3."
        text = build_text(question, proof)
        targets = {
            "function": "x^2",
            "interval": "[0, 1]",
            "sample_point": "right_endpoint",
            "integral_value": "1/3",
        }
        spans = align_parameters_to_spans(text, targets, "Integration and Approximation")
        fields = {span["field"] for span in spans}
        self.assertEqual(fields, {"function", "interval", "sample_point", "integral_value"})
        sample_span = next(span for span in spans if span["field"] == "sample_point")
        self.assertEqual(sample_span["text"], "right")
        self.assertNotIn("partition_count", fields)

    def test_curve_and_evaluation_point_are_the_only_curve_targets(self):
        question = "Compute the curvature of gamma(t) = (t, t^2) at t equal to 1."
        proof = "At t equal to 1, the curvature is 2/(5 sqrt(5))."
        targets = {"curve": ["t", "t^2"], "evaluation_point": 1, "curvature": "2/(5*sqrt(5))"}
        spans = align_parameters_to_spans(
            build_text(question, proof), targets, "Curves and Moving Frames"
        )
        self.assertEqual({span["field"] for span in spans}, {"curve", "evaluation_point"})
        curve_span = next(span for span in spans if span["field"] == "curve")
        self.assertEqual(curve_span["text"], "(t, t^2)")

    def test_every_differential_geometry_record_has_a_curve_training_span(self):
        records = [r for r in self.records if r.proof_family == "Curves and Moving Frames"]
        self.assertEqual(len(records), 17)
        for record in records:
            spans = align_parameters_to_spans(
                build_text(record.question, record.proof),
                parameters_for_ner(record.parameters, record.proof_family),
                record.proof_family,
            )
            self.assertTrue(
                any(span["field"] == "curve" for span in spans),
                f"No CURVE span aligned for {record.record_id}",
            )

    def test_curve_weak_alignment_handles_trig_and_exponential_coordinate_forms(self):
        cases = [
            (["cos(t)", "sin(t)"], "gamma(t) = (cos t, sin t)", "(cos t, sin t)"),
            (["exp(t)*cos(t)", "exp(t)*sin(t)"], "gamma(t) = (e^t cos t, e^t sin t)", "(e^t cos t, e^t sin t)"),
            (["t", "ln(cos(t))"], "gamma(t) = (t, ln(cos t))", "(t, ln(cos t))"),
        ]
        for curve, statement, expected in cases:
            spans = align_parameters_to_spans(
                build_text(statement, ""), {"curve": curve}, "Curves and Moving Frames"
            )
            self.assertEqual(len(spans), 1, statement)
            self.assertEqual(spans[0]["field"], "curve")
            self.assertEqual(spans[0]["text"], expected)

    def test_function_properties_is_not_trained_by_ner(self):
        extractor = ParameterExtractor("this-model-directory-does-not-exist")
        result = extractor.extract("question", "proof", "Function Properties")
        self.assertFalse(result["family_supported"])
        self.assertEqual(result["supported_fields"], [])

    def test_curve_is_a_ner_target_and_curvature_is_not(self):
        targets = parameters_for_ner(
            {"curve": ["t", "t^2"], "evaluation_point": 1, "curvature": "2/(5*sqrt(5))"},
            "Curves and Moving Frames",
        )
        self.assertEqual(targets, {"curve": ["t", "t^2"], "evaluation_point": 1})

    def test_scalar_identity_matrix_shorthand_is_recognized(self):
        record = self.record("E3")
        spans = align_parameters_to_spans(
            build_text(record.question, record.proof),
            record.parameters,
            record.proof_family,
        )
        self.assertTrue(any(span["field"] == "matrix" and "identity matrix" in span["text"].lower() for span in spans))

    def test_latex_fraction_span_alignment(self):
        text = r"Question: Show a_n = \frac{2}{n} converges to 0. Proof: Let eps > 0."
        spans = align_parameters_to_spans(
            text, {"sequence": r"\frac{2}{n}", "limit": 0}, "Sequence Convergence"
        )
        self.assertTrue(any(s["field"] == "sequence" for s in spans))




class UserFeedbackStoreTests(unittest.TestCase):
    def test_reviewed_feedback_is_saved_as_ner_training_record_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ner_feedback.json"
            kwargs = {
                "question": "Prove that a_n = 3/n converges to 0.",
                "proof": "The error is |a_n - 0| = 3/n.",
                "family": "Sequence Convergence",
                "template_parameters": {
                    "sequence": "3/n", "limit": "0", "bound": "3/n"
                },
                "path": path,
            }
            first = save_reviewed_feedback(**kwargs)
            second = save_reviewed_feedback(**kwargs)

            self.assertTrue(first["saved"])
            self.assertFalse(first["duplicate"])
            self.assertFalse(second["saved"])
            self.assertTrue(second["duplicate"])
            self.assertEqual(count_feedback_records(path), 1)

            records = load_records(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].proof_family, "Sequence Convergence")
            self.assertEqual(
                records[0].parameters,
                {"sequence": "3/n", "limit": "0", "bound": "3/n"},
            )
            spans = align_parameters_to_spans(
                build_text(records[0].question, records[0].proof),
                records[0].parameters,
                records[0].proof_family,
            )
            self.assertEqual({span["field"] for span in spans}, {"sequence", "limit", "bound"})

    def test_feedback_examples_are_added_to_training_but_never_test(self):
        examples = [
            {"family": "Sequence Convergence", "is_feedback": False},
            {"family": "Sequence Convergence", "is_feedback": False},
            {"family": "Integration and Approximation", "is_feedback": False},
            {"family": "Integration and Approximation", "is_feedback": False},
            {"family": "Sequence Convergence", "is_feedback": True},
            {"family": "Integration and Approximation", "is_feedback": True, "is_manual_annotation": True},
        ]
        train_indices, test_indices, base_train_indices, feedback_indices = split_examples_with_feedback(
            examples, test_size=0.5, seed=42
        )
        self.assertIn(4, train_indices)
        self.assertIn(5, train_indices)
        self.assertNotIn(4, test_indices)
        self.assertNotIn(5, test_indices)
        self.assertEqual(set(feedback_indices), {4, 5})
        self.assertEqual(set(train_indices), set(base_train_indices) | {4, 5})
        self.assertTrue(all(not examples[int(i)]["is_feedback"] for i in test_indices))

    def test_feedback_rejects_empty_proof_and_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "question or proof"):
                save_reviewed_feedback(
                    question="", proof="", family="Vector Operations",
                    template_parameters={"vectors": [[1, 2], [2, 1]]},
                    path=Path(folder) / "feedback.json",
                )
            with self.assertRaisesRegex(ValueError, "No NER-supported fields"):
                save_reviewed_feedback(
                    question="Apply Gram-Schmidt to these vectors.", proof="",
                    family="Vector Operations", template_parameters={},
                    path=Path(folder) / "feedback.json",
                )


class TrainingEntrypointTests(unittest.TestCase):
    def test_training_modules_are_separated(self):
        self.assertTrue((ROOT / "train_classifier.py").is_file())
        self.assertTrue((ROOT / "train_ner.py").is_file())
        self.assertTrue((ROOT / "train.py").is_file())
        orchestrator = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertIn("train_classifier.py", orchestrator)
        self.assertIn("train_ner.py", orchestrator)

    def test_ner_inference_has_no_rule_baseline(self):
        self.assertFalse(hasattr(ParameterExtractor, "extract_rule_parameters"))
        extractor = ParameterExtractor("this-model-directory-does-not-exist")
        result = extractor.extract("question", "proof", "Sequence Convergence")
        self.assertFalse(result["ner_enabled"])
        self.assertEqual(result["parameters"], {})

    def test_old_curvature_ner_checkpoint_is_rejected(self):
        import tempfile
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "config.json").write_text("{}", encoding="utf-8")
            old_fields = {family: list(fields) for family, fields in PARAMETER_FIELDS.items()}
            old_fields["Curves and Moving Frames"] = ["curvature", "evaluation_point"]
            (path / "ner_metadata.json").write_text(
                json.dumps({"parameter_fields": old_fields, "id2label": {"0": "O"}}),
                encoding="utf-8",
            )
            extractor = ParameterExtractor(path)
            self.assertFalse(extractor.has_ner_model)
            self.assertIn("older parameter schema", extractor.checkpoint_problem)

    def test_json_serializer_handles_numpy_metrics(self):
        report = {"support": np.int64(7), "f1": np.float64(0.75)}
        decoded = json.loads(json.dumps(report, default=json_safe_default))
        self.assertEqual(decoded, {"support": 7, "f1": 0.75})


    def test_manual_annotation_records_load_with_exact_entity_offsets(self):
        question = "Prove that a_n = 2/n converges to 0."
        proof = "The error bound is 2/n."
        text = build_text(question, proof)
        start = text.rfind("2/n")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manual_ner_annotations.json"
            save_manual_annotation(
                question=question,
                proof=proof,
                family="Sequence Convergence",
                entities=[{"start": start, "end": start + 3, "field": "bound", "text": "2/n"}],
                complete=True,
                path=path,
            )
            records = load_records(path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].parameters, {})
        self.assertEqual(records[0].manual_entities[0]["field"], "bound")
        self.assertEqual(records[0].manual_entities[0]["text"], "2/n")
        self.assertTrue(records[0].manual_annotation_complete)



if __name__ == "__main__":
    unittest.main(verbosity=2)
