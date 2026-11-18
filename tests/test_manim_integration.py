from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from math_preproc import parse_matrix, parse_vectors
from parameter_schema import MANIM_FAMILIES, TEMPLATE_FAMILY_MAP
import template_runtime
from template_runtime import (
    choose_scene_class,
    default_parameters,
    merge_parameter_suggestions,
    render_manim,
    template_path_for_family,
    validate_parameters,
)


class ManimTemplateIntegrationTests(unittest.TestCase):
    def test_all_five_classifier_families_have_template_files(self):
        self.assertEqual(len(MANIM_FAMILIES), 5)
        for family in MANIM_FAMILIES:
            self.assertTrue(template_path_for_family(family).is_file(), family)

    def test_ner_values_are_adapted_to_template_parameter_contracts(self):
        cases = {
            "Sequence Convergence": {
                "sequence": ["3/n"], "limit": ["0"], "bound": ["3/n"]
            },
            "Integration and Approximation": {
                "function": ["x^2"], "interval": ["[0, 1]"],
                "sample_point": ["right"], "integral_value": ["1/3"]
            },
            "Vector Operations": {"vectors": ["(1, 2)", "(2, 1)"]},
            "Linear Transformations": {"matrix": ["[[4, 0], [0, 2]]"]},
            "Curves and Moving Frames": {
                "curve": ["(t, t^2)"], "evaluation_point": ["1"]
            },
        }
        for family, extracted in cases.items():
            params = default_parameters(family, extracted)
            self.assertFalse(validate_parameters(family, params), family)

        integration = default_parameters("Integration and Approximation", cases["Integration and Approximation"])
        self.assertEqual(integration["sample_point"], "right_endpoint")

    def test_ner_curve_span_is_parsed_for_the_manim_constructor(self):
        params = default_parameters(
            "Curves and Moving Frames",
            {"curve": ["(t, t^2)"], "evaluation_point": ["1"]},
        )
        self.assertEqual(params["curve"], ["t", "t^2"])
        self.assertEqual(params["evaluation_point"], "1")
        self.assertFalse(validate_parameters("Curves and Moving Frames", params))

    def test_f01_style_curve_entity_flows_to_curvature_template_inputs(self):
        question = "Compute the curvature of the parabola gamma(t) = (t, t^2) at t = 1."
        params = merge_parameter_suggestions(
            "Curves and Moving Frames",
            {"curve": ["(t, t^2)"], "evaluation_point": ["1"]},
            question,
            "",
        )
        self.assertEqual(params, {"curve": ["t", "t^2"], "evaluation_point": "1"})
        self.assertEqual(choose_scene_class("Curves and Moving Frames", params), "CurvatureCurve2D")
        self.assertFalse(validate_parameters("Curves and Moving Frames", params))

    def test_three_dimensional_curve_entity_is_parsed_as_three_coordinates(self):
        params = default_parameters(
            "Curves and Moving Frames",
            {"curve": ["(cos t, sin t, t)"], "evaluation_point": ["0"]},
        )
        self.assertEqual(params["curve"], ["cos(t)", "sin(t)", "t"])
        self.assertEqual(choose_scene_class("Curves and Moving Frames", params), "CurvatureCurve3D")
        self.assertFalse(validate_parameters("Curves and Moving Frames", params))

    def test_curve_fallback_parses_nested_functions_without_splitting_them(self):
        params = merge_parameter_suggestions(
            "Curves and Moving Frames",
            {"evaluation_point": ["1"]},
            "Compute the curvature of gamma(t) = (t, ln(cos(t))) at t = 1.",
            "",
        )
        self.assertEqual(params["curve"], ["t", "ln(cos(t))"])
        self.assertFalse(validate_parameters("Curves and Moving Frames", params))

    def test_vector_and_matrix_ner_text_spans_are_converted_to_python_lists(self):
        vector_parameters = default_parameters(
            "Vector Operations",
            {"vectors": ["(2,3,6)", "(3,-6,2)"]},
        )
        self.assertEqual(vector_parameters["vectors"], [[2, 3, 6], [3, -6, 2]])
        self.assertTrue(all(isinstance(x, int) for vector in vector_parameters["vectors"] for x in vector))

        matrix_parameters = default_parameters(
            "Linear Transformations",
            {"matrix": ["[[0,2],[2,0]]"]},
        )
        self.assertEqual(matrix_parameters["matrix"], [[0, 2], [2, 0]])
        self.assertTrue(all(isinstance(x, int) for row in matrix_parameters["matrix"] for x in row))
        self.assertEqual(choose_scene_class("Linear Transformations", matrix_parameters), "EigenDiagonal2D")

        # Existing mathematical parsers must still accept the normalized data.
        self.assertEqual(len(parse_vectors(vector_parameters["vectors"])), 2)
        self.assertEqual(parse_matrix(matrix_parameters["matrix"]).shape, (2, 2))

        symbolic_matrix = default_parameters(
            "Linear Transformations",
            {"matrix": ["[[a,0],[0,2]]"]},
        )
        self.assertEqual(symbolic_matrix["matrix"], [["a", 0], [0, 2]])
        self.assertEqual(parse_matrix(symbolic_matrix["matrix"]).shape, (2, 2))

    def test_partial_vector_ner_result_can_be_completed_by_text_suggestion(self):
        params = merge_parameter_suggestions(
            "Vector Operations",
            {"vectors": ["(1, 2)"]},
            "Apply Gram-Schmidt to v1 = (1, 2) and v2 = (2, 1).",
            "",
        )
        self.assertEqual(params["vectors"], [[1, 2], [2, 1]])
        self.assertFalse(validate_parameters("Vector Operations", params))

    def test_scene_class_is_selected_by_curve_dimension(self):
        self.assertEqual(
            choose_scene_class("Curves and Moving Frames", {"curve": ["t", "t^2"]}),
            "CurvatureCurve2D",
        )
        self.assertEqual(
            choose_scene_class("Curves and Moving Frames", {"curve": ["cos(t)", "sin(t)", "t"]}),
            "CurvatureCurve3D",
        )

    def test_required_template_input_error_is_clear(self):
        errors = validate_parameters("Curves and Moving Frames", {"curve": None, "evaluation_point": 1})
        self.assertTrue(any("Missing required parameter: curve" in error for error in errors))

    def test_render_runner_streams_subprocess_output_to_callback(self):
        class FakeProcess:
            def __init__(self):
                self.stdout = io.StringIO("Manim Community v0.22.1\nINFO Writing a TeX file\nRenderedScene done\n")

            @staticmethod
            def poll():
                return 0

            @staticmethod
            def wait(timeout=None):
                return 0

        live_updates = []
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(template_runtime, "GENERATED_DIR", Path(folder)), mock.patch.object(
                template_runtime.subprocess, "Popen", return_value=FakeProcess()
            ):
                result = render_manim(
                    "Sequence Convergence",
                    {"sequence": "3/n", "limit": "0", "bound": "3/n"},
                    log_callback=live_updates.append,
                )
        self.assertIn("Manim Community v0.22.1", result["log_text"])
        self.assertTrue(any("Writing a TeX file" in update for update in live_updates))


if __name__ == "__main__":
    unittest.main()
