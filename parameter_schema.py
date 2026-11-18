"""Shared proof-family and parameter-field definitions.

The hierarchical classifier knows all six proof families. Parameter NER is
intentionally scoped to five families while Function Properties is on hold.
"""
from __future__ import annotations

NER_SCHEMA_VERSION = 2

SUBJECT_FAMILIES: dict[str, tuple[str, ...]] = {
    "Real Analysis": (
        "Sequence Convergence",
        "Integration and Approximation",
    ),
    "Linear Algebra": (
        "Vector Operations",
        "Linear Transformations",
    ),
    "Differential Geometry": (
        "Curves and Moving Frames",
    ),
}

# Only these fields are learned by NER. Keep Function Properties as an empty
# entry so the classifier/app can still recognise the family without asking
# the current NER model to extract parameters for it.
PARAMETER_FIELDS: dict[str, tuple[str, ...]] = {
    "Sequence Convergence": (
        "sequence", "limit", "bound",
    ),
    "Integration and Approximation": (
        "function", "interval", "sample_point", "integral_value",
    ),
    "Vector Operations": (
        "vectors",
    ),
    "Linear Transformations": (
        "matrix",
    ),
    "Curves and Moving Frames": (
        "curve", "evaluation_point",
    ),
}

NER_FAMILIES: tuple[str, ...] = tuple(
    family for family, fields in PARAMETER_FIELDS.items() if fields
)

# Data fields may still appear in the JSON files for classifier training,
# animation templates, or later processing, but they are not NER labels.
DERIVED_OR_DEFAULT_FIELDS = frozenset({
    "index_start", "epsilon", "convergence_type", "parameter_c",
    "monotone", "upper_bound", "proof_technique",
    "partition_count", "partition_type", "dimension",
    "parameter_interval", "curvature", "curvature_formula", "curve_type",
    "torsion", "tangent", "normal", "osculating_radius", "binormal",
    "eigenvalues", "eigenvectors", "determinant", "P", "D",
    "area_scaling", "image_area", "unit_square_area", "orientation_reversal",
    "eigenspace_dimension", "rotation_angle", "technique", "process",
    "inner_product", "projection", "onto", "direction", "vector", "u", "v",
    "u1", "u2", "orthogonal", "parallel", "perpendicular", "norm",
    "norm_u", "norm_v", "norm_sum", "result", "animation_style",
    "theorem", "special_point", "derivative", "secant_slope",
    "endpoint_values", "target_value", "max_value", "max_point",
    "min_value", "min_point",
})


def build_text(question: str, proof: str) -> str:
    """Build exactly the input text shared by classifier and parameter NER."""
    parts: list[str] = []
    if question and question.strip():
        parts.append("Question: " + question.strip())
    if proof and proof.strip():
        parts.append("Proof: " + proof.strip())
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Manim template contracts
# ---------------------------------------------------------------------------
# These names deliberately remain separate from PARAMETER_FIELDS above:
# NER learns the fields requested for each proof family; the template contract
# describes the arguments passed to the corresponding Manim scene.
TEMPLATE_FAMILY_MAP: dict[str, str] = {
    "Sequence Convergence": "Sequence Convergence",
    "Integration and Approximation": "Integration",
    "Vector Operations": "Vector",
    "Linear Transformations": "Matrix",
    "Curves and Moving Frames": "Curvature",
}

TEMPLATE_FILES: dict[str, str] = {
    "Sequence Convergence": "sequence_convergence_template.py",
    "Integration": "integration_approximation_template.py",
    "Vector": "gram_schmidt_template.py",
    "Matrix": "eigen_diagonal_template.py",
    "Curvature": "curvature_template.py",
}

TEMPLATE_PARAMETER_FIELDS: dict[str, tuple[str, ...]] = {
    "Sequence Convergence": ("sequence", "limit", "bound"),
    "Integration": ("function", "interval", "sample_point", "integral_value"),
    "Vector": ("vectors",),
    "Matrix": ("matrix",),
    # The learned CURVE entity is parsed into coordinate expressions such as
    # ["t", "t^2"], then passed directly to CurvatureCurve2D/3D.
    "Curvature": ("curve", "evaluation_point"),
}

TEMPLATE_FAMILY_FIELDS: dict[str, tuple[str, ...]] = {
    family: TEMPLATE_PARAMETER_FIELDS[template_family]
    for family, template_family in TEMPLATE_FAMILY_MAP.items()
}

MANIM_FAMILIES: tuple[str, ...] = tuple(TEMPLATE_FAMILY_MAP)
