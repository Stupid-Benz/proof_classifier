"""Parameter preparation, validation, symbolic reports and live Manim rendering."""
from __future__ import annotations

import ast
import json
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from parameter_schema import (
    TEMPLATE_FAMILY_MAP,
    TEMPLATE_FILES,
    TEMPLATE_PARAMETER_FIELDS,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = PROJECT_ROOT / "templates"
GENERATED_DIR = PROJECT_ROOT / "generated"

# Public proof-family labels are the classifier's current labels.  The short
# names are internal template-family names retained for compatibility.
FAMILY_ALIASES = {**TEMPLATE_FAMILY_MAP}
FAMILY_ALIASES.update({value: value for value in TEMPLATE_FAMILY_MAP.values()})

DEFAULT_SCENE_CLASSES = {
    "Sequence Convergence": "SequenceConvergenceTemplate",
    "Integration": "IntegrationApproximationTemplate",
    "Vector": None,
    "Curvature": None,
    "Matrix": None,
}


def _template_family(family: str) -> str:
    try:
        return FAMILY_ALIASES[family]
    except KeyError as exc:
        supported = ", ".join(TEMPLATE_FAMILY_MAP)
        raise ValueError(f"No Manim template is configured for {family!r}. Supported families: {supported}") from exc


def template_path_for_family(family: str) -> Path:
    return TEMPLATE_DIR / TEMPLATE_FILES[_template_family(family)]


def _entity_confidence(item: Any) -> float:
    """Return the NER confidence for a span, or -inf when the span has none."""
    # print(item)
    if isinstance(item, dict):
        # print(item)
        try:
            return float(item["confidence"])
        except (KeyError, TypeError, ValueError):
            return float("-inf")
    return float("-inf")


def _best_value(value: Any) -> Any:
    """Pick the highest-confidence span; fall back to first if no scores exist.

    The extractor emits ``{"text": ..., "confidence": ...}`` spans. The
    per-span threshold has already pruned low-confidence matches, so this
    only decides *which surviving span* is used when a field matched more
    than once. Ties resolve to the earlier span for determinism.
    """
    # print(value)
    if not isinstance(value, list):
        return value
    if not value:
        return None
    if len(value) == 1:
        return value[0]

    scored = [(_entity_confidence(v), i, v) for i, v in enumerate(value)]
    # print(scored)
    best_score, _, best_item = max(scored, key=lambda t: (t[0], -t[1]))
    # print(best_score, best_item)
    if best_score == float("-inf"):
        # No span carried a confidence — keep prior first-in-list behaviour.
        return value[0]
    return best_item


def _coerce_numeric_atom(value: Any) -> Any:
    """Convert numeric text to Python numbers; preserve symbolic expressions as strings."""
    if isinstance(value, bool) or value is None or isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return value
    raw = value.strip()
    if not raw:
        return raw
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, bool):
            return raw
        if isinstance(parsed, (int, float)):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return raw


def _extract_bracket_contents(text: str) -> str:
    """Extract the first balanced (), [] or {} group from a text span."""
    start = next((i for i, char in enumerate(text) if char in "([{"), None)
    if start is None:
        return text.strip()
    opening = text[start]
    closing = {"(": ")", "[": "]", "{": "}"}[opening]
    depth = 0
    for index in range(start, len(text)):
        if text[index] == opening:
            depth += 1
        elif text[index] == closing:
            depth -= 1
            if depth == 0:
                return text[start + 1:index]
    return text[start:].strip()


def _parse_vector_span(value):
    """Convert one NER vector span (plain or LaTeX) into a Python list."""
    if isinstance(value, (list, tuple)):
        return [_coerce_numeric_atom(item) for item in value]
    if not isinstance(value, str):
        raise ValueError(f"Unsupported vector entity: {value!r}")
    text = value.strip()
    if "=" in text and not text.startswith(("(", "[")):
        text = text.rsplit("=", 1)[-1].strip()
    # LaTeX \langle ... \rangle
    m = re.match(r"\\langle(.+?)\\rangle", text)
    if m:
        return [_coerce_numeric_atom(p) for p in _split_top_level_commas(m.group(1))]
    # LaTeX \begin{pmatrix}...\end{pmatrix} (column or row)
    latex_rows = _latex_matrix_rows(text)
    if latex_rows is not None:
        if len(latex_rows) == 1:
            return [_coerce_numeric_atom(c) for c in latex_rows[0]]
        if all(len(row) == 1 for row in latex_rows):
            return [_coerce_numeric_atom(row[0]) for row in latex_rows]
        raise ValueError(f"Could not interpret LaTeX vector {value!r}")
    # \left( ... \right) and literal forms.
    text = text.replace(r"\left", "").replace(r"\right", "")
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)):
            return [_coerce_numeric_atom(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass
    inner = _extract_bracket_contents(text)
    parts = _split_top_level_commas(inner)
    if len(parts) < 2:
        raise ValueError(f"Could not parse vector entity {value!r}; expected at least two components.")
    return [_coerce_numeric_atom(item) for item in parts]


def _is_comma_vector_string(s: Any) -> bool:
    """True if ``s`` encodes a whole vector (comma-separated, or bracketed)."""
    return isinstance(s, str) and ("," in s or s.strip().startswith(("(", "[")))


def _parse_vectors_parameter(value: Any) -> list[list[Any]]:
    """Convert NER vector entities into nested Python lists for Manim.

    Accepts every shape the extractor can produce, including the ones where a
    single text span is split into comma-strings:

        [[1, 1, 1], [1, -1, 0]]           (canonical)
        ["1,1,1", "1,-1,0"]               (list of comma-strings)
        [["1,1,1", "1,-1,0"]]             (one-level nested comma-strings)
        ["(1,1,1)", "(1,-1,0)"]           (bracketed strings)
        "(1,1,1), (1,-1,0)"               (single string, several vectors)
        ["1", "1", "1"]                   (scalar-string components — one vector)
        [1, 2, 3]                         (scalar components — one vector)
    """
    # --- Unwrap a serialized string and recurse on its literal value -----
    if isinstance(value, str):
        raw = value.strip()
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return _parse_vectors_parameter(parsed)
        return [_parse_vector_span(raw)]

    if not isinstance(value, (list, tuple)) or not value:
        return []

    # --- Nested list whose items are each complete comma-strings --------
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        inner = value[0]
        if inner and all(_is_comma_vector_string(x) for x in inner):
            return [_parse_vector_span(x) for x in inner]

    # --- Flat list of complete comma-strings ----------------------------
    if all(_is_comma_vector_string(x) for x in value):
        return [_parse_vector_span(x) for x in value]

    # --- Flat list of scalar components -> one vector -------------------
    if all(not isinstance(item, (list, tuple)) for item in value):
        if len(value) >= 2:
            return [[_coerce_numeric_atom(item) for item in value]]

    # --- General case (mix of lists, bracketed strings, scalars) --------
    result: list[list[Any]] = []
    for item in value:
        if (
            isinstance(item, (list, tuple))
            and item
            and all(_is_comma_vector_string(x) for x in item)
        ):
            result.extend(_parse_vector_span(x) for x in item)
            continue
        try:
            result.append(_parse_vector_span(item))
        except (TypeError, ValueError):
            continue
    return result

# Matches "diag(...)", "\operatorname{diag}(...)", "\mathrm{diag}(...)".
_DIAG_RE = re.compile(
    r"^(?:\\operatorname\s*\{\s*diag\s*\}|\\mathrm\s*\{\s*diag\s*\}|diag)\s*",
    re.I,
)


def _parse_diag_expression(raw: str) -> list[list[Any]] | None:
    """Parse ``diag(a,b,c)`` (or its LaTeX aliases) into a nested list.

    Returns a square matrix with ``a, b, c`` on the diagonal and zeros
    elsewhere, or ``None`` if ``raw`` is not a ``diag(...)`` expression.
    """
    m = _DIAG_RE.match(raw)
    if not m:
        return None
    rest = raw[m.end():].lstrip()
    rest = re.sub(r"^\\left\s*", "", rest)  # tolerate \left(
    if not rest.startswith("("):
        return None
    depth = 0
    end = None
    for i, ch in enumerate(rest):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None
    inner = rest[1:end].strip()
    # Accept both "diag(1,3,5)" and "diag([1,3,5])".
    if inner.startswith("[") and inner.endswith("]"):
        candidate = inner[1:-1].strip()
        # Only treat it as a flat diagonal list if there are no nested brackets.
        if "[" not in candidate and "]" not in candidate:
            inner = candidate
    entries = [_coerce_numeric_atom(p) for p in _split_top_level_commas(inner)]
    if not entries:
        return None
    n = len(entries)
    matrix = [[0] * n for _ in range(n)]
    for i, value in enumerate(entries):
        matrix[i][i] = value
    return matrix

def _parse_matrix_parameter(value):
    if isinstance(value, (list, tuple)) and value and all(isinstance(row, (list, tuple)) for row in value):
        return [[_coerce_numeric_atom(item) for item in row] for row in value]
    if isinstance(value, (list, tuple)):
        value = _best_value(value)
    if not isinstance(value, str):
        return None
    raw = value.strip()

    # NEW: LaTeX matrix environments.
    latex_rows = _latex_matrix_rows(raw)
    if latex_rows is not None:
        widths = {len(row) for row in latex_rows}
        if len(widths) != 1:
            return None
        return [[_coerce_numeric_atom(c) for c in row] for row in latex_rows]

    # diag(...) / \operatorname{diag}(...)
    diag_matrix = _parse_diag_expression(raw)
    if diag_matrix is not None:
        return diag_matrix

    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, (list, tuple)) and all(isinstance(row, (list, tuple)) for row in parsed):
            return [[_coerce_numeric_atom(item) for item in row] for row in parsed]
    except (ValueError, SyntaxError):
        pass
    # Parse Python-style nested brackets even when entries are symbolic (for
    # example [[a, 0], [0, 2]]), which ast.literal_eval correctly rejects.
    if raw.startswith("[") and raw.endswith("]"):
        row_texts = _split_top_level_commas(raw[1:-1])
        rows: list[list[Any]] = []
        for row_text in row_texts:
            row_inner = _extract_bracket_contents(row_text)
            entries = _split_top_level_commas(row_inner)
            if not entries:
                rows = []
                break
            rows.append([_coerce_numeric_atom(item) for item in entries])
        if rows and all(len(row) == len(rows[0]) for row in rows):
            return rows
    # Fall back to the project's mathematical matrix parser for LaTeX or
    # symbolic matrices, then convert the result to JSON-friendly values.
    try:
        from math_preproc import parse_matrix
        parsed_matrix = parse_matrix(raw)
        converted = []
        for i in range(parsed_matrix.rows):
            row = []
            for j in range(parsed_matrix.cols):
                item = parsed_matrix[i, j]
                if getattr(item, "is_Integer", False):
                    row.append(int(item))
                elif getattr(item, "is_Float", False):
                    row.append(float(item))
                else:
                    row.append(str(item))
            converted.append(row)
        return converted
    except Exception:
        return None


def _normalize_sample_point(value: Any) -> Any:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "right": "right_endpoint", "right_end": "right_endpoint",
        "right_endpoint": "right_endpoint", "right_endpoints": "right_endpoint",
        "left": "left_endpoint", "left_end": "left_endpoint",
        "left_endpoint": "left_endpoint", "left_endpoints": "left_endpoint",
        "midpoint": "midpoint", "mid": "midpoint", "middle": "midpoint",
    }
    return aliases.get(normalized, value)

# Match one integral bound: either a braced group like "{\pi/2}" or a
# single LaTeX atom like "0" or "\pi".
_INTEGRAL_BOUND = r"(?:\{([^{}]*)\}|(\\[A-Za-z]+|[^\s{}]))"
_INTEGRAL_LOW_HIGH_RE = re.compile(
    r"(?:\\displaystyle\s*)?\\int\s*(?:\\limits)?\s*_\s*"
    + _INTEGRAL_BOUND + r"\s*\^\s*" + _INTEGRAL_BOUND
)
_INTEGRAL_HIGH_LOW_RE = re.compile(
    r"(?:\\displaystyle\s*)?\\int\s*(?:\\limits)?\s*\^\s*"
    + _INTEGRAL_BOUND + r"\s*_\s*" + _INTEGRAL_BOUND
)


def _latex_integral_bounds(text: str) -> tuple[str, str] | None:
    """Extract (low, high) bounds from a LaTeX definite integral.

    Handles both ``\\int_<low>^<high>`` and ``\\int^<high>_<low>`` orders, with
    each bound either braced (``_{\\pi/2}``) or a single LaTeX atom
    (``_0``, ``_\\pi``).  Returns ``None`` if the string is not exactly a
    bare integral bound expression (e.g. it contains an integrand).
    """
    match = _INTEGRAL_LOW_HIGH_RE.fullmatch(text)
    if match:
        low = match.group(1) if match.group(1) is not None else match.group(2)
        high = match.group(3) if match.group(3) is not None else match.group(4)
        return low.strip(), high.strip()
    match = _INTEGRAL_HIGH_LOW_RE.fullmatch(text)
    if match:
        high = match.group(1) if match.group(1) is not None else match.group(2)
        low = match.group(3) if match.group(3) is not None else match.group(4)
        return low.strip(), high.strip()
    return None


def _normalize_interval_text(value: Any) -> Any:
    """Convert interval phrases such as "from 0 to 1" into the canonical "[0,1]" form.

    Values that already use bracketed interval notation (e.g. "[0,1]",
    "(0,1]") are returned unchanged.  LaTeX definite-integral bounds
    (e.g. "\\int_0^1" → "[0,1]", "\\int_{\\pi/2}^{\\pi}" → "[\\pi/2,\\pi]")
    are converted to the same canonical form.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    # Already a bracketed interval, e.g. "[0,1]" or "(0,1]".
    if re.fullmatch(r"[\[(]\s*[^,]+?\s*,\s*[^,]+?\s*[\])]", text):
        return text
    # "from <left> to <right>" phrasing.
    match = re.fullmatch(r"from\s+(.+?)\s+to\s+(.+)", text, re.I)
    if match:
        left = match.group(1).strip()
        right = match.group(2).strip().rstrip(".")
        return f"[{left},{right}]"
    # LaTeX definite integral, e.g. "\int_0^1" or "\int_{\pi/2}^{\pi}".
    bounds = _latex_integral_bounds(text)
    if bounds is not None:
        low, high = bounds
        return f"[{low},{high}]"
    return value

def _split_top_level_commas(text: str) -> list[str]:
    """Split a coordinate tuple without splitting commas inside function calls."""
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(text):
        if char in "([{":
            stack.append(char)
        elif char in ")]}":
            if stack and stack[-1] == pairs[char]:
                stack.pop()
        elif char == "," and not stack:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return [part for part in parts if part]


_LATEX_MATRIX_ENVS = r"pmatrix|bmatrix|Bmatrix|vmatrix|Vmatrix|matrix|smallmatrix"
_LATEX_MATRIX_RE = re.compile(
    rf"\\begin\{{({_LATEX_MATRIX_ENVS})\*?\}}(.*?)\\end\{{\1\*?\}}",
    re.S,
)

def _latex_matrix_rows(text):
    """Return list-of-rows of raw cell strings from a LaTeX matrix env, or None."""
    m = _LATEX_MATRIX_RE.search(text.strip())
    if not m:
        return None
    body = m.group(2).strip()
    if not body:
        return None
    rows = []
    for row_text in re.split(r"\\\\", body):
        row_text = row_text.strip()
        if not row_text:
            continue
        rows.append([cell.strip() for cell in row_text.split("&")])
    return rows or None


_FUNC_NAMES = r"sin|cos|tan|arcsin|arccos|arctan|sinh|cosh|tanh|ln|log|exp|sqrt"

def _canonicalize_curve_coordinate(expression):
    text = str(expression).strip()
    # 1. Strip LaTeX backslashes from function names.
    text = re.sub(rf"\\(?={_FUNC_NAMES}\b)", "", text)
    # 2. "2cos t"  /  ")cos t"  /  "]sqrt(x)"  ->  insert '*' before the name.
    text = re.sub(rf"(?<=[0-9\)\]])\s*(?=(?:{_FUNC_NAMES})\b)", "*", text)
    # 3. "sin t" -> "sin(t)"  (bare argument form only).
    text = re.sub(rf"\b({_FUNC_NAMES})\s+([A-Za-z0-9_]+)\b", r"\1(\2)", text)
    # 4. ")cos(t)"  /  ")t"  /  ")(…)"  ->  insert '*' after a closing paren.
    text = re.sub(r"(?<=\))\s*(?=[A-Za-z(])", "*", text)
    return text


def _parse_curve_components(value: Any) -> list[str] | None:
    """Convert a CURVE NER span such as ``(t, t^2)`` to coordinate strings."""
    if isinstance(value, (list, tuple)):
        if len(value) in (2, 3) and all(
            not isinstance(item, (list, tuple, dict))
            and not (isinstance(item, str) and item.strip().startswith(("(", "[")) and item.strip().endswith((")", "]")))
            for item in value
        ):
            return [_canonicalize_curve_coordinate(item) for item in value]
        if len(value) == 1:
            return _parse_curve_components(_best_value(list(value)))
        if value:
            # NER can emit repeated curve spans. Use the first complete span;
            # the editable JSON remains available for user correction.
            return _parse_curve_components(_best_value(list(value)))
        return None

    if not isinstance(value, str):
        return None
    text = value.strip().replace(r"\left", "").replace(r"\right", "")
    if not text:
        return None
    if "=" in text:
        text = text.rsplit("=", 1)[-1].strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple)) and len(parsed) in (2, 3):
            return [_canonicalize_curve_coordinate(item) for item in parsed]
    except (ValueError, SyntaxError):
        pass

    opening_index = next((i for i, char in enumerate(text) if char in "(["), None)
    if opening_index is not None:
        opening = text[opening_index]
        closing = ")" if opening == "(" else "]"
        depth = 0
        closing_index = None
        for i in range(opening_index, len(text)):
            if text[i] == opening:
                depth += 1
            elif text[i] == closing:
                depth -= 1
                if depth == 0:
                    closing_index = i
                    break
        if closing_index is not None:
            text = text[opening_index + 1:closing_index]

    components = _split_top_level_commas(text)
    return [_canonicalize_curve_coordinate(item) for item in components] if len(components) in (2, 3) else None


def _extract_curve_assignment(text: str) -> list[str] | None:
    """Read gamma(t) = (...) from question/proof text as a conservative fallback."""
    marker = re.search(r"(?:gamma|γ|\\gamma)\s*\(\s*t\s*\)\s*=", text, re.I)
    if not marker:
        return None
    rhs = text[marker.end():].lstrip()
    if not rhs or rhs[0] not in "([":
        return None
    opening = rhs[0]
    closing = ")" if opening == "(" else "]"
    depth = 0
    for index, char in enumerate(rhs):
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return _parse_curve_components(rhs[:index + 1])
    return None


def default_parameters(family: str, extracted: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create template inputs, converting learned NER spans to Manim data types."""
    template_family = _template_family(family)
    extracted = extracted or {}
    values: dict[str, Any] = {field: None for field in TEMPLATE_PARAMETER_FIELDS[template_family]}
    for field in values:
        if field not in extracted:
            continue
        value = extracted[field]
        if field == "vectors":
            # Each recognized input vector is its own NER entity. Convert the
            # source-text spans into nested Python lists before showing/passing
            # the parameters to the Manim template.
            values[field] = _parse_vectors_parameter(value)
        elif field == "matrix":
            # Matrix NER produces a text span such as "[[0,2],[2,0]]"; Manim
            # templates receive a nested list such as [[0, 2], [2, 0]].
            values[field] = _parse_matrix_parameter(value)
        elif field == "curve":
            # One CURVE span such as "(t, t^2)" becomes the list expected by
            # CurvatureCurve2D/CurvatureCurve3D.
            values[field] = _parse_curve_components(value)
        else:
            values[field] = _best_value(value)
    if template_family == "Integration":
        values["sample_point"] = _normalize_sample_point(values.get("sample_point"))
        values["interval"] = _normalize_interval_text(values.get("interval"))
    return values


def heuristic_parameter_suggestions(family: str, question: str, proof: str) -> dict[str, Any]:
    """Suggest missing template-only values; suggestions remain editable by the user.

    NER remains the learned extraction mechanism.  These conservative suggestions
    are used to fill template inputs not represented in the NER schema (most
    notably a curve parametrization for the curvature template).
    """
    template_family = _template_family(family)
    text = f"{question}\n{proof}"
    result = default_parameters(family)

    if template_family == "Integration":
        m = re.search(r"(?:f\s*\(\s*x\s*\)|function)\s*=\s*([^,;\n.]+)", text, re.I)
        if m:
            result["function"] = m.group(1).strip().rstrip(".")
        if result["function"] is None:
            m = re.search(r"integral\s+of\s+(.+?)\s+from\s+([^\s,]+)\s+to\s+([^\s,.;]+)", text, re.I)
            if m:
                result["function"] = m.group(1).strip()
                result["interval"] = [m.group(2), m.group(3)]
        if result["interval"] is None:
            m = re.search(r"([\[(]\s*[-+\w./]+\s*,\s*[-+\w./]+\s*[\])])", text)
            if m:
                result["interval"] = m.group(1)
            else:
                m = re.search(r"from\s+([-+\w./]+)\s+to\s+([-+\w./]+)", text, re.I)
                if m:
                    result["interval"] = _normalize_interval_text(m.group(0))
        lower = text.lower()
        for phrase, label in (("right endpoint", "right_endpoint"), ("right endpoints", "right_endpoint"),
                              ("left endpoint", "left_endpoint"), ("left endpoints", "left_endpoint"),
                              ("midpoint", "midpoint")):
            if phrase in lower:
                result["sample_point"] = label
                break
        m = re.search(r"\b(?:equals?|equal to)\s+([-+]?(?:\d+(?:/\d+)?|pi(?:/\d+)?))", question, re.I)
        if m:
            result["integral_value"] = m.group(1)
        return result

    if template_family == "Vector":
        found = []
        for m in re.finditer(r"(?:v\s*\d+|u\s*\d*)\s*=\s*[\[(]([^\])]+)[\])]", text, re.I):
            parts = [part.strip() for part in m.group(1).split(",")]
            if len(parts) >= 2:
                found.append(parts)
        if len(found) >= 2:
            result["vectors"] = found
        return result

    if template_family == "Matrix":
        m = re.search(r"(?:A|matrix)\s*=\s*(\[\s*\[.*?\]\s*\])", text, re.I | re.S)
        if m:
            try:
                result["matrix"] = ast.literal_eval(m.group(1))
            except (ValueError, SyntaxError):
                pass
        return result

    if template_family == "Curvature":
        result["curve"] = _extract_curve_assignment(text)
        m = re.search(r"(?:at|when|evaluate\s+at)\s+t\s*(?:=|equals?\s+to?)\s*([^\s,.;]+)", text, re.I)
        if m:
            result["evaluation_point"] = m.group(1)
        return result

    return result


def merge_parameter_suggestions(family: str, extracted: dict[str, Any], question: str, proof: str) -> dict[str, Any]:
    """Merge learned NER output with editable suggestions for missing inputs."""
    params = default_parameters(family, extracted)
    suggested = heuristic_parameter_suggestions(family, question, proof)
    for key, value in suggested.items():
        current = params.get(key)
        if current is None or current == "" or current == []:
            if value is not None:
                params[key] = value
    # If NER only found one of the two Gram-Schmidt inputs, prefer a complete
    # two-vector text suggestion when it is available rather than passing an
    # inevitably incomplete input object to the renderer.
    if _template_family(family) == "Vector":
        current_vectors = params.get("vectors")
        suggested_vectors = suggested.get("vectors")
        if isinstance(current_vectors, list) and len(current_vectors) < 2 and isinstance(suggested_vectors, list) and len(suggested_vectors) >= 2:
            params["vectors"] = suggested_vectors
    if _template_family(family) == "Integration":
        params["sample_point"] = _normalize_sample_point(params.get("sample_point"))
        params["interval"] = _normalize_interval_text(params.get("interval"))
    # A lone detected vector is not enough for Gram-Schmidt; keep it visible so
    # the user can add the missing vector in the editable JSON object.
    if _template_family(family) == "Vector":
        # Heuristic completions can also be source-text spans; normalize them
        # through the same adapter used for NER output.
        params["vectors"] = _parse_vectors_parameter(params.get("vectors") or [])
    if _template_family(family) == "Matrix":
        params["matrix"] = _parse_matrix_parameter(params.get("matrix"))
    return params


def choose_scene_class(family: str, parameters: dict[str, Any], requested: str | None = None) -> str:
    """Resolve a template class using the selected input dimensions."""
    template_family = _template_family(family)
    if requested and requested.strip():
        return requested.strip()
    if template_family == "Matrix":
        matrix = parameters.get("matrix")
        if isinstance(matrix, str):
            try:
                matrix = ast.literal_eval(matrix)
            except (ValueError, SyntaxError):
                matrix = None
        dimension = len(matrix) if isinstance(matrix, (list, tuple)) else 2
        if dimension not in (2, 3):
            raise ValueError("The matrix Manim template supports 2x2 or 3x3 matrices.")
        return f"EigenDiagonal{dimension}D"
    if template_family == "Curvature":
        curve = parameters.get("curve")
        dimension = len(curve) if isinstance(curve, (list, tuple)) else 2
        if dimension not in (2, 3):
            raise ValueError("A curvature curve must have either two or three coordinate functions.")
        return f"CurvatureCurve{dimension}D"
    if template_family == "Vector":
        dimension = _vector_dimension(parameters.get("vectors"))
        if dimension not in (2, 3):
            raise ValueError("The Gram-Schmidt template supports 2-D or 3-D vectors only.")
        return f"GramSchmidt{dimension}D"
    return DEFAULT_SCENE_CLASSES[template_family]

def _vector_dimension(vectors: Any) -> int | None:
    """Return the component count of the first input vector, or None if undecidable.

    ``vectors`` is expected to be the nested-list form produced by
    ``_parse_vectors_parameter`` (e.g. ``[[1, 2, 3], [4, 5, 6]]``), so the
    ambient dimension is the length of any one vector.  Validation has already
    guaranteed that every vector shares the same dimension by the time this
    is called from ``render_manim``, so checking the first one is enough.
    """
    if not isinstance(vectors, (list, tuple)) or not vectors:
        return None
    first = vectors[0]
    if not isinstance(first, (list, tuple)) or not first:
        return None
    return len(first)

def validate_parameters(family: str, parameters: dict[str, Any]) -> list[str]:
    """Validate required inputs and parse mathematical values with SymPy."""
    from math_preproc import parse_expression, parse_interval, parse_matrix, parse_vectors

    template_family = _template_family(family)
    errors: list[str] = []
    for field in TEMPLATE_PARAMETER_FIELDS[template_family]:
        value = parameters.get(field)
        # `bound` is an explanatory label in the sequence visualization and is
        # optional for sequence styles that only provide a_n and L.
        if template_family == "Sequence Convergence" and field == "bound":
            continue
        if value is None or (isinstance(value, str) and not value.strip()) or value == []:
            errors.append(f"Missing required parameter: {field}")
    if errors:
        return errors

    try:
        if template_family == "Sequence Convergence":
            parse_expression(parameters["sequence"])
            parse_expression(parameters["limit"])
        elif template_family == "Integration":
            function = parse_expression(parameters["function"])
            a, b = parse_interval(parameters["interval"])
            if (b - a).is_number is False:
                errors.append("Integration interval endpoints must be numeric for this template.")
            if not (float(a) < float(b)):
                errors.append("Integration interval must satisfy left endpoint < right endpoint.")
            sample = _normalize_sample_point(parameters["sample_point"])
            if sample not in {"left_endpoint", "right_endpoint", "midpoint"}:
                errors.append("sample_point must be left, right, or midpoint (or their endpoint forms).")

            # NEW — alignment guarantee.
            import sympy as _sp
            _x = _sp.Symbol("x")
            _target = parse_expression(parameters["integral_value"])
            try:
                _exact = _sp.simplify(_sp.integrate(function, (_x, a, b)))
            except Exception:
                _exact = None
            if _exact is not None and _sp.simplify(_exact - _target) != 0:
                errors.append(
                    f"integral_value {parameters['integral_value']!r} does not match "
                    f"the exact integral of {parameters['function']!r} on "
                    f"{parameters['interval']!r}; SymPy gives {_exact}."
                )
        elif template_family == "Vector":
            vectors = parse_vectors(parameters["vectors"])
            if len(vectors) < 2:
                errors.append("At least two vectors are required for Gram-Schmidt.")
            elif len({v.rows for v in vectors}) != 1:
                errors.append("All input vectors must have the same dimension.")
            elif vectors[0].rows not in (2, 3):
                errors.append("Gram-Schmidt requires 2-D or 3-D vectors.")
        elif template_family == "Matrix":
            matrix = parse_matrix(parameters["matrix"])
            if matrix.rows != matrix.cols or matrix.rows not in (2, 3):
                errors.append("The matrix template requires a square 2x2 or 3x3 matrix.")
        elif template_family == "Curvature":
            curve = parameters["curve"]
            if not isinstance(curve, (list, tuple)) or len(curve) not in (2, 3):
                errors.append("curve must be a list of two or three coordinate expressions, e.g. [\"t\", \"t^2\"].")
            else:
                for component in curve:
                    parse_expression(component)
            parse_expression(parameters["evaluation_point"])
    except Exception as exc:
        errors.append(f"Could not parse template parameters: {exc}")
    return errors


def compute_mathematical_report(family: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Calculate symbolic quantities shown beside the rendered video."""
    import sympy as sp
    from math_preproc import (
        curve_geometry, curve_geometry_3d, gs_summary, parse_expression,
        parse_interval, parse_matrix, pick_index_symbol, resolve_eigen,
        symbolic_riemann,
    )

    template_family = _template_family(family)
    stringify = lambda value: sp.sstr(value) if isinstance(value, sp.Basic) else str(value)
    if template_family == "Sequence Convergence":
        expr = parse_expression(parameters["sequence"])
        limit = parse_expression(parameters["limit"])
        index = pick_index_symbol(expr)
        return {
            "sequence_expression": stringify(expr),
            "limit": stringify(limit),
            "absolute_error_expression": stringify(sp.simplify(sp.Abs(expr - limit))),
            "bound_from_proof": parameters.get("bound"),
            "note": "The visualization's epsilon and illustrative tail index are display choices, not a proof-certified N.",
        }
    if template_family == "Integration":
        function = parse_expression(parameters["function"])
        x = sp.Symbol("x") if sp.Symbol("x") in function.free_symbols \
            else (sorted(function.free_symbols, key=str)[0] if function.free_symbols else sp.Symbol("x"))
        a, b = parse_interval(parameters["interval"])
        sample = _normalize_sample_point(parameters["sample_point"])
        sum_expr, n_symbol = symbolic_riemann(function, x, a, b, sample)
        exact = sp.simplify(sp.integrate(function, (x, a, b)))
        target = parse_expression(parameters["integral_value"])

        # NEW — alignment diagnostics.
        raw_provided = parameters.get("integral_value")
        aligned = bool(sp.simplify(exact - target) == 0)

        return {
            "integrand": stringify(function),
            "interval": [stringify(a), stringify(b)],
            "sample_point": sample,
            "symbolic_riemann_sum": stringify(sum_expr) if sum_expr is not None else None,
            "riemann_sum_index": str(n_symbol),
            "sympy_integral_value": stringify(exact),
            "provided_integral_value_raw": raw_provided,
            "provided_integral_value_parsed": stringify(target),
            "provided_value_matches_sympy": aligned,
            "misaligned": not aligned,
        }
    if template_family == "Vector":
        summary = gs_summary(parameters["vectors"], normalize=False)
        selected = {key: summary[key] for key in ("u1", "u2", "proj", "coef", "ip_v2_u1", "ip_u1_u2", "dimension") if key in summary}
        return {key: stringify(value) if not isinstance(value, sp.MatrixBase) else [stringify(v) for v in value] for key, value in selected.items()}
    if template_family == "Matrix":
        data = resolve_eigen(parse_matrix(parameters["matrix"]), order="descending")
        return {
            "matrix": [[stringify(v) for v in row] for row in data["matrix"].tolist()],
            "dimension": data["dimension"],
            "eigenvalues": [stringify(v) for v in data["eigenvalues"]],
            "eigenvectors": [[stringify(v) for v in vector] for vector in data["eigenvectors"]],
            "multiplicities": data["multiplicities"],
            "diagonalizable": data["diagonalizable"],
            "real_spectrum": data["real_spectrum"],
        }
    if template_family == "Curvature":
        curve = parameters["curve"]
        if len(curve) == 2:
            data = curve_geometry(curve, parameters["evaluation_point"])
            return {
                "dimension": 2,
                "point": [stringify(v) for v in data["point"]],
                "tangent": [stringify(v) for v in data["tangent"]],
                "normal": [stringify(v) for v in data["normal"]],
                "curvature": stringify(data["kappa"]),
                "osculating_radius": stringify(data["radius"]),
                "osculating_center": [stringify(v) for v in data["center"]] if data["center"] is not None else None,
            }
        data = curve_geometry_3d(curve, parameters["evaluation_point"])
        return {
            "dimension": 3,
            "point": [stringify(v) for v in data["point"]],
            "tangent": [stringify(v) for v in data["tangent"]] if data["tangent"] is not None else None,
            "normal": [stringify(v) for v in data["normal"]] if data["normal"] is not None else None,
            "binormal": [stringify(v) for v in data["binormal"]] if data["binormal"] is not None else None,
            "curvature": stringify(data["kappa"]),
            "torsion": stringify(data["torsion"]),
            "osculating_radius": stringify(data["radius"]),
        }
    return {}


def _write_scene_wrapper(template_path: Path, scene_class: str, parameters: dict[str, Any], wrapper_path: Path) -> None:
    template_dir = template_path.parent.resolve()
    project_root = PROJECT_ROOT.resolve()
    parameter_json = json.dumps(parameters, ensure_ascii=False)
    source = f'''"""Generated wrapper for a parameterized Manim template."""
import importlib.util
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path({str(project_root)!r})
TEMPLATE_DIR = Path({str(template_dir)!r})
TEMPLATE_PATH = Path({str(template_path.resolve())!r})
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TEMPLATE_DIR))
spec = importlib.util.spec_from_file_location("selected_template", TEMPLATE_PATH)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
BaseScene = getattr(module, {scene_class!r})
PARAMETERS = json.loads({parameter_json!r})

class RenderedScene(BaseScene):
    def __init__(self, **kwargs):
        merged = dict(PARAMETERS)
        merged.update(kwargs)
        super().__init__(**merged)
'''
    wrapper_path.write_text(source, encoding="utf-8")


def render_manim(
    family: str,
    parameters: dict[str, Any],
    scene_class: str | None = None,
    quality: str = "low",
    timeout_seconds: int = 900,
    progress_callback: Callable[[float, str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Render a scene and stream Manim stdout/stderr to ``log_callback``.

    The full log is also saved to generated/<run-id>/manim.log and returned.
    A background reader thread drains the subprocess pipe while the caller
    thread updates the Streamlit interface, avoiding Windows select() issues.
    """
    template_family = _template_family(family)
    template_path = template_path_for_family(family)
    if not template_path.exists():
        raise FileNotFoundError(f"Template file {template_path.name} is missing. Put it in {TEMPLATE_DIR}.")

    parameters = dict(parameters)
    if template_family == "Integration":
        parameters["sample_point"] = _normalize_sample_point(parameters.get("sample_point"))
        parameters["interval"] = _normalize_interval_text(parameters.get("interval"))
    errors = validate_parameters(family, parameters)
    if errors:
        raise ValueError("Template parameters are invalid:\n- " + "\n- ".join(errors))

    class_name = choose_scene_class(family, parameters, scene_class)
    mathematical_report = compute_mathematical_report(family, parameters)
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    run_dir = GENERATED_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    wrapper_path = run_dir / "generated_scene.py"
    media_dir = run_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "manim.log"
    report_path = run_dir / "report.json"
    _write_scene_wrapper(template_path, class_name, parameters, wrapper_path)

    quality_flag = {"low": "-ql", "medium": "-qm", "high": "-qh"}.get(quality)
    if quality_flag is None:
        raise ValueError("quality must be low, medium, or high")
    command = [
        sys.executable, "-m", "manim", quality_flag,
        "--media_dir", str(media_dir), "--disable_caching",
        str(wrapper_path), "RenderedScene",
    ]

    started = time.time()
    return_code: int | None = None
    timed_out = False
    log_lines: list[str] = []
    line_queue: queue.Queue[str | None] = queue.Queue()
    if progress_callback:
        progress_callback(0.03, "Preparing Manim scene…")
    if log_callback:
        log_callback("Starting Manim…\n$ " + " ".join(command) + "\n")

    try:
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Could not start Manim. Install it in the same Python environment as Streamlit: "
            "pip install -r requirements.txt"
        ) from exc

    def _read_output() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                line_queue.put(line)
        finally:
            line_queue.put(None)
            process.stdout.close()

    reader = threading.Thread(target=_read_output, name=f"manim-log-{run_id}", daemon=True)
    reader.start()
    eof = False
    last_progress = 0.0
    last_log_push = 0.0
    last_progress_message = "Manim is starting…"
    real_progress = 0.05  # Starting baseline

    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        while not eof or process.poll() is None:
            now = time.time()
            try:
                queued = line_queue.get(timeout=0.25)
            except queue.Empty:
                queued = ""
            if queued is None:
                eof = True
            elif queued:
                log_lines.append(queued)
                log_file.write(queued)
                log_file.flush()
                clean_q = queued.strip()
                if clean_q:
                    last_progress_message = clean_q[-180:]
                
                # --- PARSE MANIM PERCENTAGE LIVE ---
                pct_match = re.search(r'(\d+)\s*%', queued)
                if pct_match:
                    try:
                        val = float(pct_match.group(1))
                        if 0.0 <= val <= 100.0:
                            real_progress = max(real_progress, val / 100.0)
                    except ValueError:
                        pass

            elapsed = time.time() - started
            
            # Use real Manim progress if available, otherwise use a smooth fallback (scaled to ~35s instead of 900s)
            if real_progress > 0.05:
                current_fraction = real_progress
            else:
                current_fraction = min(0.85, 0.05 + (elapsed / 35.0) * 0.80)

            if progress_callback and (now - last_progress >= 0.2 or queued is None):
                progress_callback(current_fraction, f"Manim rendering… ({int(elapsed)}s) — {last_progress_message}")
                last_progress = now
            if log_callback and (now - last_log_push >= 0.45 or queued is None):
                log_callback("".join(log_lines))
                last_log_push = now

            if elapsed > timeout_seconds and process.poll() is None:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
                break

            if process.poll() is not None and eof:
                break

    return_code = process.poll()
    if return_code is None:
        try:
            return_code = process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait(timeout=5)
    reader.join(timeout=2)

    elapsed = time.time() - started
    log_text = "".join(log_lines)
    video_candidates = list(media_dir.rglob("RenderedScene.mp4"))
    video_path = max(video_candidates, key=lambda path: path.stat().st_mtime) if video_candidates else None
    success = return_code == 0 and video_path is not None and not timed_out
    report = {
        "run_id": run_id,
        "proof_family": family,
        "template_family": template_family,
        "template_file": template_path.name,
        "scene_class": class_name,
        "parameters": parameters,
        "mathematical_report": mathematical_report,
        "command": command,
        "return_code": return_code,
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 2),
        "success": success,
        "video_path": str(video_path) if video_path else None,
        "log_path": str(log_path),
        "message": "Rendered successfully." if success else "Manim did not produce the expected video; inspect the log.",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if log_callback:
        log_callback(log_text or "Manim did not write any console output.")
    if progress_callback:
        progress_callback(1.0, "Rendering complete." if success else "Rendering failed; see the console log.")
    return {**report, "report_path": str(report_path), "log_text": log_text}
