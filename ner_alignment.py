"""Training-time weak-label alignment for the project's parameter NER.

Known values in a proof record's ``parameters`` object are aligned to literal
spans in its question/proof. This module is used only to prepare training BIO
labels; it is not used for inference on a new proof.
"""

from __future__ import annotations

import re
from typing import Any

from parameter_schema import DERIVED_OR_DEFAULT_FIELDS, PARAMETER_FIELDS


def _normalize_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalize math punctuation/LaTeX commands while preserving source offsets."""
    out: list[str] = []
    offsets: list[int] = []
    index = 0
    latex_replacements = {
        r"\lvert": "|",
        r"\rvert": "|",
        r"\mid": "|",
        r"\vert": "|",
    }
    while index < len(text):
        replacement = next(
            (
                value
                for command, value in latex_replacements.items()
                if text.startswith(command, index)
            ),
            None,
        )
        if replacement is not None:
            for char in replacement:
                out.append(char)
                offsets.append(index)
            index += len(
                next(
                    command
                    for command in latex_replacements
                    if text.startswith(command, index)
                )
            )
            continue
        if text.startswith(r"\left", index):
            index += len(r"\left")
            continue
        if text.startswith(r"\right", index):
            index += len(r"\right")
            continue
        char = text[index]
        if char.isspace() or char in {"\\", "$", "*"}:
            index += 1
            continue
        char = {
            "−": "-",
            "–": "-",
            "×": "*",
            "·": "*",
            "π": "pi",
            "ε": "eps",
        }.get(char, char)
        for normalized_char in char.casefold():
            out.append(normalized_char)
            offsets.append(index)
        index += 1
    return "".join(out), offsets


def _format_value(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return ""
    return str(value)


def _interval_candidates(value: list[Any]) -> list[str]:
    """Return common literal and prose forms of a two-endpoint interval."""
    if len(value) != 2:
        return [_format_value(value)]
    left, right = (_format_value(item) for item in value)
    return [
        f"[{left}, {right}]",
        f"({left}, {right})",
        f"from {left} to {right}",
        f"between {left} and {right}",
    ]


def _curve_coordinate_aliases(expression: str) -> list[str]:
    """Return common equivalent textual forms of one curve coordinate."""
    raw = expression.strip()
    aliases = [raw]

    # Many proofs write cos t rather than cos(t), while datasets may store
    # either form. Restrict this alias to simple single-token arguments so
    # that nested expressions are not rewritten incorrectly.
    trig_alias = re.sub(
        r"\b(sin|cos|tan|sinh|cosh|tanh|exp|ln|log|sqrt)\s*\(\s*([A-Za-z0-9_]+)\s*\)",
        r"\1 \2",
        raw,
    )
    if trig_alias != raw:
        aliases.append(trig_alias)

    exp_aliases = [raw, trig_alias]
    for source in exp_aliases:
        exp_alias = re.sub(r"\bexp\s*\(\s*([A-Za-z0-9_]+)\s*\)", r"e^\1", source)
        if exp_alias not in aliases:
            aliases.append(exp_alias)
        # Textbooks commonly write e^t cos t, while data may store
        # exp(t)*cos(t). Normalize function-call parentheses after rewriting.
        smooth_alias = re.sub(
            r"\b(sin|cos|tan|sinh|cosh|tanh|ln|log|sqrt)\s*\(\s*([A-Za-z0-9_]+)\s*\)",
            r"\1 \2",
            exp_alias,
        )
        if smooth_alias not in aliases:
            aliases.append(smooth_alias)

    # LaTeX often writes \cos t and \sin t with a command slash; the shared
    # normalizer strips that slash, so no additional rewrite is required.
    return aliases


def _curve_candidates(value: Any) -> list[str]:
    """Format a 2D/3D parametrization as one tuple-like NER entity."""
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        # A few external datasets store the curve as a JSON/Python list string.
        try:
            import ast

            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)):
                value = parsed
            else:
                value = raw
        except (ValueError, SyntaxError):
            value = raw

    if isinstance(value, str):
        # Also support a serialized coordinate tuple in older/custom datasets.
        raw = value.strip()
        return [raw] if raw.startswith(("(", "[")) and raw.endswith((")", "]")) else []
    if not isinstance(value, (list, tuple)) or len(value) not in (2, 3):
        return []

    coordinates = [_format_value(item) for item in value]
    if any(not coordinate for coordinate in coordinates):
        return []

    # Build aliases coordinate-wise, then wrap the full parametrization. The
    # NER target is the complete coordinate tuple, not separate x(t), y(t)
    # entities: the inference adapter needs one CURVE span to reconstruct it.
    import itertools

    coordinate_options = [_curve_coordinate_aliases(item) for item in coordinates]
    candidates: list[str] = []
    for option in itertools.product(*coordinate_options):
        candidates.append("(" + ", ".join(option) + ")")
        candidates.append("[" + ", ".join(option) + "]")
    return candidates


def _value_candidates(value: Any, field: str) -> list[str]:
    """Format parameter values in likely mathematical surface forms."""
    candidates: list[str] = []
    if isinstance(value, bool) or value is None:
        return candidates

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return candidates

        if field == "curve":
            # String-valued curve records are normalized in the same way as
            # list-valued records below.
            candidates.extend(_curve_candidates(raw))
        elif field == "sample_point":
            sample_aliases = {
                "right_endpoint": "right",
                "left_endpoint": "left",
                "midpoint": "midpoint",
                "right": "right",
                "left": "left",
            }
            canonical = sample_aliases.get(raw.casefold(), raw.casefold())
            if canonical not in {"right", "left", "midpoint"}:
                return candidates
            candidates.append(canonical)
        elif raw.casefold() in {
            "any",
            "not_applicable",
            "none_real",
            "all_nonzero_vectors",
            "arbitrary_positive",
            "root_in_(0,1)",
            "root_in\\_(0,1)",
        }:
            return candidates
        else:
            candidates.append(raw)
            if field == "interval":
                interval_match = re.fullmatch(r"[\[(]\s*(.+?)\s*,\s*(.+?)\s*[\])]", raw)
                if interval_match:
                    left, right = interval_match.groups()
                    candidates.extend(
                        [
                            f"from {left} to {right}",
                            f"between {left} and {right}",
                        ]
                    )

    elif isinstance(value, (int, float)):
        candidates.append(str(value))

    elif isinstance(value, list):
        if not value:
            return candidates
        if field == "interval":
            candidates.extend(_interval_candidates(value))
        elif field == "curve":
            candidates.extend(_curve_candidates(value))
        elif field == "matrix":
            candidates.append(_format_value(value))
            # Compact scalar-identity matrices are often written as I or cI.
            if (
                value
                and all(isinstance(row, list) for row in value)
                and all(len(row) == len(value) for row in value)
            ):
                diagonal = [value[i][i] for i in range(len(value))]
                off_diagonal = [
                    value[i][j]
                    for i in range(len(value))
                    for j in range(len(value))
                    if i != j
                ]
                if all(item == 0 for item in off_diagonal) and len(set(diagonal)) == 1:
                    scalar = diagonal[0]
                    if scalar == 1:
                        candidates.extend(["I", "identity matrix"])
                    else:
                        candidates.append(f"{scalar}I")
        elif field == "vectors":
            # Each input vector is its own entity, but all entities use the
            # single canonical `vectors` field.
            if all(isinstance(vector, list) for vector in value):
                for vector in value:
                    candidates.append(
                        "(" + ", ".join(_format_value(x) for x in vector) + ")"
                    )
            else:
                # A single vector may be represented as [x, y, ...].
                candidates.append(
                    "(" + ", ".join(_format_value(x) for x in value) + ")"
                )
        else:
            candidates.append(_format_value(value))

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized, _ = _normalize_with_offsets(candidate)
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(candidate)
    return unique


def _context_is_relevant(text: str, field: str, start: int, end: int) -> bool:
    before = text[max(0, start - 120) : start].lower()
    after = text[end : min(len(text), end + 70)].lower()
    if field == "bound" and re.match(
        r"\s*(?:is\s+at\s+most|is\s+bounded\s+by|<=|≤|<)", after
    ):
        return True
    if field == "sample_point" and re.match(r"\s*[- ]?endpoints?\b", after):
        return True
    patterns = {
        "sequence": [r"a\s*[_\{]?n[_\}]?\s*=\s*$"],
        "limit": [
            r"(?:converges|tends|approaches)\s+to\s*$",
            r"limit\s+(?:is|equals)\s*$",
            r"(?:supremum|namely)\s*(?:,)?\s*$",
        ],
        "bound": [
            r"\|[^|]{0,60}\|\s*(?:=|<=|≤)\s*$",
            r"(?:error|absolute\s+difference|difference|bound)[^.;]{0,60}(?:is|equals|=|at\s+most)\s*$",
        ],
        "function": [r"f\s*\(\s*x\s*\)\s*=\s*$"],
        "interval": [r"(?:on|over|from)\s*$"],
        "integral_value": [
            # Integral followed by its evaluated result
            r"(?:integral|\\int).{0,120}(?:equals|is|=)\s*$",
            # Explicit mathematical equality or limit result
            r"(?:=|\\(?:to|rightarrow|longrightarrow)|→|equals|is|namely|therefore)\s*$",
        ],
        "sample_point": [
            r"(?:use|using|choose|chosen|sampled\s+at|with)\s+(?:the\s+)?$"
        ],
        "evaluation_point": [
            # explicit introducer + variable + connector; tolerant of $, \(, subscripts
            r"(?:at|when|evaluate(?:d)?\s+at)\s*[\s$\\(]*"
            r"(?:t|s|u|v|\\tau|[a-zA-Z])\s*_?\s*\{?\s*[A-Za-z0-9]*\s*\}?"
            r"\s*(?:=|equals?(?:\s+to)?|is|yields?|gives?)\s*$",
            # bare variable + connector; word-bounded so "output =" / "constant =" don't match
            r"\b(?:t|s|u|v|\\tau|[a-zA-Z])\s*_?\s*\{?\s*[A-Za-z0-9]*\s*\}?"
            r"\s*(?:=|equals?(?:\s+to)?|is)\s*$",
        ],
        "curve": [
            r"(?:gamma|γ|\\gamma)\s*\(\s*t\s*\)\s*=\s*$",
            r"(?:curve|parametrization|parametrisation)\s*(?:is|equals|=)\s*$",
        ],
        "matrix": [r"A\s*=\s*$"],
        "vectors": [r"(?:v\s*_?\s*[12]|u|v)\s*=\s*$"],
    }
    return any(
        re.search(pattern, before, re.IGNORECASE) for pattern in patterns.get(field, [])
    )


def parameters_for_ner(
    parameters: dict[str, Any] | None, family: str
) -> dict[str, Any]:
    """Select the canonical NER keys and normalize older vector data layouts."""
    if not isinstance(parameters, dict):
        return {}

    allowed_fields = PARAMETER_FIELDS.get(family, ())
    targets = {key: value for key, value in parameters.items() if key in allowed_fields}

    if family == "Vector Operations" and "vectors" not in targets:
        # Older records express a pair of inputs using `vector` + `onto`,
        # `vector` + `direction`, or `u` + `v`. Convert all of them into one
        # NER field instead of learning separate output labels.
        first = parameters.get("vector", parameters.get("u"))
        second = parameters.get(
            "onto", parameters.get("direction", parameters.get("v"))
        )
        if first is not None and second is not None:
            targets["vectors"] = [first, second]
        elif first is not None:
            targets["vectors"] = [first]

    # `any` is not one of the discrete sample-point classes requested by the
    # project, so do not create a misleading NER target for it.
    if str(targets.get("sample_point", "")).casefold() == "any":
        targets.pop("sample_point", None)
    if str(targets.get("evaluation_point", "")).casefold() == "any":
        targets.pop("evaluation_point", None)

    return targets


def _find_matches(
    text: str, normal_text: str, offset_map: list[int], candidate: str, field: str
) -> list[tuple[int, int]]:
    normalized, _ = _normalize_with_offsets(candidate)
    if not normalized:
        return []
    single_char_letter = len(normalized) == 1 and normalized.isalpha()
    allow_single_letter = field in {"function", "matrix", "evaluation_point", "limit"}
    if single_char_letter and not allow_single_letter:
        return []

    matches: list[tuple[int, int]] = []
    search_from = 0
    while True:
        position = normal_text.find(normalized, search_from)
        if position < 0:
            break
        end_normal = position + len(normalized)
        original_start = offset_map[position]
        original_end = offset_map[end_normal - 1] + 1

        # Guard against aligning a short value such as 1 inside 11.
        raw_left = text[original_start - 1] if original_start > 0 else " "
        raw_right = text[original_end] if original_end < len(text) else " "
        left_ok = not (normalized[0].isalnum() and raw_left.isalnum())
        right_ok = not (normalized[-1].isalnum() and raw_right.isalnum())
        if left_ok and right_ok:
            matches.append((original_start, original_end))
        search_from = position + 1
    return matches


def align_parameters_to_spans(
    text: str, parameters: dict[str, Any], family: str
) -> list[dict[str, Any]]:
    """Align known target values to non-overlapping character spans.

    When sequence and bound have the same value, the sequence is preferentially
    aligned to its `a_n = ...` occurrence, while the bound prefers the final
    occurrence (usually the error estimate in the proof). This prevents one
    piece of text from being assigned two BIO labels.
    """
    if not isinstance(parameters, dict):
        return []

    parameters = parameters_for_ner(parameters, family)
    family_fields = set(PARAMETER_FIELDS.get(family, ()))
    normal_text, offset_map = _normalize_with_offsets(text)
    candidates: list[tuple[int, int, str, str]] = []

    for field, value in parameters.items():
        if field in DERIVED_OR_DEFAULT_FIELDS or field not in family_fields:
            continue
        for surface_candidate in _value_candidates(value, field):
            matches = _find_matches(
                text, normal_text, offset_map, surface_candidate, field
            )
            if not matches:
                continue

            contextual = [
                match
                for match in matches
                if _context_is_relevant(text, field, match[0], match[1])
            ]
            if field == "bound":
                chosen = contextual[-1] if contextual else matches[-1]
            else:
                chosen = contextual[0] if contextual else matches[0]
            candidates.append((chosen[0], chosen[1], field, surface_candidate))

    # Prefer longer/more-specific expressions. Overlapping candidates cannot
    # both be represented in a single BIO sequence.
    candidates.sort(key=lambda item: (-(item[1] - item[0]), item[0], item[2]))
    selected: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    for start, end, field, expected in candidates:
        if any(
            start < used_end and end > used_start for used_start, used_end in occupied
        ):
            continue
        selected.append(
            {
                "start": start,
                "end": end,
                "field": field,
                "text": text[start:end],
                "matched_value": expected,
            }
        )
        occupied.append((start, end))

    selected.sort(key=lambda item: item["start"])
    return selected


def bio_labels_for_text(
    text: str,
    parameters: dict[str, Any],
    family: str,
    tokenizer,
    label2id: dict[str, int],
    max_length: int = 384,
):
    """Tokenise a proof and convert aligned character spans into BIO labels."""
    spans = align_parameters_to_spans(text, parameters, family)
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = encoded["offset_mapping"]
    tags = ["O"] * len(offsets)
    for span in spans:
        first = True
        for index, (token_start, token_end) in enumerate(offsets):
            if token_start == token_end:
                continue
            if token_start < span["end"] and token_end > span["start"]:
                tags[index] = ("B-" if first else "I-") + span["field"]
                first = False
    encoded["labels"] = [label2id.get(tag, label2id["O"]) for tag in tags]
    encoded["offset_mapping"] = offsets
    encoded["spans"] = spans
    return encoded
