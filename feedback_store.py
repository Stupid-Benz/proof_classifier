"""Persist user-reviewed extraction corrections for future NER training.

Feedback is stored separately from the hand-authored dataset. The NER trainer
loads these records into its training partition only, never its held-out test
partition, so user feedback cannot inflate validation metrics.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from parameter_schema import PARAMETER_FIELDS, SUBJECT_FAMILIES

PROJECT_ROOT = Path(__file__).resolve().parent
FEEDBACK_DIR = PROJECT_ROOT / "feedback"
FEEDBACK_PATH = FEEDBACK_DIR / "ner_feedback.json"
_LOCK = threading.Lock()


def _subject_for_family(family: str) -> str:
    for subject, families in SUBJECT_FAMILIES.items():
        if family in families:
            return subject
    raise ValueError(f"Unsupported proof family for feedback: {family!r}")


def ner_parameters_from_template(template_parameters: dict[str, Any], family: str) -> dict[str, Any]:
    """Keep only fields that the NER model is trained to predict for a family."""
    if not isinstance(template_parameters, dict):
        raise ValueError("Reviewed template parameters must be a JSON object.")
    allowed = PARAMETER_FIELDS.get(family)
    if allowed is None:
        raise ValueError(f"Unknown proof family: {family!r}")
    values = {
        field: template_parameters[field]
        for field in allowed
        if field in template_parameters and template_parameters[field] not in (None, "", [])
    }
    if not values:
        raise ValueError(
            f"No NER-supported fields were supplied for {family}. Expected at least one of: "
            + ", ".join(allowed)
        )
    return values


def _fingerprint(question: str, proof: str, family: str, parameters: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "question": question.strip(),
            "proof": proof.strip(),
            "proof_family": family,
            "parameters": parameters,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Feedback file must contain a JSON list of records: {path}")
    return payload


def save_reviewed_feedback(
    *,
    question: str,
    proof: str,
    family: str,
    template_parameters: dict[str, Any],
    initial_predictions: dict[str, Any] | None = None,
    classification: dict[str, Any] | None = None,
    path: str | Path = FEEDBACK_PATH,
) -> dict[str, Any]:
    """Save one explicitly reviewed proof and its corrected NER parameter values.

    Repeated submission of identical content is deduplicated by a stable
    fingerprint. This stores a training example; it does not train a model.
    """
    question = str(question or "").strip()
    proof = str(proof or "").strip()
    if not question and not proof:
        raise ValueError("A feedback example needs a question or proof.")

    ner_parameters = ner_parameters_from_template(template_parameters, family)
    target_path = Path(path)
    fingerprint = _fingerprint(question, proof, family, ner_parameters)

    record = {
        "id": f"FB-{uuid.uuid4().hex[:12].upper()}",
        "subject": _subject_for_family(family),
        "proof_family": family,
        "proof_style": "user_reviewed_feedback",
        "proof_format": "mixed",
        "theorem": "User-reviewed application feedback",
        "question": question,
        "proof": proof,
        "parameters": ner_parameters,
        "animation_notes": "",
        "feedback_metadata": {
            "kind": "user_reviewed_ner_feedback",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "initial_ner_predictions": initial_predictions or {},
            "classification": classification or {},
            "reviewed_fields": sorted(ner_parameters),
        },
    }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        records = _read_records(target_path)
        for existing in records:
            metadata = existing.get("feedback_metadata", {})
            if metadata.get("fingerprint") == fingerprint:
                return {
                    "saved": False,
                    "duplicate": True,
                    "record_id": existing.get("id"),
                    "path": str(target_path),
                    "parameters": ner_parameters,
                }
        records.append(record)
        temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(target_path)

    return {
        "saved": True,
        "duplicate": False,
        "record_id": record["id"],
        "path": str(target_path),
        "parameters": ner_parameters,
    }


def count_feedback_records(path: str | Path = FEEDBACK_PATH) -> int:
    """Return the number of stored review records, or zero when none exist."""
    try:
        return len(_read_records(Path(path)))
    except (OSError, json.JSONDecodeError, ValueError):
        return 0
