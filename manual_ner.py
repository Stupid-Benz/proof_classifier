"""Exact character-span annotation helpers for supervised proof-parameter NER.

Annotations use offsets in parameter_schema.build_text(question, proof), so the
trainer can convert them directly to BIO labels without weak alignment.
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

from parameter_schema import PARAMETER_FIELDS, SUBJECT_FAMILIES, build_text

PROJECT_ROOT = Path(__file__).resolve().parent
MANUAL_NER_PATH = PROJECT_ROOT / "feedback" / "manual_ner_annotations.json"
_LOCK = threading.Lock()


def find_exact_occurrences(text: str, phrase: str) -> list[tuple[int, int]]:
    """Return every exact, case-sensitive occurrence as [start, end) offsets."""
    if not isinstance(text, str) or not isinstance(phrase, str) or not phrase:
        return []
    matches: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(phrase, cursor)
        if start < 0:
            break
        matches.append((start, start + len(phrase)))
        cursor = start + 1  # permit overlapping matches
    return matches


def validate_manual_entities(
    text: str, family: str, entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate exact span offsets, labels, text values, and overlap constraints."""
    if family not in PARAMETER_FIELDS:
        raise ValueError(f"Unknown proof family: {family!r}")
    if not isinstance(entities, list) or not entities:
        raise ValueError("Add at least one manually labelled entity.")
    allowed = set(PARAMETER_FIELDS[family])
    cleaned: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []

    for index, entity in enumerate(entities, start=1):
        if not isinstance(entity, dict):
            raise ValueError(f"Annotation {index} must be an object.")
        field = str(entity.get("field", ""))
        if field not in allowed:
            raise ValueError(
                f"Annotation {index} uses {field!r}, which is not a NER field for {family}. "
                f"Allowed fields: {', '.join(sorted(allowed)) or '(none)'}"
            )
        try:
            start, end = int(entity["start"]), int(entity["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Annotation {index} needs integer start/end offsets.") from exc
        if start < 0 or end <= start or end > len(text):
            raise ValueError(f"Annotation {index} has invalid offsets [{start}, {end}).")
        actual = text[start:end]
        supplied = entity.get("text")
        if supplied is not None and supplied != actual:
            raise ValueError(
                f"Annotation {index} text does not match text[{start}:{end}]. "
                "Recreate the annotation after editing the source proof."
            )
        if any(start < old_end and end > old_start for old_start, old_end in occupied):
            raise ValueError(f"Annotation {index} overlaps another annotation. BIO spans cannot overlap.")
        occupied.append((start, end))
        cleaned.append({"start": start, "end": end, "field": field, "text": actual})

    return sorted(cleaned, key=lambda item: (item["start"], item["end"]))


def _subject_for_family(family: str) -> str:
    for subject, families in SUBJECT_FAMILIES.items():
        if family in families:
            return subject
    raise ValueError(f"Unsupported proof family: {family!r}")


def _read_annotation_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError(f"Manual annotation file must contain a JSON list: {path}")
    return payload


def save_manual_annotation(
    *,
    question: str,
    proof: str,
    family: str,
    entities: list[dict[str, Any]],
    complete: bool = False,
    path: str | Path = MANUAL_NER_PATH,
) -> dict[str, Any]:
    """Persist exact entity spans for future NER training.

    When ``complete`` is false, tokens outside the selected spans are ignored
    by the loss function. When true, the user confirms that every supported
    entity has been annotated, so remaining non-special tokens receive O tags.
    """
    question = str(question or "").strip()
    proof = str(proof or "").strip()
    if not question and not proof:
        raise ValueError("A manual annotation needs a question or proof.")
    text = build_text(question, proof)
    cleaned = validate_manual_entities(text, family, entities)
    fingerprint_payload = {
        "question": question,
        "proof": proof,
        "family": family,
        "manual_entities": cleaned,
        "manual_annotation_complete": bool(complete),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    record = {
        "id": f"MAN-{uuid.uuid4().hex[:12].upper()}",
        "subject": _subject_for_family(family),
        "proof_family": family,
        "theorem": "Manually annotated application proof",
        "question": question,
        "proof": proof,
        "parameters": {},
        "animation_notes": "",
        "manual_entities": cleaned,
        "manual_annotation_complete": bool(complete),
        "feedback_metadata": {
            "kind": "manual_ner_annotation",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "annotation_count": len(cleaned),
            "annotation_fields": sorted({item["field"] for item in cleaned}),
            "annotation_complete": bool(complete),
        },
    }

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        records = _read_annotation_records(target)
        for old in records:
            if old.get("feedback_metadata", {}).get("fingerprint") == fingerprint:
                return {"saved": False, "duplicate": True, "record_id": old.get("id"), "path": str(target)}
        records.append(record)
        temp = target.with_suffix(target.suffix + ".tmp")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temp.replace(target)

    return {"saved": True, "duplicate": False, "record_id": record["id"], "path": str(target)}


def count_manual_annotations(path: str | Path = MANUAL_NER_PATH) -> int:
    try:
        return len(_read_annotation_records(Path(path)))
    except (OSError, ValueError, json.JSONDecodeError):
        return 0


def encode_manual_spans(
    text: str,
    spans: list[dict[str, Any]],
    tokenizer,
    label2id: dict[str, int],
    max_length: int = 384,
    complete: bool = False,
) -> dict[str, Any]:
    """Tokenize manual spans into BIO labels.

    Partial annotation masks all unselected tokens with -100 so they do not
    become accidental negative examples. Complete annotation assigns O to
    every unselected, non-special token; special tokens always remain -100.
    """
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
        add_special_tokens=True,
    )
    offsets = encoded["offset_mapping"]
    labels = [-100] * len(offsets)
    if complete:
        for token_index, (token_start, token_end) in enumerate(offsets):
            if token_start != token_end:
                labels[token_index] = label2id["O"]

    for span in spans:
        token_indices = [
            i for i, (token_start, token_end) in enumerate(offsets)
            if token_start != token_end
            and token_start < int(span["end"])
            and token_end > int(span["start"])
        ]
        if not token_indices:
            raise ValueError(
                f"Manual entity {span['text']!r} is outside the tokenizer's max length. "
                "Increase --max-length or shorten the input proof."
            )
        covered_start = min(int(offsets[i][0]) for i in token_indices)
        covered_end = max(int(offsets[i][1]) for i in token_indices)
        if covered_start > int(span["start"]) or covered_end < int(span["end"]):
            raise ValueError(
                f"Manual entity {span['text']!r} is truncated by --max-length. "
                "Increase --max-length or shorten the input proof."
            )
        for order, token_index in enumerate(token_indices):
            prefix = "B-" if order == 0 else "I-"
            tag = prefix + span["field"]
            if labels[token_index] not in {-100, label2id["O"]}:
                raise ValueError("Manual BIO annotations overlap after tokenization.")
            labels[token_index] = label2id[tag]
    encoded["labels"] = labels
    encoded["offset_mapping"] = offsets
    encoded["spans"] = spans
    return encoded
