"""Load, combine, and validate proof records stored across JSON files.

This is the project's single data-ingestion module. Training scripts should
import ``load_records`` instead of reading JSON files themselves.

Accepted input:
- a directory: load every ``*.json`` file under it (recursively, sorted);
- one JSON file: load only that file.

Each JSON file may contain a list of proof records, a single proof record, or
an object with a ``proofs`` / ``records`` list.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from parameter_schema import SUBJECT_FAMILIES, build_text

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class ProofRecord:
    record_id: str
    subject: str
    proof_family: str
    theorem: str
    question: str
    proof: str
    parameters: dict[str, Any]
    animation_notes: str
    source_file: str = ""
    manual_entities: list[dict[str, Any]] = field(default_factory=list)
    manual_annotation_complete: bool = False

    @property
    def text(self) -> str:
        return build_text(self.question, self.proof)


def discover_json_files(path: str | Path = DEFAULT_DATA_PATH) -> list[Path]:
    """Resolve a directory or a single JSON path to a deterministic file list."""
    input_path = Path(path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".json":
            raise ValueError(f"Dataset file must have a .json extension: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise ValueError(f"Dataset path is neither a file nor a directory: {input_path}")

    files = sorted(
        file_path for file_path in input_path.rglob("*.json")
        if file_path.is_file() and not any(part.startswith(".") for part in file_path.relative_to(input_path).parts)
    )
    if not files:
        raise FileNotFoundError(
            f"No JSON files found in {input_path}. Add one or more .json files to the data folder."
        )
    return files


def _records_from_json_payload(payload: Any, source: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict) and isinstance(payload.get("proofs"), list):
        rows = payload["proofs"]
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        rows = payload["records"]
    elif isinstance(payload, dict) and "id" in payload:
        rows = [payload]
    else:
        raise ValueError(
            f"{source.name}: expected a JSON list of proof records, a single record, "
            "or an object containing a 'proofs'/'records' list."
        )

    if not rows:
        raise ValueError(f"{source.name}: JSON file contains no proof records.")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{source.name}: every proof record must be a JSON object.")
    return rows


def load_records(path: str | Path = DEFAULT_DATA_PATH) -> list[ProofRecord]:
    """Load all proof records from one JSON file or all JSONs in a directory.

    IDs must be globally unique across all files, not merely unique within each
    individual JSON file. This prevents silent duplicates after datasets merge.
    """
    json_files = discover_json_files(path)
    records: list[ProofRecord] = []
    errors: list[str] = []
    seen_ids: set[str] = set()

    for json_path in json_files:
        try:
            with json_path.open("r", encoding="utf-8-sig") as file:
                payload = json.load(file)
            rows = _records_from_json_payload(payload, json_path)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            errors.append(f"{json_path.name}: {exc}")
            continue

        for index, row in enumerate(rows, start=1):
            label = f"{json_path.name}, record {index}"
            record_id = str(row.get("id", "")).strip()
            subject = str(row.get("subject", "")).strip()
            family = str(row.get("proof_family", "")).strip()
            theorem = row.get("theorem", "")
            question = row.get("question", "")
            proof = row.get("proof", "")
            parameters = row.get("parameters", {})
            animation_notes = row.get("animation_notes", "")
            manual_entities = row.get("manual_entities", [])
            manual_annotation_complete = bool(row.get("manual_annotation_complete", False))

            if not record_id:
                errors.append(f"{label}: missing non-empty 'id'")
            elif record_id in seen_ids:
                errors.append(
                    f"{label} ({record_id}): duplicate id across the loaded dataset"
                )
            else:
                seen_ids.add(record_id)

            if subject not in SUBJECT_FAMILIES:
                errors.append(
                    f"{label} ({record_id or 'no id'}): unknown subject {subject!r}; "
                    f"valid subjects: {', '.join(SUBJECT_FAMILIES)}"
                )
            elif family not in SUBJECT_FAMILIES[subject]:
                errors.append(
                    f"{label} ({record_id or 'no id'}): family {family!r} does not belong "
                    f"to subject {subject!r}"
                )

            if not isinstance(question, str) or not isinstance(proof, str):
                errors.append(f"{label} ({record_id or 'no id'}): question and proof must be strings")
                continue
            if not question.strip() and not proof.strip():
                errors.append(f"{label} ({record_id or 'no id'}): question and proof cannot both be empty")
            if not isinstance(parameters, dict):
                errors.append(f"{label} ({record_id or 'no id'}): 'parameters' must be a JSON object")
                continue
            if not isinstance(manual_entities, list):
                errors.append(f"{label} ({record_id or 'no id'}): 'manual_entities' must be a JSON list")
                continue
            if not isinstance(theorem, str) or not isinstance(animation_notes, str):
                errors.append(f"{label} ({record_id or 'no id'}): theorem and animation_notes must be strings")
                continue

            records.append(ProofRecord(
                record_id=record_id,
                subject=subject,
                proof_family=family,
                theorem=theorem,
                question=question,
                proof=proof,
                parameters=parameters,
                animation_notes=animation_notes,
                source_file=json_path.name,
                manual_entities=manual_entities,
                manual_annotation_complete=manual_annotation_complete,
            ))

    if errors:
        excerpt = "\n".join(f"- {error}" for error in errors[:20])
        suffix = f"\n... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ValueError(f"Dataset validation found {len(errors)} problem(s):\n{excerpt}{suffix}")

    if not records:
        raise ValueError("No proof records were loaded.")
    return records
