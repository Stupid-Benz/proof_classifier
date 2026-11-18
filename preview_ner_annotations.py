"""Preview the weakly generated Sequence Convergence BIO spans without a model download."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from data_loader import DEFAULT_DATA_PATH, load_records
from json_utils import json_safe_default
from ner_alignment import align_parameters_to_spans, parameters_for_ner
from parameter_schema import build_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a reviewable preview of Sequence Convergence NER spans."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="JSON directory or one file")
    parser.add_argument(
        "--output", default="annotations/sequence_convergence_ner_preview.json",
        help="Output JSON path (kept outside data/ so the loader does not ingest it)",
    )
    args = parser.parse_args()

    records = [r for r in load_records(args.data) if r.proof_family == "Sequence Convergence"]
    preview = []
    counts: Counter[str] = Counter()
    missing_bound: list[str] = []

    for record in records:
        text = build_text(record.question, record.proof)
        targets = parameters_for_ner(record.parameters, record.proof_family, proof_text=record.proof)
        spans = align_parameters_to_spans(
            text, record.parameters, record.proof_family, proof_text=record.proof
        )
        counts.update(span["field"] for span in spans)
        if not any(span["field"] == "bound" for span in spans):
            missing_bound.append(record.record_id)
        preview.append({
            "record_id": record.record_id,
            "proof_style": record.parameters.get("proof_style"),
            "question": record.question,
            "proof": record.proof,
            "targets": targets,
            "entities": [
                {
                    "field": span["field"],
                    "text": span["text"],
                    "start": span["start"],
                    "end": span["end"],
                    "annotation_source": (
                        "dataset_parameter" if span["field"] in {"sequence", "limit"}
                        else "weak_rule_label"
                    ),
                }
                for span in spans
            ],
        })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(preview, indent=2, ensure_ascii=False, default=json_safe_default),
        encoding="utf-8",
    )
    print(f"Wrote {len(preview)} annotated examples to {output_path}")
    print("Aligned entity spans:", dict(counts))
    print(f"Examples without a BOUND span: {len(missing_bound)}")
    if missing_bound:
        print("These may be genuinely not applicable, or may need annotation review:")
        print(", ".join(missing_bound))


if __name__ == "__main__":
    main()
