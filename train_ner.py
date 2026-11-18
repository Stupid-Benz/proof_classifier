from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from data_loader import DEFAULT_DATA_PATH, discover_json_files, load_records
from feedback_store import FEEDBACK_DIR
from json_utils import json_safe_default
from parameter_schema import DERIVED_OR_DEFAULT_FIELDS, NER_FAMILIES, NER_SCHEMA_VERSION, PARAMETER_FIELDS, build_text
from ner_alignment import align_parameters_to_spans, parameters_for_ner, bio_labels_for_text
from manual_ner import validate_manual_entities, encode_manual_spans

DEFAULT_NER_BACKBONE = "sentence-transformers/all-MiniLM-L6-v2"
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass




def split_examples_with_feedback(examples, test_size: float, seed: int):
    """Split base examples for evaluation; append reviewed feedback to training only."""
    base_indices = np.array(
        [index for index, example in enumerate(examples) if not example.get("is_feedback", False)],
        dtype=int,
    )
    feedback_indices = np.array(
        [index for index, example in enumerate(examples) if example.get("is_feedback", False)],
        dtype=int,
    )
    if len(base_indices) < 2:
        raise ValueError("At least two base (non-feedback) aligned examples are required for evaluation.")

    families = np.array([examples[int(index)]["family"] for index in base_indices])
    family_counts = Counter(families)
    too_small = {family: count for family, count in family_counts.items() if count < 2}
    if too_small:
        raise ValueError(
            "NER needs at least 2 base examples per proof family for a stratified train/test split. "
            f"Add more base data to: {too_small}"
        )

    requested_test_records = math.ceil(len(base_indices) * test_size)
    if requested_test_records < len(family_counts):
        raise ValueError(
            f"The selected test size gives only {requested_test_records} test records, but "
            f"there are {len(family_counts)} proof families. Increase --test-size or add data."
        )

    base_train_indices, test_indices = train_test_split(
        base_indices, test_size=test_size, random_state=seed, stratify=families
    )
    train_indices = np.concatenate([base_train_indices, feedback_indices])
    return train_indices, test_indices, base_train_indices, feedback_indices


def span_set_from_tags(tags: list[str], offsets: list[list[int]]) -> set[tuple[int, int, str]]:
    spans: set[tuple[int, int, str]] = set()
    active = None
    start = end = None
    for tag, offset in zip(tags, offsets):
        a, b = int(offset[0]), int(offset[1])
        if a == b:
            continue
        if tag == "O":
            if active is not None:
                spans.add((start, end, active))
            active = start = end = None
            continue
        prefix, _, field = tag.partition("-")
        if prefix == "B" or active != field:
            if active is not None:
                spans.add((start, end, active))
            active, start, end = field, a, b
        else:
            end = b
    if active is not None:
        spans.add((start, end, active))
    return spans


def score_span_sets(gold: list[set], predicted: list[set]) -> dict:
    tp = sum(len(g & p) for g, p in zip(gold, predicted))
    fp = sum(len(p - g) for g, p in zip(gold, predicted))
    fn = sum(len(g - p) for g, p in zip(gold, predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    labels = sorted({x[2] for s in gold + predicted for x in s})
    per_field = {}
    for label in labels:
        ltp = sum(sum(1 for x in (g & p) if x[2] == label) for g, p in zip(gold, predicted))
        lfp = sum(sum(1 for x in (p - g) if x[2] == label) for g, p in zip(gold, predicted))
        lfn = sum(sum(1 for x in (g - p) if x[2] == label) for g, p in zip(gold, predicted))
        pr = ltp / (ltp + lfp) if ltp + lfp else 0.0
        re = ltp / (ltp + lfn) if ltp + lfn else 0.0
        per_field[label] = {
            "precision": pr, "recall": re,
            "f1": 2 * pr * re / (pr + re) if pr + re else 0.0,
            "support": ltp + lfn,
        }
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "per_field": per_field}


def main():
    parser = argparse.ArgumentParser(description="Train a BIO token-classification model for proof parameters.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to a directory of JSON files or one JSON file")
    parser.add_argument(
        "--feedback-data", default=str(FEEDBACK_DIR),
        help="Directory or JSON file containing reviewed app feedback (used for training only)",
    )
    parser.add_argument("--model-dir", default="models/ner")
    parser.add_argument("--backbone", default=DEFAULT_NER_BACKBONE)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or not 0.05 <= args.test_size <= 0.5:
        raise ValueError("Use epochs/batch-size >= 1 and test-size between 0.05 and 0.5.")

    try:
        import torch
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "NER dependencies are missing. Install them with: pip install -r requirements.txt"
        ) from exc

    set_seed(args.seed)
    base_records = load_records(args.data)
    feedback_path = Path(args.feedback_data).expanduser()
    feedback_records = []
    if feedback_path.exists():
        try:
            feedback_files = discover_json_files(feedback_path)
        except FileNotFoundError:
            feedback_files = []
        if feedback_files:
            feedback_records = load_records(feedback_path)

    base_ids = {record.record_id for record in base_records}
    duplicate_feedback_ids = base_ids.intersection(record.record_id for record in feedback_records)
    if duplicate_feedback_ids:
        raise ValueError(
            "Feedback record IDs must be unique and must not duplicate base dataset IDs: "
            + ", ".join(sorted(duplicate_feedback_ids)[:10])
        )
    feedback_ids = {record.record_id for record in feedback_records}
    all_records = base_records + feedback_records
    records = [record for record in all_records if record.proof_family in NER_FAMILIES]
    excluded_family_records = len(all_records) - len(records)
    if not records:
        raise ValueError(
            "No records belong to the five proof families currently supported by NER: "
            + ", ".join(NER_FAMILIES)
        )

    missing = [r.record_id for r in records if not r.parameters and not r.manual_entities]
    if missing:
        raise ValueError(
            f"{len(missing)} NER records have no `parameters` dictionary (first: {missing[:5]}). "
            "Add a `parameters` JSON object to each record in the relevant data/*.json file."
        )

    label_fields = sorted({field for family in NER_FAMILIES for field in PARAMETER_FIELDS[family]})
    tag_names = ["O"] + [tag for field in label_fields for tag in (f"B-{field}", f"I-{field}")]
    label2id = {tag: i for i, tag in enumerate(tag_names)}
    id2label = {i: tag for tag, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required because BIO tags are aligned using token offsets.")

    examples = []
    alignment_counts = Counter()
    target_counts = Counter()
    aligned_record_counts = Counter()
    unaligned_targets = []
    skipped_unlabelled_records = []
    feedback_skipped_unaligned_records = []
    manual_annotation_records = 0
    manual_span_counts = Counter()

    for record in records:
        text = build_text(record.question, record.proof)
        if record.manual_entities:
            # Exact character offsets entered by the user take precedence over
            # weak alignment. These examples are feedback and train only.
            spans = validate_manual_entities(text, record.proof_family, record.manual_entities)
            manual_annotation_records += 1
            for span in spans:
                alignment_counts[span["field"]] += 1
                manual_span_counts[span["field"]] += 1
                aligned_record_counts[span["field"]] += 1
                target_counts[span["field"]] += 1
            examples.append({
                "text": text,
                "family": record.proof_family,
                "parameters": {},
                "manual_spans": spans,
                "manual_annotation_complete": record.manual_annotation_complete,
                "record_id": record.record_id,
                "is_feedback": True,
                "is_manual_annotation": True,
            })
            continue

        targets = parameters_for_ner(record.parameters, record.proof_family)
        spans = align_parameters_to_spans(text, targets, record.proof_family)
        aligned_fields = {span["field"] for span in spans}

        for field in targets:
            target_counts[field] += 1
            if field not in aligned_fields:
                unaligned_targets.append({
                    "record_id": record.record_id,
                    "proof_family": record.proof_family,
                    "field": field,
                    "value": targets[field],
                })
            else:
                aligned_record_counts[field] += 1

        for span in spans:
            alignment_counts[span["field"]] += 1

        if not spans:
            skipped_unlabelled_records.append(record.record_id)
            if record.record_id in feedback_ids:
                feedback_skipped_unaligned_records.append(record.record_id)
            continue
        # Reviewed feedback is admitted only when every supplied field can be
        # aligned. Otherwise the unmatched correct expression would be trained
        # as an O-tag, teaching the model the wrong thing.
        if record.record_id in feedback_ids and set(targets) - aligned_fields:
            feedback_skipped_unaligned_records.append(record.record_id)
            skipped_unlabelled_records.append(record.record_id)
            continue
        examples.append({
            "text": text,
            "family": record.proof_family,
            "parameters": targets,
            "manual_spans": None,
            "record_id": record.record_id,
            "is_feedback": record.record_id in feedback_ids,
            "is_manual_annotation": False,
        })

    feedback_examples = sum(1 for example in examples if example["is_feedback"])
    print(f"Loaded {len(base_records)} base records and {len(feedback_records)} reviewed feedback records.")
    print(f"Kept {len(records)} records from the five NER-supported families.")
    print(f"Feedback records admitted as training examples: {feedback_examples}.")
    print(f"Direct manual-span annotation records: {manual_annotation_records}.")
    if manual_span_counts:
        print("Manual BIO spans by field:", dict(manual_span_counts))
    if feedback_skipped_unaligned_records:
        print(
            "WARNING: feedback records skipped because one or more reviewed fields "
            "could not be aligned: " + ", ".join(feedback_skipped_unaligned_records)
        )
    if excluded_family_records:
        print(f"Excluded {excluded_family_records} records from NER (families outside current NER scope).")
    print(f"Kept {len(examples)} records with at least one aligned target span.")
    print(f"Aligned {sum(alignment_counts.values())} explicit parameter spans.")
    print("Aligned spans by field:", dict(alignment_counts))
    print("Records with an aligned span by field:", dict(aligned_record_counts))
    if unaligned_targets:
        print(f"WARNING: {len(unaligned_targets)} target field(s) did not match literal text in question/proof.")
        for item in unaligned_targets[:20]:
            print(f"  {item['record_id']} | {item['field']} = {item['value']!r}")
        if len(unaligned_targets) > 20:
            print(f"  ... and {len(unaligned_targets) - 20} more; see ner_evaluation.json")
    if skipped_unlabelled_records:
        print("Skipped records without any aligned NER spans:", skipped_unlabelled_records)
    print("Note: weak labels come only from literal parameter values; derived/template fields are excluded.")

    if not examples:
        raise ValueError("No parameter values could be aligned to the question/proof text. Check parameters and wording.")

    # Feedback examples are deliberately appended to training only. The test
    # partition remains a split of the original/base records.
    train_indices, test_indices, base_train_indices, feedback_indices = split_examples_with_feedback(
        examples, test_size=args.test_size, seed=args.seed
    )

    # Include the model backbone's existing BIO vocabulary. The output head is
    # randomly initialized and then learned from the automatically aligned data.
    model = AutoModelForTokenClassification.from_pretrained(
        args.backbone,
        num_labels=len(tag_names),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    model.to("cpu")

    class ProofNerDataset(Dataset):
        def __init__(self, selected_indices):
            self.items = []
            for idx in selected_indices:
                example = examples[int(idx)]
                if example.get("manual_spans") is not None:
                    encoded = encode_manual_spans(
                        example["text"], example["manual_spans"], tokenizer, label2id,
                        max_length=args.max_length,
                        complete=example.get("manual_annotation_complete", False),
                    )
                else:
                    encoded = bio_labels_for_text(
                        example["text"], example["parameters"], example["family"],
                        tokenizer, label2id, max_length=args.max_length,
                    )
                self.items.append({
                    "input_ids": encoded["input_ids"],
                    "attention_mask": encoded["attention_mask"],
                    "labels": encoded["labels"],
                })

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    def collate(batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        pad_id = tokenizer.pad_token_id or 0
        input_ids, attention_masks, labels = [], [], []
        for item in batch:
            pad = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [pad_id] * pad)
            attention_masks.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    train_ds = ProofNerDataset(train_indices)
    test_ds = ProofNerDataset(test_indices)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Counter the severe O-tag imbalance without overwhelming the small dataset.
    counts = Counter()
    for item in train_ds.items:
        counts.update(label for label in item["labels"] if label != -100)
    total = sum(counts.values())
    class_weights = torch.ones(len(tag_names), dtype=torch.float32)
    for label_id, count in counts.items():
        if label_id == label2id["O"]:
            class_weights[label_id] = 1.0
        elif count:
            class_weights[label_id] = min(5.0, max(1.0, (total / (len(tag_names) * count)) ** 0.5))
    class_weights = class_weights.to("cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    print(
        f"Training on CPU: {len(train_ds)} train records "
        f"({len(base_train_indices)} base + {len(feedback_indices)} feedback), "
        f"{len(test_ds)} held-out base test records, {args.epochs} epochs."
    )
    model.train()
    for epoch in range(args.epochs):
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, len(tag_names)), batch["labels"].view(-1),
                weight=class_weights, ignore_index=-100,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {np.mean(losses):.4f}")

    # Entity-level exact-span evaluation against the automatically aligned
    # pseudo-labels. This is not a substitute for manually reviewed gold labels.
    model.eval()
    gold_sets, ner_sets = [], []
    gold_tag_sequences, ner_tag_sequences = [], []

    with torch.no_grad():
        for local_idx, original_idx in enumerate(test_indices):
            item = test_ds.items[local_idx]
            input_ids = torch.tensor([item["input_ids"]], dtype=torch.long)
            attention_mask = torch.tensor([item["attention_mask"]], dtype=torch.long)
            pred_ids = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(-1)[0].tolist()
            encoded = bio_labels_for_text(
                examples[int(original_idx)]["text"], examples[int(original_idx)]["parameters"],
                examples[int(original_idx)]["family"], tokenizer, label2id, max_length=args.max_length,
            )
            offsets = encoded["offset_mapping"]
            valid = [(pred_ids[i], encoded["labels"][i], offsets[i]) for i in range(min(len(pred_ids), len(offsets))) if offsets[i][0] != offsets[i][1]]
            pred_tags = [id2label[int(x[0])] for x in valid]
            gold_tags = [id2label[int(x[1])] for x in valid]
            valid_offsets = [x[2] for x in valid]
            gold_sets.append(span_set_from_tags(gold_tags, valid_offsets))
            ner_sets.append(span_set_from_tags(pred_tags, valid_offsets))
            gold_tag_sequences.append(gold_tags)
            ner_tag_sequences.append(pred_tags)

    ner_metrics = score_span_sets(gold_sets, ner_sets)
    try:
        from seqeval.metrics import classification_report as seqeval_classification_report
        seqeval_ner_report = seqeval_classification_report(
            gold_tag_sequences, ner_tag_sequences, output_dict=True, zero_division=0
        )
    except ImportError:
        seqeval_ner_report = None

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    metadata = {
        "ner_schema_version": NER_SCHEMA_VERSION,
        "backbone": args.backbone,
        "max_length": args.max_length,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "parameter_fields": PARAMETER_FIELDS,
        "ner_families": list(NER_FAMILIES),
        "derived_fields_not_extracted": sorted(DERIVED_OR_DEFAULT_FIELDS),
        "training_records": len(train_indices),
        "base_training_records": len(base_train_indices),
        "feedback_records_loaded": len(feedback_records),
        "manual_annotation_records_loaded": manual_annotation_records,
        "manual_annotation_span_counts": dict(manual_span_counts),
        "feedback_training_records_aligned": len(feedback_indices),
        "feedback_skipped_unaligned_records": feedback_skipped_unaligned_records,
        "manual_annotation_records": manual_annotation_records,
        "manual_annotation_span_counts": dict(manual_span_counts),
        "test_records": len(test_indices),
        "excluded_family_record_count": excluded_family_records,
        "skipped_unlabelled_records": skipped_unlabelled_records,
        "aligned_span_count": int(sum(alignment_counts.values())),
        "alignment_counts": dict(alignment_counts),
        "aligned_record_counts": dict(aligned_record_counts),
        "target_key_counts": dict(target_counts),
        "unaligned_targets": unaligned_targets,
        "ner_threshold_default": args.threshold,
        "evaluation_note": "Metrics are against automatically aligned dataset parameter spans (weak/pseudo labels), not manually annotated gold NER labels.",
    }
    (model_dir / "ner_metadata.json").write_text(json.dumps(metadata, indent=2, default=json_safe_default), encoding="utf-8")

    report = {
        "split": {
            "train_records": len(train_indices),
            "base_train_records": len(base_train_indices),
            "feedback_train_records": len(feedback_indices),
            "test_records": len(test_indices),
            "seed": args.seed,
        },
        "ner_entity_metrics": ner_metrics,
        "seqeval_ner_report": seqeval_ner_report,
        "alignment_counts": dict(alignment_counts),
        "aligned_record_counts": dict(aligned_record_counts),
        "target_key_counts": dict(target_counts),
        "unaligned_targets": unaligned_targets,
        "skipped_unlabelled_records": skipped_unlabelled_records,
        "feedback_skipped_unaligned_records": feedback_skipped_unaligned_records,
        "manual_annotation_records": manual_annotation_records,
        "manual_annotation_span_counts": dict(manual_span_counts),
        "excluded_family_record_count": excluded_family_records,
        "evaluation_note": metadata["evaluation_note"],
    }
    report_path = model_dir / "ner_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, default=json_safe_default), encoding="utf-8")
    print("\nEntity-level results (exact spans):")
    print(f"NER: P={ner_metrics['precision']:.3f} R={ner_metrics['recall']:.3f} F1={ner_metrics['f1']:.3f}")
    print(f"Saved NER model to {model_dir}")
    print(f"Saved evaluation to {report_path}")


if __name__ == "__main__":
    main()
