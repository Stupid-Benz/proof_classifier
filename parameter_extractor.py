"""Inference-time parameter extraction using the trained BIO NER model only.

This module deliberately contains no regular-expression or rule-based
parameter extractor.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from parameter_schema import DERIVED_OR_DEFAULT_FIELDS, NER_SCHEMA_VERSION, PARAMETER_FIELDS, build_text


def highest_confidence_entity(entities: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the single highest-confidence NER entity, ignoring invalid scores."""
    scored_entities: list[dict[str, Any]] = []
    for entity in entities:
        try:
            score = float(entity.get("confidence"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            scored_entities.append(entity)
    if not scored_entities:
        return None
    return max(scored_entities, key=lambda item: float(item["confidence"]))


def highest_confidence_parameters(
    entities: list[dict[str, Any]], family: str
) -> dict[str, list[str]]:
    """Select the highest-confidence predicted value for each template field.

    Most template fields are single-valued, so only their strongest entity is
    passed to the template. ``vectors`` is intentionally multi-valued: the
    Gram-Schmidt template needs two input vectors. In that case, keep the two
    highest-confidence vector entities and restore their original text order
    so v1/v2 are not accidentally swapped by confidence ranking.
    """
    if family not in PARAMETER_FIELDS:
        raise ValueError(f"Unknown proof family: {family}")

    allowed_fields = PARAMETER_FIELDS[family]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entity in entities:
        field = entity.get("field")
        if field not in allowed_fields or not entity.get("text"):
            continue
        try:
            score = float(entity.get("confidence"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(score):
            grouped.setdefault(field, []).append(entity)

    selected_parameters: dict[str, list[str]] = {}
    for field in allowed_fields:
        candidates = grouped.get(field, [])
        if not candidates:
            continue

        ranked = sorted(
            candidates,
            key=lambda item: float(item["confidence"]),
            reverse=True,
        )
        if field == "vectors":
            # Gram-Schmidt requires two vectors. Retain the best two entities,
            # but preserve their source positions to retain vector order.
            selected = ranked[:2]
            selected.sort(key=lambda item: (int(item.get("start", 0)), int(item.get("end", 0))))
        else:
            selected = ranked[:1]

        selected_parameters[field] = [str(item["text"]) for item in selected]

    return selected_parameters

class ParameterExtractor:
    def __init__(self, model_dir: str | Path = "models/ner", device: str = "cpu") -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.tokenizer = None
        self.model = None
        self.metadata: dict[str, Any] = {}
        self.checkpoint_problem: str | None = None
        metadata_path = self.model_dir / "ner_metadata.json"
        if metadata_path.exists():
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_fields = {family: list(fields) for family, fields in PARAMETER_FIELDS.items()}
            stored_fields = self.metadata.get("parameter_fields")
            if stored_fields is not None and stored_fields != expected_fields:
                self.checkpoint_problem = (
                    "The saved NER checkpoint uses an older parameter schema. "
                    "The current Differential Geometry labels are `curve` and `evaluation_point`, "
                    "not `curvature` and `evaluation_point`. Retrain with `python train_ner.py`."
                )
            elif self.metadata.get("ner_schema_version", NER_SCHEMA_VERSION) != NER_SCHEMA_VERSION:
                self.checkpoint_problem = (
                    "The saved NER checkpoint was trained with a different NER schema. "
                    "Retrain with `python train_ner.py`."
                )

    @property
    def has_ner_model(self) -> bool:
        return (
            (self.model_dir / "config.json").exists()
            and bool(self.metadata)
            and self.checkpoint_problem is None
        )

    def _load_ner(self):
        if self.model is not None:
            return
        if not self.has_ner_model:
            detail = self.checkpoint_problem or f"No compatible NER model found at {self.model_dir}."
            raise FileNotFoundError(detail + " Run `python train_ner.py` first.")
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the NER dependencies with `pip install -r requirements.txt`.") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        self._torch = torch

    def predict_entities(self, question: str, proof: str, threshold: float = 0.50) -> list[dict[str, Any]]:
        self._load_ner()
        text = build_text(question, proof)
        encoded = self.tokenizer(
            text, return_offsets_mapping=True, truncation=True,
            max_length=int(self.metadata.get("max_length", 384)), return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        inputs = {k: v.to(self.device) for k, v in encoded.items()}
        with self._torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = self._torch.softmax(logits, dim=-1)
            confidence, pred_ids = probs.max(dim=-1)
        id2label = {int(k): v for k, v in self.metadata["id2label"].items()}
        tags = [id2label[int(i)] for i in pred_ids.cpu().tolist()]
        confs = confidence.cpu().tolist()

        entities: list[dict[str, Any]] = []
        active_label = None
        start = end = None
        scores: list[float] = []

        def flush():
            nonlocal active_label, start, end, scores
            if active_label is not None and start is not None and end is not None:
                value = text[start:end].strip(" \t\n,;:.")
                score = float(np.mean(scores)) if scores else 0.0
                if value and score >= threshold:
                    entities.append({
                        "field": active_label, "text": value,
                        "start": start, "end": end, "confidence": score,
                        "source": "MiniLM-NER",
                    })
            active_label, start, end, scores = None, None, None, []

        for tag, offset, score in zip(tags, offsets, confs):
            token_start, token_end = int(offset[0]), int(offset[1])
            if token_start == token_end:
                continue
            if tag == "O":
                flush()
                continue
            prefix, _, field = tag.partition("-")
            if prefix == "B" or active_label != field:
                flush()
                active_label, start, end = field, token_start, token_end
                scores = [score]
            else:
                end = token_end
                scores.append(score)
        flush()
        return entities

    def extract(self, question: str, proof: str, family: str, threshold: float = 0.50) -> dict[str, Any]:
        """Extract family-specific entities using the trained NER model only.

        No hand-written regex/rule fallback is applied to the user's input.
        If the model has not been trained, return an empty result and let the
        UI explain that NER training is required.
        """
        if family not in PARAMETER_FIELDS:
            raise ValueError(f"Unknown proof family: {family}")

        entities: list[dict[str, Any]] = []
        parameters: dict[str, list[str]] = {}
        family_supported = bool(PARAMETER_FIELDS[family])
        if self.has_ner_model and family_supported:
            allowed = set(PARAMETER_FIELDS[family])
            entities = [
                entity for entity in self.predict_entities(question, proof, threshold)
                if entity["field"] in allowed
            ]
            # Keep every candidate in ``entities`` for transparency, but send
            # only the strongest prediction for each template field onward.
            # Vector Operations is the exception: its template needs two
            # vectors, so the two strongest vector spans are retained.
            parameters = highest_confidence_parameters(entities, family)

        return {
            "proof_family": family,
            "parameters": parameters,
            "entities": entities,
            "ner_enabled": self.has_ner_model,
            "family_supported": family_supported,
            "supported_fields": list(PARAMETER_FIELDS[family]),
            "derived_fields_not_extracted": sorted(DERIVED_OR_DEFAULT_FIELDS),
            "extraction_method": "MiniLM BIO NER only",
        }
