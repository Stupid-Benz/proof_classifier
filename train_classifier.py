from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from json_utils import json_safe_default
from data_loader import DEFAULT_DATA_PATH, load_records
from embedding import MODEL_NAME, make_embeddings

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
SEED = 42


def make_logistic_regression() -> LogisticRegression:
    return LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=SEED,
    )


def fit_probe(X: np.ndarray, y: np.ndarray, cv_folds: int = 5):
    counts = Counter(y)
    if len(counts) == 1:
        return {"kind": "constant", "label": next(iter(counts))}, 0.0
    min_count = min(counts.values())
    if min_count < 2:
        raise ValueError(f"Not enough examples per class for a probe: {counts}")
    folds = min(cv_folds, min_count)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        make_logistic_regression(),
        {"C": C_GRID},
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    return search.best_estimator_, float(search.best_params_["C"])


def fit_hierarchical(X: np.ndarray, subjects: np.ndarray, families: np.ndarray, cv_folds: int = 5):
    subject_model, subject_c = fit_probe(X, subjects, cv_folds)
    family_models = {}
    family_cs = {}
    for subject in sorted(set(subjects)):
        indices = np.where(subjects == subject)[0]
        sub_families = families[indices]
        model, best_c = fit_probe(X[indices], sub_families, cv_folds)
        family_models[subject] = model
        family_cs[subject] = best_c
    return subject_model, family_models, {"subject_C": subject_c, "family_C": family_cs}


def predict_family_for_subject(model, vector: np.ndarray) -> str:
    if isinstance(model, dict) and model.get("kind") == "constant":
        return str(model["label"])
    return str(model.predict(vector.reshape(1, -1))[0])


def evaluate_nested_cv(X, subjects, families, folds=5, repeats=3):
    """Repeated outer stratified CV; C is tuned only inside each outer training fold."""
    subject_gold, subject_pred = [], []
    family_gold, family_pred_e2e = [], []
    family_pred_oracle, family_oracle_gold = [], []
    fold_details = []
    repeat_metrics = []

    # Stratify by the joint label so every proof family is represented in every fold.
    strata = np.array([f"{s}::{f}" for s, f in zip(subjects, families)], dtype=object)
    min_stratum_count = min(Counter(strata).values())
    if folds > min_stratum_count:
        raise ValueError(
            f"Cannot run {folds}-fold CV: the smallest subject-family group has "
            f"only {min_stratum_count} records. Reduce --cv-folds."
        )

    for repeat_no in range(repeats):
        subject_start = len(subject_gold)
        family_start = len(family_gold)
        oracle_start = len(family_oracle_gold)
        outer = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=SEED + repeat_no
        )

        for fold_no, (train_idx, test_idx) in enumerate(outer.split(X, strata), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            s_train, s_test = subjects[train_idx], subjects[test_idx]
            f_train, f_test = families[train_idx], families[test_idx]

            subject_model, family_models, best_cs = fit_hierarchical(
                X_train, s_train, f_train
            )
            predicted_subjects = subject_model.predict(X_test)

            e2e_family = []
            for i, predicted_subject in enumerate(predicted_subjects):
                e2e_family.append(
                    predict_family_for_subject(family_models[predicted_subject], X_test[i])
                )

            for subject in sorted(set(s_train)):
                selected = np.where(s_test == subject)[0]
                if len(selected) == 0:
                    continue
                model = family_models[subject]
                for local_i in selected:
                    oracle_prediction = predict_family_for_subject(model, X_test[local_i])
                    family_oracle_gold.append(f_test[local_i])
                    family_pred_oracle.append(oracle_prediction)

            subject_gold.extend(s_test.tolist())
            subject_pred.extend(predicted_subjects.tolist())
            family_gold.extend(f_test.tolist())
            family_pred_e2e.extend(e2e_family)
            fold_details.append({
                "repeat": repeat_no + 1,
                "fold": fold_no,
                "selected_C": best_cs,
            })
            print(f"  completed repeat {repeat_no + 1}/{repeats}, fold {fold_no}/{folds}")

        rs_gold = subject_gold[subject_start:]
        rs_pred = subject_pred[subject_start:]
        rf_gold = family_gold[family_start:]
        rf_pred = family_pred_e2e[family_start:]
        ro_gold = family_oracle_gold[oracle_start:]
        ro_pred = family_pred_oracle[oracle_start:]
        repeat_metrics.append({
            "repeat": repeat_no + 1,
            "subject_accuracy": float(accuracy_score(rs_gold, rs_pred)),
            "subject_macro_f1": float(f1_score(rs_gold, rs_pred, average="macro")),
            "family_end_to_end_accuracy": float(accuracy_score(rf_gold, rf_pred)),
            "family_end_to_end_macro_f1": float(f1_score(rf_gold, rf_pred, average="macro")),
            "family_oracle_accuracy": float(accuracy_score(ro_gold, ro_pred)),
            "family_oracle_macro_f1": float(f1_score(ro_gold, ro_pred, average="macro")),
            "joint_accuracy": float(np.mean([
                s_true == s_hat and f_true == f_hat
                for s_true, s_hat, f_true, f_hat in zip(rs_gold, rs_pred, rf_gold, rf_pred)
            ])),
        })

    all_subject_labels = sorted(set(subjects))
    all_family_labels = sorted(set(families))
    subject_report = classification_report(
        subject_gold, subject_pred, labels=all_subject_labels, output_dict=True, zero_division=0
    )
    family_report = classification_report(
        family_gold, family_pred_e2e, labels=all_family_labels, output_dict=True, zero_division=0
    )
    oracle_report = classification_report(
        family_oracle_gold, family_pred_oracle,
        labels=all_family_labels, output_dict=True, zero_division=0
    )
    joint_accuracy = float(np.mean([
        s_true == s_hat and f_true == f_hat
        for s_true, s_hat, f_true, f_hat in zip(subject_gold, subject_pred, family_gold, family_pred_e2e)
    ]))

    def mean_std(metric):
        values = [entry[metric] for entry in repeat_metrics]
        return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}

    return {
        "outer_folds": folds,
        "repeats": repeats,
        "subject_accuracy": float(accuracy_score(subject_gold, subject_pred)),
        "subject_macro_f1": float(f1_score(subject_gold, subject_pred, average="macro")),
        "subject_accuracy_across_repeats": mean_std("subject_accuracy"),
        "subject_macro_f1_across_repeats": mean_std("subject_macro_f1"),
        "subject_report": subject_report,
        "subject_confusion_matrix": confusion_matrix(subject_gold, subject_pred, labels=all_subject_labels).tolist(),
        "subject_labels": all_subject_labels,
        "family_end_to_end_accuracy": float(accuracy_score(family_gold, family_pred_e2e)),
        "family_end_to_end_macro_f1": float(f1_score(family_gold, family_pred_e2e, average="macro")),
        "family_end_to_end_accuracy_across_repeats": mean_std("family_end_to_end_accuracy"),
        "family_end_to_end_macro_f1_across_repeats": mean_std("family_end_to_end_macro_f1"),
        "family_end_to_end_report": family_report,
        "family_end_to_end_confusion_matrix": confusion_matrix(family_gold, family_pred_e2e, labels=all_family_labels).tolist(),
        "family_labels": all_family_labels,
        "family_oracle_accuracy": float(accuracy_score(family_oracle_gold, family_pred_oracle)),
        "family_oracle_macro_f1": float(f1_score(family_oracle_gold, family_pred_oracle, average="macro")),
        "family_oracle_accuracy_across_repeats": mean_std("family_oracle_accuracy"),
        "family_oracle_macro_f1_across_repeats": mean_std("family_oracle_macro_f1"),
        "family_oracle_report": oracle_report,
        "joint_subject_and_family_accuracy": joint_accuracy,
        "joint_accuracy_across_repeats": mean_std("joint_accuracy"),
        "repeat_metrics": repeat_metrics,
        "fold_details": fold_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the two-layer proof classifier (classifier training only).")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to a directory of JSON files or one JSON file")
    parser.add_argument("--model-dir", default="models", help="Directory to save the trained model")
    parser.add_argument("--skip-cv", action="store_true", help="Skip nested cross-validation")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3, help="Number of repeated outer CV runs")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    records = load_records(args.data)
    subjects = np.array([r.subject for r in records], dtype=object)
    families = np.array([r.proof_family for r in records], dtype=object)
    texts = [r.text for r in records]
    print(f"Loaded {len(records)} proofs from {args.data}")
    print("Subjects:", dict(Counter(subjects)))
    print("Proof families:", dict(Counter(families)))
    print("Encoding texts with frozen MiniLM on CPU...")
    X = make_embeddings(texts, batch_size=args.batch_size)
    print(f"Embedding matrix: {X.shape}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_cv:
        if args.cv_folds < 2:
            raise ValueError("--cv-folds must be at least 2")
        print(f"Running nested {args.cv_folds}-fold evaluation...")
        if args.cv_repeats < 1:
            raise ValueError("--cv-repeats must be at least 1")
        report = evaluate_nested_cv(X, subjects, families, folds=args.cv_folds, repeats=args.cv_repeats)
        report_path = model_dir / "evaluation.json"
        report_path.write_text(json.dumps(report, indent=2, default=json_safe_default), encoding="utf-8")
        print("\nCross-validation results")
        print(f"Subject: accuracy={report['subject_accuracy']:.3f}, macro-F1={report['subject_macro_f1']:.3f}")
        print(f"  repeated-CV subject accuracy: {report['subject_accuracy_across_repeats']['mean']:.3f} ± {report['subject_accuracy_across_repeats']['std']:.3f}")
        print(f"Family (oracle subject): accuracy={report['family_oracle_accuracy']:.3f}, macro-F1={report['family_oracle_macro_f1']:.3f}")
        print(f"Family (end-to-end): accuracy={report['family_end_to_end_accuracy']:.3f}, macro-F1={report['family_end_to_end_macro_f1']:.3f}")
        print(f"  repeated-CV family accuracy: {report['family_end_to_end_accuracy_across_repeats']['mean']:.3f} ± {report['family_end_to_end_accuracy_across_repeats']['std']:.3f}")
        print(f"Both layers correct: {report['joint_subject_and_family_accuracy']:.3f}")
        print(f"Detailed report saved to {report_path}")

    print("\nFitting final probes on all records...")
    subject_model, family_models, selected_c = fit_hierarchical(X, subjects, families, args.cv_folds)
    bundle = {
        "feature_type": "minilm",
        "model_name": MODEL_NAME,
        "input_fields": ["question", "proof"],
        "subject_model": subject_model,
        "family_models": family_models,
        "selected_C": selected_c,
        "subjects": sorted(set(subjects)),
        "proof_families": sorted(set(families)),
        "training_records": len(records),
    }
    bundle_path = model_dir / "proof_classifier.joblib"
    joblib.dump(bundle, bundle_path)
    print(f"Model saved to {bundle_path}")
    print("Done. Launch the test UI with: streamlit run app.py")


if __name__ == "__main__":
    main()
