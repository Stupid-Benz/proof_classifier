"""Run classifier training and NER training sequentially.

The detailed training implementations live in train_classifier.py and
train_ner.py. This file is intentionally only an orchestration entry point.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from data_loader import DEFAULT_DATA_PATH, load_records

PROJECT_ROOT = Path(__file__).resolve().parent


def run_stage(script_name: str, arguments: list[str]) -> None:
    command = [sys.executable, str(PROJECT_ROOT / script_name), *arguments]
    print("\n" + "=" * 72, flush=True)
    print(f"Starting {script_name}", flush=True)
    print("Command:", " ".join(command), flush=True)
    print("=" * 72, flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train both the hierarchical proof classifier and parameter NER model."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Directory containing JSON proof files, or one JSON file")
    parser.add_argument("--skip-cv", action="store_true", help="Skip classifier nested cross-validation")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--classifier-batch-size", type=int, default=32)
    parser.add_argument("--ner-epochs", type=int, default=8)
    parser.add_argument("--ner-batch-size", type=int, default=8)
    parser.add_argument("--ner-learning-rate", type=float, default=3e-5)
    parser.add_argument("--ner-max-length", type=int, default=384)
    parser.add_argument("--ner-test-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_path = Path(args.data).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()
    if not dataset_path.exists():
        parser.error(f"Dataset path not found: {dataset_path}")
    try:
        loaded_records = load_records(dataset_path)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Validated {len(loaded_records)} proof records from JSON data.", flush=True)

    common_data = ["--data", str(dataset_path)]
    classifier_args = [*common_data]
    if args.skip_cv:
        classifier_args.append("--skip-cv")
    classifier_args += [
        "--cv-folds", str(args.cv_folds),
        "--cv-repeats", str(args.cv_repeats),
        "--batch-size", str(args.classifier_batch_size),
    ]
    ner_args = [
        *common_data,
        "--epochs", str(args.ner_epochs),
        "--batch-size", str(args.ner_batch_size),
        "--learning-rate", str(args.ner_learning_rate),
        "--max-length", str(args.ner_max_length),
        "--test-size", str(args.ner_test_size),
        "--seed", str(args.seed),
    ]

    try:
        run_stage("train_classifier.py", classifier_args)
        run_stage("train_ner.py", ner_args)
    except subprocess.CalledProcessError as exc:
        print(
            f"\nTraining stopped because {Path(exc.cmd[1]).name} exited with code {exc.returncode}.",
            file=sys.stderr,
        )
        return int(exc.returncode or 1)

    print("\nBoth training stages completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
