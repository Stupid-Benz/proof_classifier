"""Validate and summarize all proof JSON files in the data directory."""
from __future__ import annotations

import argparse
from collections import Counter

from data_loader import DEFAULT_DATA_PATH, discover_json_files, load_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate proof records from one or more JSON files.")
    parser.add_argument(
        "--data", default=str(DEFAULT_DATA_PATH),
        help="Directory containing JSON files (default: data/) or a single JSON file",
    )
    args = parser.parse_args()

    files = discover_json_files(args.data)
    records = load_records(args.data)
    subject_counts = Counter(record.subject for record in records)
    family_counts = Counter(record.proof_family for record in records)
    file_counts = Counter(record.source_file for record in records)

    print(f"Dataset valid: {len(records)} proof records from {len(files)} JSON file(s).")
    print("Files:")
    for filename in sorted(file_counts):
        print(f"  {filename}: {file_counts[filename]} records")
    print("Subjects:")
    for label, count in sorted(subject_counts.items()):
        print(f"  {label}: {count}")
    print("Proof families:")
    for label, count in sorted(family_counts.items()):
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
