#!/usr/bin/env python
import argparse
import csv
from pathlib import Path
from typing import List
import random


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare CIF hardness CSV with train/valid/test splits."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    return parser.parse_args()


def read_rows(csv_path: Path) -> List[dict]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in CSV file: {csv_path}")
        required = {"id", "cif_path", "hardness"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"Missing required columns in {csv_path}: {', '.join(sorted(missing))}"
            )
        rows = [row for row in reader if any(row.values())]
    return rows


def assign_splits(rows: List[dict], seed: int, ratios: List[float]) -> List[dict]:
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    train_size = int(len(rows) * ratios[0])
    valid_size = int(len(rows) * ratios[1])
    train_idx = set(indices[:train_size])
    valid_idx = set(indices[train_size : train_size + valid_size])
    for idx, row in enumerate(rows):
        if idx in train_idx:
            row["split"] = "train"
        elif idx in valid_idx:
            row["split"] = "valid"
        else:
            row["split"] = "test"
    return rows


def write_rows(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "cif_path", "hardness", "split"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "cif_path": row["cif_path"],
                    "hardness": row["hardness"],
                    "split": row["split"],
                }
            )


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv)
    rows = assign_splits(
        rows,
        seed=args.seed,
        ratios=[args.train_ratio, args.valid_ratio, args.test_ratio],
    )
    write_rows(rows, args.output_csv)


if __name__ == "__main__":
    main()
