"""Summarize paired LUT trajectory scores from one audit run."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--count", type=int, default=512)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob("rank-*.jsonl")):
        with path.open() as shard:
            rows.extend(json.loads(line) for line in shard)
    rows.sort(key=lambda row: row["id"])
    if len(rows) != args.count or len({r["id"] for r in rows}) != len(rows) or not rows:
        parser.error("Expected a complete audit with one row per requested example")
    pairs = (
        ("native", "random"),
        ("shifted", "shifted_control"),
        ("trained_bounds", "trained_bounds_control"),
        ("trained_bounds_shifted", "trained_bounds_shifted_control"),
        ("native", "wrong_image"),
        ("trained_bounds", "trained_bounds_wrong_image"),
    )
    samples = np.random.default_rng(42).integers(len(rows), size=(10_000, len(rows)))
    result = {}
    for condition, control in pairs:
        if not any(control in row for row in rows):
            continue  # Earlier audit runs include a subset of the controls.
        delta = np.array([row[condition] - row[control] for row in rows])
        result[f"{condition} minus {control}"] = {
            "questions": len(rows),
            "mean_delta": delta.mean().item(),
            "bootstrap_ci95": np.quantile(
                delta[samples].mean(1), [0.025, 0.975]
            ).tolist(),
            "positive": int((delta > 0).sum()),
            "retained": int((delta > -0.5).sum()),
        }
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
