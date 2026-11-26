"""Build the LUT student curriculum from frozen-judge answer token scores."""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean


def answer_score(logprobs):
    if not isinstance(logprobs, list) or not logprobs:
        raise ValueError("expected a nonempty list of answer log probabilities")
    if any(
        type(p) not in (int, float) or not math.isfinite(p) or p > 0 for p in logprobs
    ):
        raise ValueError("expected finite, nonpositive answer log probabilities")
    return fmean(logprobs)


def build_curriculum(rows, *, seed, threshold=-0.5):
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    subsets = defaultdict(list)
    seen = set()
    for line, row in enumerate(rows, 1):
        try:
            sample_id, subset = row["id"], row["subset"]
            if not all(
                isinstance(key, str) and key.strip() for key in (sample_id, subset)
            ):
                raise ValueError("id and subset must be nonempty strings")
            if sample_id in seen:
                raise ValueError(f"duplicate sample id {sample_id!r}")
            seen.add(sample_id)
            teacher, control = row["teacher_logprobs"], row["control_logprobs"]
            delta = answer_score(teacher) - answer_score(control)
            if len(teacher) != len(control):
                raise ValueError(
                    "teacher and control must score the same answer tokens"
                )
            if delta > threshold:
                subsets[subset].append((delta, sample_id))
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"score row {line}: {error}") from error

    low, high = [], []
    for subset in sorted(subsets):
        ranked = sorted(subsets[subset])
        middle = len(ranked) // 2
        low.extend(sample_id for _, sample_id in ranked[:middle])
        high.extend(sample_id for _, sample_id in ranked[middle:])

    rng = random.Random(seed)
    rng.shuffle(low)
    rng.shuffle(high)
    return {
        "seed": seed,
        "threshold": threshold,
        "input_count": len(seen),
        "filtered_count": len(seen) - len(low) - len(high),
        "low": low,
        "high": high,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, help="one scored trajectory per JSON line")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=-0.5)
    args = parser.parse_args()
    with args.scores.open(encoding="utf-8") as scores:
        curriculum = build_curriculum(
            map(json.loads, scores), seed=args.seed, threshold=args.threshold
        )
    json.dump(curriculum, sys.stdout)
    print()


if __name__ == "__main__":
    main()
