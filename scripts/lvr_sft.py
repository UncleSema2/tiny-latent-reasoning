"""Train LVR-SFT, then evaluate its checkpoint on the pinned LVR suite."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", choices=("3b", "7b"), default="3b")
    parser.add_argument("--gpus", type=int, default=1, help="Training GPUs on one node")
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-limit", type=int)
    args, training_options = parser.parse_known_args()
    if (
        args.gpus < 1
        or args.eval_batch_size < 1
        or (args.eval_limit is not None and args.eval_limit < 1)
    ):
        parser.error("GPU count, evaluation batch size and limit must be positive")
    checkpoint = args.output.resolve() / "model"
    train = [sys.executable, "-m"]
    if args.gpus > 1:
        train.extend(
            [
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node",
                str(args.gpus),
                "--module",
            ]
        )
        training_options.append("--fsdp")
    if args.answer_only:
        training_options.append("--answer-only")
    subprocess.run(
        [
            *train,
            "methods.lvr.training",
            "--data",
            str(args.data.resolve()),
            "--size",
            args.size,
            "--output",
            str(checkpoint),
            *training_options,
        ],
        cwd=ROOT,
        check=True,
    )
    base = [
        sys.executable,
        str(ROOT / "scripts/baseline.py"),
        "lvr",
        "--checkpoint",
        str(checkpoint),
        "--size",
        args.size,
        "--batch-size",
        str(args.eval_batch_size),
    ]
    if args.eval_limit is not None:
        base.extend(["--limit", str(args.eval_limit)])
    for budget in [None] if args.answer_only else [4, 8, 16]:
        options = [] if budget is None else ["--lvr-steps", str(budget)]
        output = checkpoint.parent / (
            "evaluation" if budget is None else f"evaluation-{budget}"
        )
        subprocess.run([*base, *options, "--output", str(output)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
