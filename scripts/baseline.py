"""Run a Qwen baseline with a paper's evaluation tasks."""

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    recipes = tomllib.loads((ROOT / "recipes/benchmarks.toml").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", choices=recipes)
    parser.add_argument("--size", choices=["3b", "7b"], default="7b")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    tasks = recipes[args.paper].get(f"tasks_{args.size}")
    if not tasks:
        parser.error(f"No published {args.size} suite for {args.paper}")
    if not (args.checkpoint / "config.json").is_file():
        parser.error("--checkpoint must point to a downloaded model snapshot")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch size and limit must be positive")
    model_args = f"model={args.checkpoint.resolve()}"
    if args.paper == "lvr":
        model_args += f",min_pixels={128 * 28**2},max_pixels={5120 * 28**2}"
    command = [
        sys.executable,
        "-m",
        "lmms_eval",
        "--model",
        "sglang",
        "--model_args",
        model_args,
        "--include_path",
        str(ROOT / "tasks"),
        "--tasks",
        ",".join(tasks),
        "--batch_size",
        str(args.batch_size),
        "--log_samples",
        "--output_path",
        str(args.output.resolve()),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
