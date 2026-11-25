"""Run a Qwen baseline with a paper's evaluation tasks."""

import argparse
import json
import subprocess
import sys
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def write_task_configs(directory, tasks):
    datasets = tomllib.loads((ROOT / "recipes/datasets.toml").read_text())
    sources = {
        task: (repo, entry["revision"])
        for repo, entry in datasets.items()
        for task in entry["tasks"]
    }
    for task in tasks:
        repo, revision = sources[task]
        config = {
            "task": task,
            "dataset_path": repo,
            "dataset_kwargs": {"revision": revision},
        }
        (directory / f"{task}.yaml").write_text(json.dumps(config))


def main():
    recipes = tomllib.loads((ROOT / "recipes/benchmarks.toml").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paper", choices=recipes)
    parser.add_argument("--size", choices=["3b", "7b"], default="7b")
    parser.add_argument(
        "--checkpoint", type=Path, help="Override the pinned Qwen snapshot"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    tasks = recipes[args.paper].get(f"tasks_{args.size}")
    if not tasks:
        parser.error(f"No published {args.size} suite for {args.paper}")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch size and limit must be positive")
    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import snapshot_download

        revision = {
            "3b": "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "7b": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        }[args.size]
        checkpoint = Path(
            snapshot_download(
                f"Qwen/Qwen2.5-VL-{args.size.upper()}-Instruct",
                revision=revision,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"],
            )
        )
    if not (checkpoint / "config.json").is_file():
        parser.error("--checkpoint must point to a downloaded model snapshot")
    model_args = f"model={checkpoint.resolve()}"
    if args.paper in {"lvr", "hylar"}:
        max_tokens = 5120 if args.paper == "lvr" else 16384
        image_size = {"shortest_edge": 128 * 28**2, "longest_edge": max_tokens * 28**2}
        config = json.dumps({"image": {"size": image_size}}, separators=(",", ":"))
        model_args += f",mm_process_config={config}"
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
        "--batch_size",
        str(args.batch_size),
        "--log_samples",
        "--fail_on_error",
        "--output_path",
        str(args.output.resolve()),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    with TemporaryDirectory() as directory:
        write_task_configs(Path(directory), tasks)
        subprocess.run([*command, "--tasks", directory], check=True)


if __name__ == "__main__":
    main()
