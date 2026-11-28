"""Run a Qwen baseline with a paper's evaluation tasks."""

import argparse
import json
import os
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
    parser.add_argument(
        "--lvr-steps",
        type=int,
        choices=(4, 8, 16),
        help="Use fixed-step decoding for an LVR checkpoint",
    )
    parser.add_argument(
        "--lut-steps",
        type=int,
        choices=(8, 10, 12, 16),
        help="Use fixed-step decoding for a local LUT checkpoint",
    )
    parser.add_argument(
        "--lut-core",
        action="store_true",
        help="Evaluate the four perception benchmarks with the LUT training image limits",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--nccl-port",
        type=int,
        help="Use a dedicated SGLang rendezvous port for a concurrent evaluation",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    tasks = recipes[args.paper].get(f"tasks_{args.size}")
    if not tasks:
        parser.error(f"No published {args.size} suite for {args.paper}")
    if args.batch_size < 1 or (args.limit is not None and args.limit < 1):
        parser.error("batch size and limit must be positive")
    if args.nccl_port is not None and not 1 <= args.nccl_port <= 65535:
        parser.error("--nccl-port must be between 1 and 65535")
    if args.lvr_steps and (
        args.paper != "lvr" or (args.size == "3b" and not args.checkpoint)
    ):
        parser.error("--lvr-steps requires the LVR suite and a local checkpoint for 3B")
    if (args.lut_steps or args.lut_core) and args.paper != "lut":
        parser.error("--lut-steps and --lut-core require the LUT suite")
    if args.lut_steps and args.checkpoint is None:
        parser.error("--lut-steps requires a local LUT checkpoint")
    if args.lut_core:
        tasks = recipes[args.paper]["core_tasks"]
    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import snapshot_download

        revision = {
            "3b": "66285546d2b821cf421d4f5eb2576359d3770cd3",
            "7b": "cc594898137f460bfe9f0759e9844b3ce807cfb5",
        }[args.size]
        repo = f"Qwen/Qwen2.5-VL-{args.size.upper()}-Instruct"
        if args.lvr_steps:
            repo = "vincentleebang/LVR-7B"
            revision = "413c55c908c152f47538320615178cddcf73201a"
        checkpoint = Path(
            snapshot_download(
                repo,
                revision=revision,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.jinja"],
            )
        )
    if not (checkpoint / "config.json").is_file():
        parser.error("--checkpoint must point to a downloaded model snapshot")
    # Keep memory available for the vision encoder and high-resolution image resizing.
    model_args = (
        f"model={checkpoint.resolve()},gpu_memory_utilization=0.55,"
        f"max_running_requests={args.batch_size},mm_processor_worker_num=1"
    )
    if args.nccl_port is not None:
        model_args += f",nccl_port={args.nccl_port}"
    run_kwargs = {}
    model_package = None
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("architectures") == ["QwenWithLVR"] and not args.lvr_steps:
        parser.error("Specify --lvr-steps for an LVR checkpoint")
    if args.lvr_steps:
        if config["architectures"] != ["QwenWithLVR"]:
            parser.error("--lvr-steps requires an LVR checkpoint")
        override = {
            "architectures": ["Qwen2_5_VLForConditionalGeneration"],
            "lvr_steps": args.lvr_steps,
        }
        model_args += ",json_model_override_args=" + json.dumps(
            override, separators=(",", ":")
        )
        model_args += ",disable_radix_cache=True"
    if args.lvr_steps or (checkpoint / "sft_recipe.json").is_file():
        # Float32 optimizer weights must use the BF16 training compute dtype.
        model_args += ",dtype=bfloat16"
        model_package = "methods.lvr"
    if args.lut_steps:
        tokenizer_file = checkpoint / "tokenizer.json"
        if not tokenizer_file.is_file():
            parser.error("The LUT checkpoint must contain tokenizer.json")
        tokens = {
            row["content"]: row["id"]
            for row in json.loads(tokenizer_file.read_text())["added_tokens"]
            if row["special"]
        }
        if not {"<latent>", "</latent>"} <= tokens.keys():
            parser.error("The LUT checkpoint tokenizer must define both delimiters")
        override = {
            "lut_steps": args.lut_steps,
            "lut_start_id": tokens["<latent>"],
            "lut_end_id": tokens["</latent>"],
        }
        model_args += ",json_model_override_args=" + json.dumps(
            override, separators=(",", ":")
        )
        model_args += ",dtype=bfloat16,disable_radix_cache=True"
        model_package = "methods.lut"
    if model_package:
        run_kwargs["env"] = {
            **os.environ,
            "SGLANG_EXTERNAL_MODEL_PACKAGE": model_package,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(ROOT), os.environ.get("PYTHONPATH")))
            ),
        }
    if args.paper in {"lvr", "hylar"} or args.lut_core:
        max_tokens = {"lvr": 5120, "hylar": 16384, "lut": 512}[args.paper]
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
    if args.lvr_steps:
        command.extend(["--gen_kwargs", "max_new_tokens=512,temperature=0"])
    if args.lut_core or args.lut_steps:
        overhead = args.lut_steps + 2 if args.lut_steps else 0
        answer_budget = 32 if args.lut_core else 512
        command.extend(
            ["--gen_kwargs", f"max_new_tokens={answer_budget + overhead},temperature=0"]
        )
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    with TemporaryDirectory() as directory:
        write_task_configs(Path(directory), tasks)
        subprocess.run([*command, "--tasks", directory], check=True, **run_kwargs)


if __name__ == "__main__":
    main()
