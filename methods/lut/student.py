"""Score the full dataset, build the LUT curriculum and distill a fresh base student."""

import argparse
import fcntl
import hashlib
import json
import os
from datetime import timedelta
from math import ceil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors import safe_open
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoProcessor, set_seed

from .score_training import CHUNK, curriculum, load_model, score_chunks, write_json
from .teacher import (
    BASE,
    BOUNDARIES,
    REVISION,
    Teacher,
    epoch_batches,
    save_training_state,
)


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def recipe_for(args, world):
    files = sorted(args.teacher.glob("*.safetensors"))
    if not files:
        raise ValueError("Expected a teacher checkpoint with safetensors weights")
    return {
        "base": str(args.base),
        "revision": REVISION,
        "teacher_files": {p.name: digest(p) for p in files},
        "processor_files": {
            p.name: digest(p)
            for p in sorted(args.teacher.iterdir())
            if p.suffix == ".jinja"
            or p.name
            in (
                "config.json",
                "processor_config.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            )
        },
        "data_sha256": digest(args.data),
        "world_size": world,
        "score_chunk_size": CHUNK,
        "latent_steps": 8,
        "judge_boundaries": "teacher",
        "target_positions": "slot_outputs",
        "score_positions": "slot_inputs",
        "threshold": -0.5,
        "control_seed": 43,
        "seed": 42,
        "learning_rate": 1e-5,
        "alignment_weight": 1.0,
        "epochs": 1,
    }


def train(args, rows, processor, boundaries, recipe, device, rank, world):
    manifest = args.output / "curriculum.json"
    schedule = json.loads(manifest.read_text())
    ids = schedule["low"] + schedule["high"]
    if not ids:
        raise ValueError("No trajectories passed the utility threshold")
    by_id = {r["id"]: (i, r) for i, r in enumerate(rows)}
    ordered = [by_id[key][1] for key in ids]
    updates = ceil(len(ordered) / world)
    recipe = dict(recipe, curriculum_sha256=digest(manifest), updates=updates)
    checkpoints = sorted(
        args.output.glob("checkpoint-[0-9]*"), key=lambda p: int(p.name.split("-")[1])
    )
    resume = checkpoints[-1] if checkpoints else None
    completed = 0
    if resume:
        state = json.loads((resume / "training_state.json").read_text())
        completed = state["step"]
        if state["recipe"] != recipe or not 0 < completed <= updates:
            raise ValueError("Checkpoint does not match the student curriculum")
        if completed == updates:
            return
    base = load_model(resume or args.base, device)
    base.model.visual.requires_grad_(False)
    model = DistributedDataParallel(
        Teacher(base, boundaries, bottleneck=False).train(),
        device_ids=[device.index],
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )
    optim = ZeroRedundancyOptimizer(
        [p for p in model.parameters() if p.requires_grad],
        optimizer_class=torch.optim.AdamW,
        fused=True,
        lr=recipe["learning_rate"],
    )
    if resume:
        optim.optim.load_state_dict(
            torch.load(
                resume / f"optimizer-{rank}.pt", weights_only=True, map_location="cpu"
            )
        )
        cpu, gpu = torch.load(
            resume / f"rng-{rank}.pt", weights_only=True, map_location="cpu"
        )
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state(gpu)
    batches = epoch_batches(ordered, rank, world, completed, processor, args.images)
    for step, (prompt, answer, weight) in enumerate(batches, completed + 1):
        offset = (step - 1) * world
        active = min(world, len(ids) - offset)
        key = ids[offset + (rank if rank < active else 0)]
        chunk = by_id[key][0] // CHUNK
        with safe_open(
            args.output / "scores" / f"{chunk}.safetensors", framework="pt"
        ) as target:
            layers = target.get_tensor(key).to(device)
        answer = answer.to(device)
        optim.zero_grad(set_to_none=True)
        logits, _, predicted = model(prompt.to(device), answer, collect_layers=True)
        if predicted.shape != layers.shape:
            raise ValueError("Teacher/student layer shapes differ")
        ce = F.cross_entropy(logits.flatten(0, 1), answer.flatten())
        alignment = (
            1 - F.cosine_similarity(predicted.float(), layers.float(), dim=-1).mean()
        )
        ((ce + recipe["alignment_weight"] * alignment) * weight).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optim.step()
        if step == completed + 1 or step % 10 == 0 or step == updates:
            stats = torch.stack([ce.detach(), alignment.detach()]) * weight
            dist.all_reduce(stats, op=dist.ReduceOp.AVG)
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "step": step,
                            "updates": updates,
                            "ce": stats[0].item(),
                            "alignment": stats[1].item(),
                        }
                    ),
                    flush=True,
                )
        if step % args.save_every == 0 or step == updates:
            save_training_state(
                base, processor, optim, args.output, step, recipe, rank, save_rng=True
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "teacher", "images", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--save-every", type=int, default=250)
    args = parser.parse_args()
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    set_seed(42)
    with args.data.open() as source:
        rows = [json.loads(line) for line in source]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty training data with unique IDs")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        lock = (args.output / ".lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recipe = recipe_for(args, world)
        recipe_path = args.output / "recipe.json"
        if recipe_path.exists():
            if json.loads(recipe_path.read_text()) != recipe:
                raise ValueError("Existing output belongs to a different recipe")
        else:
            if any(p.name != ".lock" for p in args.output.iterdir()):
                raise ValueError("Output directory contains unrelated artifacts")
            write_json(recipe_path, recipe)
        (args.output / "scores").mkdir(exist_ok=True)
    dist.barrier()
    recipe = json.loads((args.output / "recipe.json").read_text())
    processor = AutoProcessor.from_pretrained(args.teacher)
    boundaries = processor.tokenizer.convert_tokens_to_ids(list(BOUNDARIES))
    score_chunks(args, rows, processor, boundaries, device, rank, world)
    torch.cuda.empty_cache()
    dist.barrier()
    if rank == 0:
        manifest = curriculum(rows, args.output / "scores")
        write_json(args.output / "curriculum.json", manifest)
        print(
            json.dumps({k: manifest[k] for k in ("input_count", "filtered_count")}),
            flush=True,
        )
    dist.barrier()
    train(args, rows, processor, boundaries, recipe, device, rank, world)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
