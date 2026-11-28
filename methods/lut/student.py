"""Distill a LUT student or continue it with end-of-answer supervision."""

import argparse
import fcntl
import hashlib
import json
import os
from contextlib import nullcontext
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


def continuation_recipe(args, processor, world):
    initial = args.continue_from
    source = initial.parent
    original = json.loads((source / "recipe.json").read_text())
    state = json.loads((initial / "training_state.json").read_text())
    manifest = source / "curriculum.json"
    manifest_hash = digest(manifest)
    if (
        original["data_sha256"] != digest(args.data)
        or state["recipe"]["curriculum_sha256"] != manifest_hash
        or any(state["recipe"].get(key) != value for key, value in original.items())
    ):
        raise ValueError(
            "Initial checkpoint, training data and cached curriculum must match"
        )
    eos = processor.tokenizer.get_added_vocab().get("<|im_end|>")
    generation = json.loads((initial / "generation_config.json").read_text())
    eos_ids = generation["eos_token_id"]
    if eos is None or eos not in (eos_ids if isinstance(eos_ids, list) else [eos_ids]):
        raise ValueError("Expected <|im_end|> to be a generation EOS token")
    weights = sorted(initial.glob("*.safetensors"))
    if not weights:
        raise ValueError("Expected student safetensors weights")
    return dict(
        original,
        world_size=world,
        accumulation_steps=args.accumulation_steps,
        learning_rate=5e-6,
        answer_eos_id=eos,
        source_step=state["step"],
        source_curriculum_sha256=manifest_hash,
        source_checkpoint_files={
            path.name: digest(path)
            for path in weights
            + sorted(initial.glob("*.json"))
            + sorted(initial.glob("*.jinja"))
        },
    )


def train(args, rows, processor, boundaries, recipe, device, rank, world):
    source = args.continue_from.parent if args.continue_from else args.output
    manifest = source / "curriculum.json"
    schedule = json.loads(manifest.read_text())
    ids = schedule["low"] + schedule["high"]
    if not ids:
        raise ValueError("No trajectories passed the utility threshold")
    by_id = {r["id"]: (i, r) for i, r in enumerate(rows)}
    ordered = [by_id[key][1] for key in ids]
    accumulate = args.accumulation_steps
    batch_size = world * accumulate
    updates = ceil(len(ordered) / batch_size)
    eos = recipe.get("answer_eos_id")
    manifest_hash = digest(manifest)
    if args.continue_from and recipe["source_curriculum_sha256"] != manifest_hash:
        raise ValueError("Cached curriculum changed")
    recipe = dict(recipe, curriculum_sha256=manifest_hash, updates=updates)
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
    base = load_model(resume or args.continue_from or args.base, device)
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
    start = completed * accumulate
    batches = epoch_batches(ordered, rank, world, start, processor, args.images)
    for micro, (prompt, answer, weight) in enumerate(batches, start):
        step = micro // accumulate + 1
        offset = micro * world
        active = min(world, len(ids) - offset)
        final_micro = (micro + 1) % accumulate == 0 or offset + active == len(ids)
        weight *= active / min(batch_size, len(ids) - (step - 1) * batch_size)
        log_step = step == completed + 1 or step % 10 == 0 or step == updates
        if micro % accumulate == 0:
            optim.zero_grad(set_to_none=True)
            stats = torch.zeros(2 if eos is None else 3, device=device)
        key = ids[offset + (rank if rank < active else 0)]
        chunk = by_id[key][0] // CHUNK
        with safe_open(
            source / "scores" / f"{chunk}.safetensors", framework="pt"
        ) as target:
            layers = target.get_tensor(key).to(device)
        answer = answer.to(device)
        if eos is not None:
            answer = torch.cat((answer, answer.new_full((1, 1), eos)), dim=1)
        with nullcontext() if final_micro else model.no_sync():
            logits, _, predicted = model(prompt.to(device), answer, collect_layers=True)
            if predicted.shape != layers.shape:
                raise ValueError("Teacher/student layer shapes differ")
            ce = F.cross_entropy(logits.flatten(0, 1), answer.flatten())
            alignment = (
                1
                - F.cosine_similarity(predicted.float(), layers.float(), dim=-1).mean()
            )
            ((ce + recipe["alignment_weight"] * alignment) * weight).backward()
        if log_step:
            losses = [ce.detach(), alignment.detach()]
            if eos is not None:
                losses.append(F.cross_entropy(logits[:, -1].detach(), answer[:, -1]))
            stats += torch.stack(losses) * weight
        if not final_micro:
            continue
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optim.step()
        if log_step:
            dist.all_reduce(stats, op=dist.ReduceOp.AVG)
            if rank == 0:
                ce_avg, align_avg, *eos_ce = stats.tolist()
                report = {
                    "step": step,
                    "updates": updates,
                    "ce": ce_avg,
                    "alignment": align_avg,
                }
                if eos is not None:
                    report["eos_ce"] = eos_ce[0]
                print(json.dumps(report), flush=True)
        if step % args.save_every == 0 or step == updates:
            save_training_state(
                base, processor, optim, args.output, step, recipe, rank, save_rng=True
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "images", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--teacher", type=Path)
    source.add_argument(
        "--continue-from",
        type=Path,
        help="Start from a numbered student checkpoint and supervise <|im_end|> on its cached curriculum",
    )
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    args = parser.parse_args()
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    if args.accumulation_steps <= 0:
        parser.error("--accumulation-steps must be positive")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(hours=2))
    rank, world = dist.get_rank(), dist.get_world_size()
    set_seed(42)
    with args.data.open() as source:
        rows = [json.loads(line) for line in source]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Expected nonempty training data with unique IDs")
    processor = AutoProcessor.from_pretrained(args.continue_from or args.teacher)
    boundaries = processor.tokenizer.convert_tokens_to_ids(list(BOUNDARIES))
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=True)
        lock = (args.output / ".lock").open("a")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recipe = (
            continuation_recipe(args, processor, world)
            if args.continue_from
            else recipe_for(args, world)
        )
        if args.accumulation_steps != 1:
            recipe["accumulation_steps"] = args.accumulation_steps
        recipe_path = args.output / "recipe.json"
        if recipe_path.exists():
            if json.loads(recipe_path.read_text()) != recipe:
                raise ValueError("Existing output belongs to a different recipe")
        else:
            if any(p.name != ".lock" for p in args.output.iterdir()):
                raise ValueError("Output directory contains unrelated artifacts")
            write_json(recipe_path, recipe)
        if args.teacher:
            (args.output / "scores").mkdir(exist_ok=True)
    dist.barrier()
    recipe = json.loads((args.output / "recipe.json").read_text())
    if args.teacher:
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
