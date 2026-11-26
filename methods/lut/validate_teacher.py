"""Measure held-out teacher answer loss and trajectory utility with a frozen judge."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .prepare import SHA256, project
from .teacher import BASE, BOUNDARIES, REVISION, Teacher, prepare_example


def held_out(annotations, train, count):
    raw = annotations.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Unexpected Visual-CoT revision")
    rows = json.loads(raw)
    with train.open() as f:
        images = {json.loads(line)["image"] for line in f}
    order = list(range(len(rows)))
    random.Random(43).shuffle(order)
    result = []
    for i in order:
        row = project(rows[i], i)
        if row["image"] in images:
            continue
        result.append(row)
        images.add(row["image"])
        if len(result) == count:
            return result
    raise ValueError("Insufficient held-out images")


def judge_scores(base, boundaries, prompt, answer, trajectory, generator):
    """Score both trajectories with identical image, prompt, delimiters and answer."""
    start, end = boundaries
    n = trajectory.size(1)
    prefix = prompt["input_ids"]
    ids = torch.cat(
        [prefix, prefix.new_tensor([[start] * (n + 1) + [end]]), answer], dim=1
    )
    image = ids == base.config.image_token_id
    pos, _ = base.model.get_rope_index(ids, image.int(), prompt["image_grid_thw"])
    visual = torch.cat(
        base.get_image_features(
            prompt["pixel_values"], prompt["image_grid_thw"]
        ).pooler_output
    )
    emb = base.get_input_embeddings()(ids).masked_scatter(image[..., None], visual)
    control = torch.randn(
        trajectory.shape, device=ids.device, dtype=torch.float32, generator=generator
    )
    control = F.normalize(control, dim=-1) * trajectory.float().norm(
        dim=-1, keepdim=True
    )
    offset = prefix.size(1) + 1
    embeds = emb[:, :-1].repeat(2, 1, 1)
    embeds[:, offset : offset + n] = torch.cat(
        [trajectory, control.to(trajectory.dtype)]
    )
    hidden = base.model.language_model(
        inputs_embeds=embeds,
        position_ids=pos[..., :-1].expand(-1, 2, -1),
        use_cache=False,
    ).last_hidden_state
    logits = base.lm_head(hidden[:, -answer.size(1) :]).float()
    return (
        logits.log_softmax(-1)
        .gather(-1, answer.expand(2, -1)[..., None])
        .mean(dim=(1, 2))
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--count", type=int, default=512)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    selected = [
        held_out(args.annotations, args.train, args.count) if rank == 0 else None
    ]
    dist.broadcast_object_list(selected, src=0)
    processor = AutoProcessor.from_pretrained(args.teacher)
    boundaries = processor.tokenizer.convert_tokens_to_ids(list(BOUNDARIES))
    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": {"": device},
        "attn_implementation": "sdpa",
    }
    teacher = Teacher(
        Qwen2_5_VLForConditionalGeneration.from_pretrained(args.teacher, **kwargs),
        boundaries,
    ).eval()
    judge = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base, revision=REVISION, **kwargs
    ).eval()
    results = []
    for i in range(rank, args.count, world):
        row = selected[0][i]
        prompt, answer = prepare_example(processor, row, args.images, device)
        logits, trajectory = teacher(prompt, answer, return_trajectory=True)
        correct = logits.argmax(-1) == answer
        scores = judge_scores(
            judge,
            boundaries,
            prompt,
            answer,
            trajectory,
            torch.Generator(device=device).manual_seed(43 + i),
        )
        results.append(
            {
                "id": row["id"],
                "subset": row["subset"],
                "tokens": answer.numel(),
                "correct_tokens": correct.sum().item(),
                "exact_prefix": correct.all().item(),
                "ce": F.cross_entropy(logits.flatten(0, 1), answer.flatten()).item(),
                "teacher_score": scores[0].item(),
                "control_score": scores[1].item(),
                "delta": (scores[0] - scores[1]).item(),
            }
        )
    shards = [None] * world if rank == 0 else None
    dist.gather_object(results, shards, dst=0)
    if rank == 0:
        rows = sorted((r for shard in shards for r in shard), key=lambda r: r["id"])
        tokens = sum(r["tokens"] for r in rows)
        summary = {
            "examples": len(rows),
            "tokens": tokens,
            "answer_token_accuracy": sum(r["correct_tokens"] for r in rows) / tokens,
            "exact_answer_prefix_rate": sum(r["exact_prefix"] for r in rows)
            / len(rows),
            "mean_answer_ce": sum(r["ce"] for r in rows) / len(rows),
            "mean_delta": sum(r["delta"] for r in rows) / len(rows),
            "positive_delta_rate": sum(r["delta"] > 0 for r in rows) / len(rows),
            "retained_rate": sum(r["delta"] > -0.5 for r in rows) / len(rows),
        }
        report = args.teacher / "held-out.json"
        tmp = report.with_suffix(".tmp")
        tmp.write_text(json.dumps({"summary": summary, "rows": rows}, allow_nan=False))
        tmp.replace(report)
        print(json.dumps(summary), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
