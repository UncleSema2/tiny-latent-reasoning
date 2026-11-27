"""Cache teacher layer targets and paired utility scores for the student curriculum."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import Qwen2_5_VLForConditionalGeneration

from .curriculum import answer_score, build_curriculum
from .teacher import REVISION, Teacher, prepare_example
from .validate_teacher import judge_scores

CHUNK = 16


def write_json(path, obj):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, allow_nan=False))
    temp.replace(path)


def load_model(path, device):
    model, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        path,
        revision=REVISION,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(info.values()):
        raise ValueError(f"Incomplete checkpoint: {info}")
    return model


@torch.inference_mode()
def score_chunks(args, rows, processor, boundaries, device, rank, world):
    cache = args.output / "scores"
    chunks = range(rank, (len(rows) + CHUNK - 1) // CHUNK, world)
    remaining = [i for i in chunks if not (cache / f"{i}.json").is_file()]
    if not remaining:
        return
    teacher = Teacher(load_model(args.teacher, device), boundaries).eval()
    judge = load_model(args.base, device).eval()
    judge.get_input_embeddings().weight[boundaries] = (
        teacher.base.get_input_embeddings().weight[boundaries]
    )
    for chunk in remaining:
        scores, targets = [], {}
        for index in range(chunk * CHUNK, min((chunk + 1) * CHUNK, len(rows))):
            row = rows[index]
            prompt, answer = prepare_example(processor, row, args.images, device)
            _, z, layers = teacher(prompt, answer, collect_layers=True)
            good, control = judge_scores(
                judge,
                boundaries,
                prompt,
                answer,
                z,
                torch.Generator(device=device).manual_seed(43 + index),
                per_token=True,
            ).tolist()
            scores.append(
                {
                    "id": row["id"],
                    "subset": row["subset"],
                    "teacher_logprobs": good,
                    "control_logprobs": control,
                }
            )
            if answer_score(good) - answer_score(control) > -0.5:
                targets[row["id"]] = layers.cpu().contiguous()
        path = cache / f"{chunk}.safetensors"
        if targets:
            tmp = path.with_suffix(".partial")
            save_file(targets, tmp)
            tmp.replace(path)
        # Publish scores last. A partial chunk can be recomputed after interruption.
        write_json(cache / f"{chunk}.json", scores)
        if chunk == remaining[0] or (chunk // world) % 10 == 0:
            print(
                json.dumps(
                    {"rank": rank, "scored_chunk": chunk, "retained": len(targets)}
                ),
                flush=True,
            )


def curriculum(rows, cache: Path):
    scores = []
    for index in range(0, len(rows), CHUNK):
        chunk = json.loads((cache / f"{index // CHUNK}.json").read_text())
        if [r["id"] for r in chunk] != [r["id"] for r in rows[index : index + CHUNK]]:
            raise ValueError(f"Incomplete training scores at row {index}")
        scores.extend(chunk)
    return build_curriculum(scores, seed=42)
