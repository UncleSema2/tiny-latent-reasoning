"""Replay LUT answer scores and compare latent inputs on the same held-out images."""

import argparse
import json
import os
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from .teacher import BASE, BOUNDARIES, REVISION, Teacher, prepare_example, teacher_mask
from .validate_teacher import held_out, judge_scores


def native_score(
    model,
    prompt,
    answer,
    visual,
    boundaries=(),
    states=None,
    boundary_vectors=None,
    bottleneck=False,
):
    """Use the native VLM forward and its shifted-label loss for a single sequence."""
    prefix = prompt.input_ids
    n = 0 if states is None else states.shape[1]
    span = [boundaries[0]] * (1 + n) + [boundaries[1]] if boundaries else []
    ids = torch.cat([prefix, prefix.new_tensor([span]).long(), answer], dim=1)
    image = ids == model.config.image_token_id
    offset = prefix.size(1)

    def inject(_module, _args, embeds):
        if states is not None:
            embeds[:, offset + 1 : offset + 1 + n] = states
        if boundary_vectors is not None:
            embeds[:, [offset, offset + 1 + n]] = boundary_vectors
        return embeds

    kw = {}
    if bottleneck:
        pos, _ = model.model.get_rope_index(ids, image.int(), prompt.image_grid_thw)
        latent = torch.zeros_like(image[0])
        latent[offset : offset + 1 + n] = True
        kw.update(position_ids=pos, attention_mask=teacher_mask(image[0], latent))
    labels = torch.full_like(ids, -100)
    labels[:, -answer.size(1) :] = answer
    hook = model.get_input_embeddings().register_forward_hook(inject)
    try:
        # Include one predecessor for the native causal loss shift.
        keep = answer.size(1) + 1
        result = model(
            input_ids=ids,
            image_grid_thw=prompt.image_grid_thw,
            mm_token_type_ids=image.int(),
            mm_encoder_outputs={"image": visual},
            use_cache=False,
            logits_to_keep=keep,
            labels=labels[:, -keep:],
            **kw,
        )
        return -result.loss.item()
    finally:
        hook.remove()


def random_direction(states, seed):
    rng = torch.Generator(device=states.device).manual_seed(seed)
    return F.normalize(
        torch.randn(states.shape, generator=rng, device=states.device), dim=-1
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("teacher", "images", "train", "annotations", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()
    if args.skip < 0 or args.count < 1:
        parser.error("--skip must be nonnegative and --count must be positive")
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    rows = held_out(args.annotations, args.train, args.skip + args.count)[args.skip :]
    previous = {
        r["id"]: r
        for r in json.loads((args.teacher / "held-out.json").read_text())["rows"]
    }
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
    teacher_bounds = teacher.base.get_input_embeddings().weight[boundaries]
    judge_bounds = judge.get_input_embeddings().weight[boundaries]
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"rank-{rank}.jsonl"
    shard = list(range(rank, len(rows), world))

    def extract(i):
        prompt, answer = prepare_example(processor, rows[i], args.images, device)
        last = []

        # Input eight is consumed at position zero of the last decoder call.
        def capture(_module, _args, out):
            last[:] = [out.last_hidden_state[:, :1].clone()]

        hook = teacher.base.model.language_model.register_forward_hook(capture)
        try:
            logits, states = teacher(prompt, answer, return_trajectory=True)
        finally:
            hook.remove()
        return (
            prompt,
            answer,
            logits,
            states,
            torch.cat([states[:, 1:], last[0]], dim=1),
        )

    if len(shard) < 2:
        raise ValueError(
            "The wrong-image control requires at least two distinct images per rank"
        )
    wrong = extract(shard[-1])[-2]
    with path.open("x", buffering=1) as dest:
        for i in shard:
            row = rows[i]
            prompt, answer, logits, z, post = extract(i)
            visual = judge.get_image_features(
                prompt.pixel_values, prompt.image_grid_thw
            )

            score = partial(native_score, judge, prompt, answer, visual, boundaries)

            norm = z.float().norm(dim=-1, keepdim=True)
            index = args.skip + i
            direction = random_direction(z, 43 + index)
            control = (direction * norm).to(z.dtype)
            post_control = (direction * post.float().norm(dim=-1, keepdim=True)).to(
                z.dtype
            )
            other_image = (F.normalize(wrong.float(), dim=-1) * norm).to(z.dtype)
            text = prompt.input_ids[
                0, prompt.input_ids[0] != judge.config.image_token_id
            ]
            text_norm = judge.get_input_embeddings()(text).float().norm(dim=-1).median()
            ref = judge_scores(
                judge,
                boundaries,
                prompt,
                answer,
                z,
                torch.Generator(device=device).manual_seed(43 + index),
            )
            original = previous.get(row["id"])
            ce = F.cross_entropy(logits.flatten(0, 1), answer.flatten()).item()
            teacher_visual = teacher.base.get_image_features(
                prompt.pixel_values, prompt.image_grid_thw
            )
            result = {
                "id": row["id"],
                "image": row["image"],
                "subset": row["subset"],
                "tokens": answer.numel(),
                "saved": original["delta"] if original else None,
                "replayed": (ref[0] - ref[1]).item(),
                "saved_ce_error": ce - original["ce"] if original else None,
                "cached_teacher": -ce,
                "replayed_teacher": native_score(
                    teacher.base,
                    prompt,
                    answer,
                    teacher_visual,
                    boundaries,
                    z,
                    bottleneck=True,
                ),
                "judge_paired": ref.tolist(),
                "native": score(z),
                "random": score(control),
                "random2": score(
                    (random_direction(z, 1703 + index) * norm).to(z.dtype)
                ),
                "shifted": score(post),
                "shifted_control": score(post_control),
                "scaled": score(
                    (F.normalize(z.float(), dim=-1) * text_norm).to(z.dtype)
                ),
                "scaled_control": score((direction * text_norm).to(z.dtype)),
                "wrong_image": score(other_image),
                "trained_bounds_wrong_image": score(
                    other_image, boundary_vectors=teacher_bounds
                ),
                "trained_bounds_shifted": score(post, boundary_vectors=teacher_bounds),
                "trained_bounds_shifted_control": score(
                    post_control, boundary_vectors=teacher_bounds
                ),
                "trained_bounds": score(z, boundary_vectors=teacher_bounds),
                "trained_bounds_control": score(
                    control, boundary_vectors=teacher_bounds
                ),
                "bare": native_score(judge, prompt, answer, visual),
                "boundary_only": score(None),
                "latent_norm": norm.mean().item(),
                "text_norm": text_norm.item(),
                "relative_norm_error": (
                    control.float().norm(dim=-1) / norm.squeeze(-1) - 1
                )
                .abs()
                .max()
                .item(),
                "base_boundary_norms": judge_bounds.float().norm(dim=-1).tolist(),
                "teacher_boundary_norms": teacher_bounds.float().norm(dim=-1).tolist(),
                "boundary_ids": boundaries,
            }
            dest.write(json.dumps(result, allow_nan=False) + "\n")
            wrong = z


if __name__ == "__main__":
    main()
