"""Run one latent GRPO update on the first four multiple-choice ViRL prompts."""

import argparse
import hashlib
import json
import math
import re
from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

from . import grpo_release
from .training.__main__ import load_base

ANNOTATIONS_SHA256 = "4527db825fb5725e01e849e6386f8d943315b6ec2f157cbb086911e15a2d7f8f"
GROUP = 8
LATENTS = 8
TEMPERATURE = 0.9
MAX_TOKENS = 64


class Action(NamedTuple):
    token: int
    latent_input: torch.Tensor | None
    sampled_logp: float


def prepare(model, processor, row, images, *, released=False):
    question, answer = row["conversations"]
    text = question["value"]
    if released:
        text = re.sub(r"\n?<image>\n?", "", text)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": text}],
        },
    ]
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    with Image.open(images / row["image"]) as im:
        batch = processor(
            text=[prompt], images=[im.convert("RGB")], return_tensors="pt"
        ).to(model.device)
    ids = batch.input_ids
    positions, _ = model.model.get_rope_index(
        ids, batch.mm_token_type_ids, image_grid_thw=batch.image_grid_thw
    )
    features = model.model.get_image_features(
        batch.pixel_values, batch.image_grid_thw, return_dict=True
    )
    vision = torch.cat(features.pooler_output)
    if ids.eq(model.config.image_token_id).sum() != len(vision):
        raise ValueError(f"Image token count mismatch on {row['id']}")
    if released:
        if ids.shape[1] > grpo_release.MAX_PROMPT:
            ids = ids[:, -grpo_release.MAX_PROMPT :]
            positions = positions[..., -grpo_release.MAX_PROMPT :]
            vision = vision[
                len(vision) - ids.eq(model.config.image_token_id).sum().item() :
            ]
        return (ids, positions, vision), answer["value"]
    return (ids, positions, vision), re.fullmatch(
        r"<answer>(.+?)</answer>", answer["value"], re.DOTALL
    )[1]


def embed_prompt(model, prompt):
    ids, _, vision = prompt
    embeds = model.get_input_embeddings()(ids)
    return embeds.masked_scatter(
        (ids == model.config.image_token_id).unsqueeze(-1), vision.to(embeds.dtype)
    )


@torch.no_grad()
def rollout(model, prompt, stops, *, released=False):
    _, positions, _ = prompt
    embeds = embed_prompt(model, prompt).expand(GROUP, -1, -1)
    pos = positions.expand(-1, GROUP, -1)
    lm = model.model.language_model
    state = lm(inputs_embeds=embeds, position_ids=pos, use_cache=True)
    cache = state.past_key_values
    remaining = torch.zeros(GROUP, dtype=torch.long, device=model.device)
    alive = torch.ones_like(remaining, dtype=torch.bool)
    trajectories = [[] for _ in range(GROUP)]
    max_tokens = grpo_release.MAX_COMPLETION if released else MAX_TOKENS
    for t in range(max_tokens):
        hidden = state.last_hidden_state[:, -1]
        scores = model.lm_head(hidden).float() / TEMPERATURE
        latent = remaining > 0
        if released:
            cutoff = scores.topk(grpo_release.TOP_K).values[:, -1:]
            weights = scores.masked_fill(scores < cutoff, -torch.inf).softmax(dim=-1)
        else:
            for stop in stops:
                scores[:, stop].masked_fill_(latent, -torch.inf)
        distribution = scores.log_softmax(dim=-1)
        tokens = torch.multinomial(weights if released else distribution.exp(), 1)
        chosen = distribution.gather(1, tokens).squeeze(1).tolist()
        active = alive.tolist()
        slots = latent.tolist()
        for i, token in enumerate(tokens.flatten().tolist()):
            if active[i]:
                trajectories[i].append(
                    Action(token, hidden[i].clone() if slots[i] else None, chosen[i])
                )
        for stop in stops:
            ended = tokens[:, 0] == stop
            if released:
                ended &= ~latent
            alive &= ~ended
        if not alive.any() or t + 1 == max_tokens:
            break
        embeds = torch.where(
            latent[:, None], hidden, model.get_input_embeddings()(tokens[:, 0])
        )
        remaining = torch.where(
            latent,
            remaining - 1,
            (tokens[:, 0] == model.config.lvr_start_id).long() * LATENTS,
        )
        pos = (positions.max() + t + 1).expand(3, GROUP, 1)
        state = lm(
            inputs_embeds=embeds[:, None],
            position_ids=pos,
            past_key_values=cache,
            use_cache=True,
        )
    return trajectories


# Long image prefixes can exhaust memory when gradients and AdamW are resident.
CHECKPOINT_TOKENS = 1024


@contextmanager
def checkpoint_long_replay(decoder, length):
    if length < CHECKPOINT_TOKENS or not torch.is_grad_enabled():
        yield
        return
    if not decoder.is_gradient_checkpointing:
        decoder.gradient_checkpointing_enable({"use_reentrant": False})
    # Only the decoder-layer wrappers need training mode for recomputation.
    # Attention, dropout and the rest of the policy keep their rollout behavior.
    modes = [layer.training for layer in decoder.layers]
    for layer in decoder.layers:
        layer.training = True
    try:
        yield
    finally:
        for layer, mode in zip(decoder.layers, modes, strict=True):
            layer.training = mode


def replay_distribution(model, prompt, trajectory, token_mask=None):
    _, pos, _ = prompt
    device = model.device
    ids = torch.tensor([a.token for a in trajectory], device=device)
    embeds = model.get_input_embeddings()(ids)
    # Each saved vector is the input at the sampled token's position.
    for i, (_, vector, _) in enumerate(trajectory):
        if vector is not None:
            embeds[i] = vector.to(embeds.dtype)
    inputs = torch.cat([embed_prompt(model, prompt), embeds[None, :-1]], dim=1)
    tail = (pos.max() + 1 + torch.arange(len(ids) - 1, device=device)).expand(3, 1, -1)
    positions = torch.cat([pos, tail], dim=2)
    decoder = model.model.language_model
    with checkpoint_long_replay(decoder, inputs.shape[1]):
        hidden = decoder(
            inputs_embeds=inputs, position_ids=positions, use_cache=False
        ).last_hidden_state
    tail_hidden = hidden[0, -len(ids) :]
    if token_mask is None:
        # Calibration scores the opening decision.
        mask = torch.tensor([a.latent_input is None for a in trajectory], device=device)
        scores = model.lm_head(tail_hidden).float() / TEMPERATURE
        scores = scores[mask]
    else:
        mask = torch.tensor(token_mask, device=device)
        scores = model.lm_head(tail_hidden[mask]).float() / TEMPERATURE
    return scores.log_softmax(-1), ids[mask, None], mask


def replay(model, prompt, trajectory):
    distribution, targets, mask = replay_distribution(model, prompt, trajectory)
    return distribution.gather(1, targets).squeeze(1), mask


# The calibration accepts plain decimal and rational answers from ViRL.
NUMBER = re.compile(r"[+-]?(?:[0-9]+/[1-9][0-9]*|[0-9]+(?:\.[0-9]+)?|\.[0-9]+)")


def number(text):
    if len(text) <= 128 and NUMBER.fullmatch(text):
        return Fraction(text)
    return None


def rewards(text, target):
    answer = re.search(r"<answer>(.*?)</answer>", text)
    prediction = (answer[1] if answer else text).strip()
    target_number = number(target)
    correct = (
        number(prediction) == target_number
        if target_number is not None
        else prediction == target
    )
    formatted = re.fullmatch(
        r"<\|lvr_start\|>.*?<\|lvr_end\|>\s*<answer>.*?</answer>", text
    )
    return int(correct), int(formatted is not None)


def load_processor(checkpoint, *, released=False):
    processor = AutoProcessor.from_pretrained(checkpoint)
    processor.image_processor.size = {
        "shortest_edge": 128 * 28**2,
        "longest_edge": (grpo_release.MAX_IMAGE_TOKENS if released else 5120) * 28**2,
    }
    return processor


def load_policy(checkpoint, model_class=Qwen2_5_VLForConditionalGeneration):
    model = (
        load_base(
            model_class,
            checkpoint,
            dtype=torch.float32,
            attn_implementation="sdpa",
        )
        .cuda()
        .eval()
    )
    model.model.visual.requires_grad_(False)
    return model


def make_optimizer(model, learning_rate=1e-5, *, released=False):
    params = [param for param in model.parameters() if param.requires_grad]
    if released:
        # The Qwen decoder's biases and normalization scales are one-dimensional.
        params = [
            {
                "params": [p for p in params if p.ndim > 1],
                "weight_decay": grpo_release.WEIGHT_DECAY,
            },
            {"params": [p for p in params if p.ndim <= 1]},
        ]
    return torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0, fused=True)


def run_update(
    model, processor, optimizer, rows, images, reference=None, *, released=False
):
    tokenizer = processor.tokenizer
    eos = tokenizer.eos_token_id
    controls = {eos, tokenizer.pad_token_id}
    excluded = controls
    reward = rewards
    if released:
        controls = {eos}
        excluded = set(tokenizer.all_special_ids) - {
            model.config.lvr_start_id,
            model.config.lvr_end_id,
            model.config.lvr_id,
            model.config.lvr_latent_end_id,
        }
        reward = grpo_release.rewards
    summary = []
    probes = []
    kl_means = []
    conditional_kls = []
    torch.cuda.synchronize()
    started = perf_counter()
    replay_errors = []
    for row in rows:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prompt, target = prepare(model, processor, row, images, released=released)
            paths = rollout(model, prompt, controls, released=released)
        texts = [
            tokenizer.decode(
                [tok for tok, _, _ in path if tok not in excluded],
                skip_special_tokens=False,
            )
            for path in paths
        ]
        reward_pairs = [reward(text, target) for text in texts]
        total = torch.tensor(
            [sum(pair) for pair in reward_pairs], device="cuda", dtype=torch.float32
        )
        advantages = (total - total.mean()) / (total.std() + 1e-4)
        masks = [None] * len(paths)
        denominators = [None] * len(paths)
        if released:
            masks = [
                grpo_release.completion_mask(
                    path, model.config.lvr_start_id, model.config.lvr_end_id, eos
                )
                for path in paths
            ]
            denominators = grpo_release.token_denominators(masks)
        for path, advantage, token_mask, denominator in zip(
            paths, advantages, masks, denominators, strict=True
        ):
            if token_mask is not None and not any(token_mask):
                continue
            with torch.autocast("cuda", dtype=torch.bfloat16):
                policy_dist, targets, mask = replay_distribution(
                    model, prompt, path, token_mask
                )
                logp = policy_dist.gather(1, targets).squeeze(1)
            sampled = torch.tensor([a.sampled_logp for a in path], device="cuda")[mask]
            old = logp.detach()
            ratio = (logp - old).exp()
            if reference is None:
                reference_logp = old
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    ref_dist, _, _ = replay_distribution(
                        reference, prompt, path, token_mask
                    )
                    reference_logp = ref_dist.gather(1, targets).squeeze(1)
                    # Sum over the vocabulary on the very same replayed prefixes.
                    detached = policy_dist.detach()
                    conditional_kls.append(
                        (detached.exp() * (detached - ref_dist)).sum(-1).mean()
                    )
                    del ref_dist, detached
            difference = reference_logp - logp
            kl = difference.expm1() - difference
            kl_means.append(kl.detach().mean())
            objective = (
                torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
                - 0.04 * kl
            )
            if released:
                divisor = len(rows) * (GROUP // grpo_release.MICROBATCH) * denominator
                (-objective.sum() / divisor).backward()
            else:
                (-objective.mean() / (len(rows) * GROUP)).backward()
                probes.append((prompt, path, old.cpu()))
            replay_errors.append((old - sampled).abs().cpu())
        summary.append(
            {
                "id": row["id"],
                "image": row["image"],
                "target": target,
                "texts": texts,
                "rewards": reward_pairs,
                "latent_inputs": [
                    sum(v is not None for _, v, _ in path) for path in paths
                ],
            }
        )
        if released:
            summary[-1]["completion_lengths"] = [len(path) for path in paths]
            summary[-1]["scored_tokens"] = [sum(mask) for mask in masks]
        print(row["id"], reward_pairs, flush=True)
    if not replay_errors:
        raise RuntimeError("No scored completion tokens in the update")
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True
    ).item()
    if not norm and reference is None:
        raise RuntimeError("No gradient from the sampled groups")
    optimizer.step()
    torch.cuda.synchronize()
    update_seconds = perf_counter() - started
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        changes = [
            (replay(model, prompt, path)[0].cpu() - old).abs().mean().item()
            for prompt, path, old in probes
        ]
    result = {
        "seed": 42,
        "groups": summary,
        "grad_norm": norm,
        "replay_max_abs_logp_error": torch.cat(replay_errors).max().item(),
        "replay_mean_abs_logp_error": torch.cat(replay_errors).mean().item(),
        "update_seconds": update_seconds,
        "peak_gpu_GiB": torch.cuda.max_memory_allocated() / 1024**3,
    }
    if changes:
        result["mean_abs_logp_change"] = sum(changes) / len(changes)
    if reference is not None:
        result["mean_reference_kl"] = torch.stack(kl_means).mean().item()
        result["mean_conditional_kl"] = torch.stack(conditional_kls).mean().item()
        result["step_with_diagnostics_seconds"] = perf_counter() - started
    if not all(math.isfinite(v) for v in changes) or (
        reference is None and not any(changes)
    ):
        raise RuntimeError("The optimizer step did not produce a finite policy change")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "annotations", "images", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    raw = args.annotations.read_bytes()
    if hashlib.sha256(raw).hexdigest() != ANNOTATIONS_SHA256:
        parser.error("Expected the original LVR ViRL annotation file")
    if args.output.exists():
        parser.error("Use a fresh output file")
    rows = [
        r
        for r in json.loads(raw)
        if isinstance(r["image"], str)
        and re.fullmatch(r"<answer>[A-D]</answer>", r["conversations"][1]["value"])
    ][:4]
    set_seed(42)
    processor = load_processor(args.checkpoint)
    model = load_policy(args.checkpoint)
    result = run_update(model, processor, make_optimizer(model), rows, args.images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
