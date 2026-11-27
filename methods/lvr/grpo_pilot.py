"""Run one latent GRPO update on the first four multiple-choice ViRL prompts."""

import argparse
import hashlib
import json
import math
import re
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

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


def prepare(model, processor, row, images):
    question, answer = row["conversations"]
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": question["value"]}],
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
def rollout(model, prompt, stops):
    _, positions, _ = prompt
    embeds = embed_prompt(model, prompt).expand(GROUP, -1, -1)
    pos = positions.expand(-1, GROUP, -1)
    lm = model.model.language_model
    state = lm(inputs_embeds=embeds, position_ids=pos, use_cache=True)
    cache = state.past_key_values
    remaining = torch.zeros(GROUP, dtype=torch.long, device=model.device)
    alive = torch.ones_like(remaining, dtype=torch.bool)
    trajectories = [[] for _ in range(GROUP)]
    for t in range(MAX_TOKENS):
        hidden = state.last_hidden_state[:, -1]
        scores = model.lm_head(hidden).float() / TEMPERATURE
        latent = remaining > 0
        for stop in stops:
            scores[:, stop].masked_fill_(latent, -torch.inf)
        distribution = scores.log_softmax(dim=-1)
        tokens = torch.multinomial(distribution.exp(), 1)
        chosen = distribution.gather(1, tokens).squeeze(1).tolist()
        active = alive.tolist()
        slots = latent.tolist()
        for i, token in enumerate(tokens.flatten().tolist()):
            if active[i]:
                trajectories[i].append(
                    Action(token, hidden[i].clone() if slots[i] else None, chosen[i])
                )
        for stop in stops:
            alive &= tokens[:, 0] != stop
        if not alive.any() or t + 1 == MAX_TOKENS:
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


def replay_distribution(model, prompt, trajectory):
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
    hidden = model.model.language_model(
        inputs_embeds=inputs, position_ids=positions, use_cache=False
    ).last_hidden_state
    scores = model.lm_head(hidden[0, -len(ids) :]).float() / TEMPERATURE
    # Discrete actions outside the latent interval, including the opening decision.
    mask = torch.tensor([a.latent_input is None for a in trajectory], device=device)
    return scores[mask].log_softmax(-1), ids[mask, None], mask


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


def load_processor(checkpoint):
    processor = AutoProcessor.from_pretrained(checkpoint)
    processor.image_processor.size = {
        "shortest_edge": 128 * 28**2,
        "longest_edge": 5120 * 28**2,
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


def make_optimizer(model, learning_rate=1e-5):
    return torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=0.0,
        fused=True,
    )


def run_update(model, processor, optimizer, rows, images, reference=None):
    controls = {processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id}
    summary = []
    probes = []
    kl_means = []
    conditional_kls = []
    torch.cuda.synchronize()
    started = perf_counter()
    replay_errors = []
    for row in rows:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prompt, target = prepare(model, processor, row, images)
            paths = rollout(model, prompt, controls)
        texts = [
            processor.tokenizer.decode(
                [tok for tok, _, _ in path if tok not in controls],
                skip_special_tokens=False,
            )
            for path in paths
        ]
        reward_pairs = [rewards(text, target) for text in texts]
        total = torch.tensor(
            [sum(pair) for pair in reward_pairs], device="cuda", dtype=torch.float32
        )
        advantages = (total - total.mean()) / (total.std() + 1e-4)
        for path, advantage in zip(paths, advantages, strict=True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                policy_dist, targets, mask = replay_distribution(model, prompt, path)
                logp = policy_dist.gather(1, targets).squeeze(1)
            sampled = torch.tensor([a.sampled_logp for a in path], device="cuda")[mask]
            old = logp.detach()
            ratio = (logp - old).exp()
            if reference is None:
                reference_logp = old
            else:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    ref_dist, _, _ = replay_distribution(reference, prompt, path)
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
            (-objective.mean() / (len(rows) * GROUP)).backward()
            replay_errors.append((old - sampled).abs().cpu())
            probes.append((prompt, path, old.cpu()))
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
        print(row["id"], reward_pairs, flush=True)
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
        "mean_abs_logp_change": sum(changes) / len(changes),
        "peak_gpu_GiB": torch.cuda.max_memory_allocated() / 1024**3,
    }
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
