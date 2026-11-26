"""Train the LUT teacher on Visual-CoT with a fixed pilot batch or one full epoch."""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from math import ceil
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

BASE = "Qwen/Qwen2.5-VL-7B-Instruct"
REVISION = "cc594898137f460bfe9f0759e9844b3ce807cfb5"
BOUNDARIES = ("<latent>", "</latent>")


def prepare_example(processor, row, images, device):
    with Image.open(images / row["image"]) as image:
        content = [
            {"type": "image", "image": image.convert("RGB")},
            {"type": "text", "text": row["question"]},
        ]
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
    answer = processor.tokenizer(
        row["answer"], add_special_tokens=False, return_tensors="pt"
    ).input_ids
    if not answer.numel():
        raise ValueError(f"Empty tokenized answer for {row['id']}")
    return prompt.to(device), answer.to(device)


def teacher_mask(image, latent):
    """Boolean SDPA mask for a single unpadded image/question/latents/answer."""
    causal = torch.ones(
        image.numel(), image.numel(), dtype=torch.bool, device=image.device
    ).tril()
    return (causal & (~image[None, :] | (image | latent)[:, None]))[None, None]


class Teacher(torch.nn.Module):
    def __init__(self, base, boundaries, steps=8):
        super().__init__()
        self.base = base
        self.boundaries = boundaries
        self.steps = steps

    def forward(self, prompt, answer, *, return_trajectory=False):
        base = self.base
        decoder = base.model.language_model
        start, end = self.boundaries
        prefix = prompt["input_ids"]
        prefix_len = prefix.size(1) + 1
        # Reserve one embedding position for each recurrent latent.
        ids = torch.cat(
            [prefix, prefix.new_tensor([[start] * (1 + self.steps) + [end]]), answer],
            dim=1,
        )
        image = ids[0] == base.config.image_token_id
        positions, _ = base.model.get_rope_index(
            ids, image[None].int(), prompt["image_grid_thw"]
        )
        indices = torch.arange(ids.size(1), device=ids.device)
        latent = (indices >= prefix_len - 1) & (indices < prefix_len + self.steps)
        mask = teacher_mask(image, latent)
        embeddings = base.get_input_embeddings()(ids)
        vision = base.get_image_features(
            prompt["pixel_values"], prompt["image_grid_thw"]
        ).pooler_output
        embeddings = embeddings.masked_scatter(
            image[None, :, None], torch.cat(vision).to(embeddings.dtype)
        )

        def decode(inputs, offset, cache):
            stop = offset + inputs.size(1)
            return decoder(
                inputs_embeds=inputs,
                position_ids=positions[..., offset:stop],
                attention_mask={"full_attention": mask[..., offset:stop, :stop]},
                past_key_values=cache,
                use_cache=True,
            )

        result = decode(embeddings[:, :prefix_len], 0, None)
        states = [result.last_hidden_state[:, -1:]]
        for offset in range(prefix_len, prefix_len + self.steps - 1):
            result = decode(states[-1], offset, result.past_key_values)
            states.append(result.last_hidden_state)
        tail = torch.cat(
            [states[-1], embeddings[:, prefix_len + self.steps : -1]], dim=1
        )
        result = decode(tail, prefix_len + self.steps - 1, result.past_key_values)
        # The closing boundary predicts the first answer token.
        logits = base.lm_head(result.last_hidden_state[:, 1:]).float()
        ce = F.cross_entropy(logits.flatten(0, 1), answer.flatten())
        trajectory = torch.cat(states, dim=1)
        if return_trajectory:
            return logits, trajectory
        unit = F.normalize(trajectory.float(), dim=-1)
        gram = unit @ unit.transpose(-1, -2)
        diagonal = torch.eye(self.steps, dtype=torch.bool, device=ids.device)
        return ce, gram[:, ~diagonal].square().mean()


def epoch_batches(rows, rank, world, completed, processor, images):
    """Prefetch one CPU example per rank while the current update runs."""

    def batches():
        for offset in range(completed * world, len(rows), world):
            active = min(world, len(rows) - offset)
            # In the last partial batch all ranks join DDP, with zero loss on padding.
            index = offset + (rank if rank < active else 0)
            yield rows[index], world / active if rank < active else 0.0

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = None
        for row, weight in batches():
            future = pool.submit(prepare_example, processor, row, images, "cpu")
            if pending is not None:
                yield *pending[0].result(), pending[1]
            pending = (future, weight)
        if pending is not None:
            yield *pending[0].result(), pending[1]


def save_training_state(base, processor, optim, output, step, recipe, rank):
    target = output / f"checkpoint-{step}"
    stage = output / f".checkpoint-{step}"
    if rank == 0:
        stage.mkdir()
    dist.barrier()
    # ZeRO owns a deterministic optimizer partition on each rank.
    torch.save(optim.optim.state_dict(), stage / f"optimizer-{rank}.pt")
    if rank == 0:
        base.save_pretrained(stage)
        processor.save_pretrained(stage)
        (stage / "training_state.json").write_text(
            json.dumps({"step": step, "recipe": recipe})
        )
    dist.barrier()
    if rank == 0:
        stage.rename(target)
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", default=BASE)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--updates", type=int, help="updates on the fixed pilot batch (default 24)"
    )
    mode.add_argument(
        "--full-epoch", action="store_true", help="consume every JSONL row once"
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="full-epoch checkpoint to continue in a fresh output directory",
    )
    args = parser.parse_args()
    if args.resume_from and not args.full_epoch:
        parser.error("--resume-from requires --full-epoch")
    if args.updates is not None and args.updates < 1:
        parser.error("--updates must be positive")
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()
    set_seed(42)
    with args.data.open() as source:
        rows = list(
            map(json.loads, source if args.full_epoch else islice(source, world))
        )
    if not rows:
        raise ValueError("Empty training file")
    if not args.full_epoch and len(rows) < world:
        raise ValueError("The pilot needs one example per rank")
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    processor = AutoProcessor.from_pretrained(args.base, revision=REVISION)
    processor.image_processor.size = {
        "shortest_edge": 128 * 28**2,
        "longest_edge": 512 * 28**2,
    }
    processor.tokenizer.add_tokens(list(BOUNDARIES), special_tokens=True)
    boundaries = processor.tokenizer.convert_tokens_to_ids(list(BOUNDARIES))
    base, info = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.resume_from or args.base,
        revision=REVISION,
        dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    if any(info.values()):
        raise ValueError(f"Incomplete base checkpoint: {info}")
    if max(boundaries) >= base.get_input_embeddings().num_embeddings:
        base.resize_token_embeddings(len(processor.tokenizer))
    base.model.visual.requires_grad_(False)
    model = DistributedDataParallel(
        Teacher(base, boundaries).train(),
        device_ids=[device.index],
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    )
    optim = ZeroRedundancyOptimizer(
        [p for p in model.parameters() if p.requires_grad],
        optimizer_class=torch.optim.AdamW,
        lr=1e-5,
        fused=True,
    )
    updates = ceil(len(rows) / world) if args.full_epoch else (args.updates or 24)
    recipe = {
        "base": BASE,
        "revision": REVISION,
        "world_size": world,
        "updates": updates,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "latent_steps": 8,
        "decorrelation_weight": 0.1,
        "seed": 42,
        "learning_rate": 1e-5,
    }
    completed = 0
    if args.full_epoch:
        recipe.update(full_epoch=True, examples=len(rows), max_image_tokens=512)
        if args.resume_from:
            state = json.loads((args.resume_from / "training_state.json").read_text())
            if state["recipe"] != recipe or not 0 < state["step"] < updates:
                raise ValueError("Checkpoint does not match this training recipe")
            completed = state["step"]
            optim.optim.load_state_dict(
                torch.load(
                    args.resume_from / f"optimizer-{rank}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
        batches = epoch_batches(rows, rank, world, completed, processor, args.images)
    else:
        recipe["ids"] = [row["id"] for row in rows]
        prompt, answer = prepare_example(processor, rows[rank], args.images, device)
        batches = ((prompt, answer, 1.0) for _ in range(updates))
    if rank == 0:
        (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2))
    for step, (prompt, answer, weight) in enumerate(batches, completed + 1):
        optim.zero_grad(set_to_none=True)
        ce, decorrelation = model(prompt.to(device), answer.to(device))
        loss = (ce + 0.1 * decorrelation) * weight
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optim.step()
        stats = torch.stack([ce.detach(), decorrelation.detach()]) * weight
        dist.all_reduce(stats, op=dist.ReduceOp.AVG)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        "ce": stats[0].item(),
                        "decorrelation": stats[1].item(),
                    }
                ),
                flush=True,
            )
        if args.full_epoch and step % 1000 == 0 and step < updates:
            save_training_state(base, processor, optim, args.output, step, recipe, rank)
    if rank == 0:
        base.save_pretrained(args.output)
        processor.save_pretrained(args.output)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
