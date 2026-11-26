"""Decode the training subset of an overfit run and report each answer and latent span."""

import argparse
import json
import os
import random
from pathlib import Path

from transformers import AutoProcessor

from .__main__ import TOKENS, format_prompt
from .data import load_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--lvr-steps", type=int, choices=(4, 8, 16), default=8)
    args = parser.parse_args()
    recipe = json.loads((args.checkpoint / "sft_recipe.json").read_text())
    if not recipe["overfit"]:
        parser.error("The checkpoint must come from a fixed --overfit subset")
    rows = random.Random(recipe["seed"]).sample(load_rows(args.data), recipe["overfit"])
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    os.environ["SGLANG_EXTERNAL_MODEL_PACKAGE"] = "methods.lvr"
    from sglang import Engine

    override = {"architectures": ["Qwen2_5_VLForConditionalGeneration"]}
    if not recipe["answer_only"]:
        override["lvr_steps"] = args.lvr_steps
    engine = Engine(
        model_path=str(args.checkpoint.resolve()),
        dtype="bfloat16",
        max_running_requests=min(len(rows), 4),
        mem_fraction_static=0.55,
        mm_processor_worker_num=1,
        disable_radix_cache=True,
        json_model_override_args=json.dumps(override),
        mm_process_config={
            "image": {
                "size": {
                    key: processor.image_processor.size[key]
                    for key in ("shortest_edge", "longest_edge")
                }
            }
        },
    )
    try:
        outputs = engine.generate(
            prompt=[format_prompt(processor, row) for row in rows],
            image_data=[[str(args.data / "images" / row["image"][0])] for row in rows],
            sampling_params={
                "temperature": 0,
                "max_new_tokens": 96,
                "skip_special_tokens": False,
            },
        )
        for row, output in zip(rows, outputs, strict=True):
            target = row["conversations"][1]["value"].replace("<lvr>", "").strip()
            prediction = output["text"]
            print(
                json.dumps(
                    {
                        "question_id": row["question_id"],
                        "target": target,
                        "prediction": prediction,
                        "latent_starts": prediction.count(TOKENS[0]),
                        "target_suffix_match": prediction.rstrip().endswith(target),
                    }
                ),
                flush=True,
            )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
