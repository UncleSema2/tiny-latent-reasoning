"""Sample Visual-CoT training questions for the LUT teacher and vanilla SFT control."""

import argparse
import hashlib
import json
import random
import tempfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

REVISION = "223d2d8c1146fda2bb918801b8276c587b78b61c"
ANNOTATIONS = "viscot_363k.json"
URL = f"https://huggingface.co/datasets/deepcs233/Visual-CoT/resolve/{REVISION}/{ANNOTATIONS}"
SHA256 = "39c2a901d357fcb1eb377f6046a9124ddb9ac97f1620ca73b87af0340a7f7f8b"
BOX_REQUEST = "Please provide the bounding box coordinate of the region that can help you answer the question better."


def project(row, index):
    turns = row["conversations"]
    question = turns[0]["value"]
    if (
        row["split"] != "train"
        or [t["from"] for t in turns] != ["human", "gpt", "human", "gpt"]
        or not question.startswith("<image>\n")
        or not question.endswith(BOX_REQUEST)
        or turns[2]["value"].strip() != "<image>"
    ):
        raise ValueError(f"Unexpected Visual-CoT dialogue at row {index}")
    question = question.removeprefix("<image>\n").removesuffix(BOX_REQUEST).strip()
    answer = turns[3]["value"].strip()
    original, crop = row["image"]
    if not original.startswith("cot/") or not crop.startswith(original + "###"):
        raise ValueError(f"Unexpected Visual-CoT image pair at row {index}")
    if not question or not answer:
        raise ValueError(f"Empty question or answer at row {index}")
    return {
        "id": f"viscot:{index}",
        "subset": row["dataset"],
        "question_id": row["question_id"],
        "image": original.removeprefix("cot/"),
        "question": question,
        "answer": answer,
    }


def prepare(rows, images, output, count, seed):
    indices = random.Random(seed).sample(range(len(rows)), count)
    selected = Counter()
    checked = set()
    digest = hashlib.sha256()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        staged = Path(tmp) / "train.jsonl"
        with staged.open("wb") as dest:
            for index in indices:
                row = project(rows[index], index)
                image = images / row["image"]
                if image not in checked:
                    if not image.is_file():
                        raise FileNotFoundError(image)
                    checked.add(image)
                line = (json.dumps(row, ensure_ascii=False) + "\n").encode()
                dest.write(line)
                digest.update(line)
                selected[row["subset"]] += 1
        staged.replace(output)
    return {
        "revision": REVISION,
        "annotations": ANNOTATIONS,
        "annotations_sha256": SHA256,
        "population": len(rows),
        "seed": seed,
        "sample_size": count,
        "images": len(checked),
        "subsets": dict(sorted(selected.items())),
        "sample_sha256": digest.hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations", type=Path, help="local copy of viscot_363k.json"
    )
    parser.add_argument(
        "--images", required=True, type=Path, help="extracted cot_image_data directory"
    )
    parser.add_argument("--output", required=True, type=Path, help="destination JSONL")
    parser.add_argument("--count", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if args.annotations and args.annotations.resolve() == args.output.resolve():
        parser.error("--output must differ from --annotations")
    if args.annotations:
        data = args.annotations.read_bytes()
    else:
        with urlopen(URL, timeout=120) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise ValueError(f"Checksum mismatch for {ANNOTATIONS}")
    report = prepare(json.loads(data), args.images, args.output, args.count, args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
