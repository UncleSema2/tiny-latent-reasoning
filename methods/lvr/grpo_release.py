"""Data, reward and token masking for the released LVR GRPO recipe."""

import hashlib
import json
import random
import re
from functools import lru_cache

MAX_PROMPT = 4096
MAX_COMPLETION = 512
MAX_IMAGE_TOKENS = 2560
TOP_K = 50
MICROBATCH = 4
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.1
LEARNING_RATE = 5e-6

_ANSWER = re.compile(r"<answer>(.*?)</answer>")
_FORMAT = re.compile(r"^<\|lvr_start\|>.*?<\|lvr_end\|>\s*<answer>.*?</answer>$")


def batches(annotations, steps, expected_digest):
    raw = annotations.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("Unexpected ViRL annotation digest")
    rows = json.loads(raw)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate ViRL question IDs")
    for row in rows:
        if not isinstance(row.get("image"), str) or [
            message["from"] for message in row["conversations"]
        ] != ["human", "gpt"]:
            raise ValueError(f"Unsupported ViRL row {row['id']}")
    random.Random(42).shuffle(rows)
    stop = 4 * steps
    if stop > len(rows):
        raise ValueError("Insufficient ViRL rows for the requested number of updates")
    return [rows[i : i + 4] for i in range(0, stop, 4)]


def answer_text(text):
    match = _ANSWER.search(text)
    return (match[1] if match else text).strip()


@lru_cache(maxsize=1024)
def symbolic_answer(text):
    from math_verify import parse
    from math_verify.errors import TimeoutException

    try:
        return parse(text)
    except (Exception, TimeoutException):  # noqa: BLE001 - symbolic parsing is optional
        return []


def symbolic_match(text, target):
    from math_verify import verify
    from math_verify.errors import TimeoutException

    try:
        return verify(symbolic_answer(text), symbolic_answer(target))
    except (Exception, TimeoutException):  # noqa: BLE001 - use the string fallback
        return False


def rewards(text, target):
    correct = symbolic_match(text, target) or answer_text(text) == answer_text(target)
    return int(correct), int(_FORMAT.match(text) is not None)


def completion_mask(path, start_id, end_id, eos_id):
    """Keep the closing marker and text through the first EOS."""
    inside = False
    stopped = False
    mask = []
    for action in path:
        if action.token in (start_id, end_id):
            inside = action.token == start_id
        mask.append(not inside and not stopped)
        stopped |= action.token == eos_id
    return mask


def token_denominators(masks):
    """Count scored tokens per four-completion microbatch for BNPO."""
    counts = []
    for start in range(0, len(masks), MICROBATCH):
        chunk = masks[start : start + MICROBATCH]
        total = max(1, sum(sum(mask) for mask in chunk))
        counts.extend([total] * len(chunk))
    return counts
