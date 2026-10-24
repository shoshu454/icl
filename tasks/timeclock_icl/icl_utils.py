"""
ICL (in-context-learning) variant of TimeClock for memory/retrieval-based
continual-learning baselines.

Structurally the CUB200/PixmoCount clone (single HF split, drop-first-N to avoid
pool/eval overlap), with scoring swapped to clock-time parsing (hour/minute/both).

Design (identical pool protocol to cub200_icl / pixmocount_icl):
  - Single usable split (train). Pool = shuffle(seed).select(range(N)) -> first N items.
    icl_process_docs drops those same first N (same shuffle+seed) from the eval set.
  - Pool loaded INDEPENDENTLY on first access inside each doc_to_* (not via a global from
    process_docs), to avoid silent 0-shot degeneration in worker contexts.
  - parse_time / timeclock_process_results COPIED VERBATIM so ICL scoring == zero-shot.
  - In-context example target = doc['answer'] verbatim (the full "H:MM" string), matching
    the original doc_to_target (which returns answer unchanged, unlike pixmocount which
    extracts a number).

Env vars:
    ICL_NUM_SHOTS     int  number of in-context examples (default 4)
    ICL_POOL_SEED     int  seed for selecting the pool (default 0)
    ICL_DATASET_PATH  str  HF dataset id (default AvaXiao/clockreading-time)
    ICL_DATASET_SPLIT str  split to load (default train)
"""

import io
import os
import re
from PIL import Image
from loguru import logger as eval_logger


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------
def _num_shots():
    try:
        return max(0, int(os.environ.get("ICL_NUM_SHOTS", "4")))
    except ValueError:
        eval_logger.warning("[icl] invalid ICL_NUM_SHOTS; defaulting to 4")
        return 4


def _pool_seed():
    try:
        return int(os.environ.get("ICL_POOL_SEED", "0"))
    except ValueError:
        eval_logger.warning("[icl] invalid ICL_POOL_SEED; defaulting to 0")
        return 0


def _dataset_path():
    return os.environ.get("ICL_DATASET_PATH", "AvaXiao/clockreading-time")


def _dataset_split():
    return os.environ.get("ICL_DATASET_SPLIT", "train")


def _to_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and "bytes" in image:
        return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    raise ValueError(f"Unexpected image format: {type(image)}")


def _strip_image_tags(text):
    return text.replace("\n<image>", "").replace("<image>", "")


# ---------------------------------------------------------------------------
# Scoring -- COPIED VERBATIM from original timeclock/utils.py
# ---------------------------------------------------------------------------
def icl_doc_to_target(doc):
    return doc["answer"]


def parse_time(text):
    """Parse hour and minute from text."""
    text = text.lower().replace('.', '')

    time_match = re.search(r'(\d{1,2}):(\d{1,2})', text)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        return hour, minute

    time_match = re.search(r'time (?:is|displayed|shown).*?(\d{1,2}):(\d{1,2})', text)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        return hour, minute

    numbers = re.findall(r'\d+', text)
    if len(numbers) >= 2:
        hour = int(numbers[0])
        minute = int(numbers[1])
        return hour, minute

    return None, None


def icl_process_results(doc, results):
    """COPIED VERBATIM from timeclock_process_results (fn renamed icl_ for yaml wiring)."""
    pred_text = results[0].strip()
    pred_hour, pred_minute = parse_time(pred_text)

    true_text = doc["answer"]
    true_hour, true_minute = parse_time(true_text)

    if pred_hour is None or true_hour is None:
        return {
            "hour_accuracy": 0,
            "minute_accuracy": 0,
            "both_accuracy": 0
        }

    hour_correct = (pred_hour == true_hour)
    minute_correct = abs(pred_minute - true_minute) <= 1 or abs(pred_minute - true_minute) == 59
    both_correct = hour_correct and minute_correct

    return {
        "hour_accuracy": int(hour_correct),
        "minute_accuracy": int(minute_correct),
        "both_accuracy": int(both_correct)
    }


# ---------------------------------------------------------------------------
# Lazy, self-contained example pool (identical protocol to cub200_icl).
# ---------------------------------------------------------------------------
_POOL = None        # list of {"image": PIL, "question": str, "answer": str}


def _ensure_pool():
    global _POOL
    if _POOL is not None:
        return _POOL

    n = _num_shots()
    seed = _pool_seed()

    if n == 0:
        _POOL = []
        eval_logger.info("[icl] ICL_NUM_SHOTS=0; running as zero-shot")
        return _POOL

    from datasets import load_dataset

    path = _dataset_path()
    split = _dataset_split()
    eval_logger.info(
        f"[icl] loading pool from {path}:{split} (shots={n}, seed={seed})"
    )
    ds = load_dataset(path, split=split)
    shuffled = ds.shuffle(seed=seed)
    pool_slice = shuffled.select(range(min(n, len(shuffled))))  # reserve FIRST n

    pool = []
    for ex in pool_slice:
        pool.append(
            {
                "image": _to_pil(ex["image"]),
                "question": _strip_image_tags(ex["question"]),
                "answer": ex["answer"],  # full "H:MM" string, used verbatim in example
            }
        )
    _POOL = pool
    eval_logger.info(f"[icl] pool ready with {len(pool)} shots")
    return _POOL


def icl_process_docs(dataset):
    """Drop the first N (post-shuffle, same seed) items so eval never overlaps the pool."""
    n = _num_shots()
    seed = _pool_seed()
    if n == 0:
        return dataset
    shuffled = dataset.shuffle(seed=seed)
    total = len(shuffled)
    if n >= total:
        eval_logger.warning(
            f"[icl] ICL_NUM_SHOTS={n} >= dataset size {total}; keeping full set"
        )
        return shuffled
    test_set = shuffled.select(range(n, total))
    eval_logger.info(
        f"[icl] eval set size={len(test_set)} (dropped {n} pool items from {total})"
    )
    return test_set


def icl_doc_to_visual(doc):
    """Return [pool_img_1, ..., pool_img_N, test_img]."""
    pool = _ensure_pool()
    images = [ex["image"] for ex in pool]
    images.append(_to_pil(doc["image"]))
    return images

def icl_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    pool = _ensure_pool()
    parts = []
    for ex in pool:
        parts.append(f"<image>{ex['question']}\nAnswer: {ex['answer']}")
    test_q = _strip_image_tags(doc["question"])
    parts.append(f"<image>{test_q}{post_prompt}")
    return f"{pre_prompt}" + "\n".join(parts)