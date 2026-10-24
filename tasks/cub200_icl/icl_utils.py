"""
icl_utils.py
In-context-learning (few-shot) variant of the cub200 task utils.

Key design fix (vs first version):
  The example pool is NO LONGER built inside the lmms-eval `process_docs`
  hook and stashed in a module global. That was unreliable: process_docs and
  doc_to_visual can run at different times / in different worker contexts, so
  the global came back None inside doc_to_visual and the model silently saw
  ZERO example images (degenerating to 0-shot).

  Instead the pool is loaded INDEPENDENTLY from the HuggingFace dataset the
  first time any of the doc_to_* functions needs it, and cached. Because the
  dataset id, split, seed and shot count are all fixed, every process rebuilds
  the identical pool deterministically. No cross-function / cross-process
  state hand-off is required.

No-overlap guarantee:
  We still expose icl_process_docs to DROP the first N (post-shuffle) items
  from the evaluation set, and the pool is exactly those same first N items
  under the same seed -- so test items never include a pool example. But the
  pool's existence no longer DEPENDS on process_docs having run.

Env vars:
    ICL_NUM_SHOTS     int  number of in-context examples (default 4)
    ICL_POOL_SEED     int  seed for selecting the pool (default 0)
    ICL_DATASET_PATH  str  HF dataset id (default Jessemel/cub200-test)
    ICL_DATASET_SPLIT str  split to load (default train)
"""

import io
import os
import re
from PIL import Image
from loguru import logger as eval_logger


# ---------------------------------------------------------------------------
# Copied VERBATIM from the original cub200/utils.py so ICL and zero-shot share
# identical answer-matching logic (otherwise the numbers are not comparable).
# ---------------------------------------------------------------------------
def icl_doc_to_target(doc):
    return doc["answer"]


def icl_process_results(doc, results):
    gold = doc["answer"].lower()

    if isinstance(results, list):
        results = results[0] if len(results) > 0 else ""

    cleaned_pred = str(results).strip()
    letter_match = re.search(r"\b([A-Za-z])[.)]?\b", cleaned_pred)
    if letter_match:
        cleaned_pred = letter_match.group(1).lower()
    else:
        if cleaned_pred and cleaned_pred[0].isalpha():
            cleaned_pred = cleaned_pred[0].lower()
        else:
            cleaned_pred = cleaned_pred.lower()

    is_correct = int(gold == cleaned_pred)
    return {"acc": is_correct, "gold": gold, "pred": cleaned_pred}


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
    return os.environ.get("ICL_DATASET_PATH", "Jessemel/cub200-test")


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
# Lazy, self-contained example pool. Built on first access, cached thereafter.
# Independent of any lmms-eval hook.
# ---------------------------------------------------------------------------
_POOL = None        # list of {"image": PIL, "question": str, "answer": str}


def _ensure_pool():
    """Load the pool directly from HF the first time it's needed, then cache."""
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
    pool_slice = shuffled.select(range(n))

    pool = []
    for ex in pool_slice:
        pool.append(
            {
                "image": _to_pil(ex["image"]),
                "question": _strip_image_tags(ex["question"]),
                "answer": ex["answer"],
            }
        )
    _POOL = pool
    eval_logger.info(f"[icl] pool ready with {len(pool)} shots")
    return _POOL


def icl_process_docs(dataset):
    """
    Drop the first N (post-shuffle, same seed) items so the eval set never
    overlaps with the pool. The pool itself is built independently by
    _ensure_pool(); this hook only removes the held-out examples.
    """
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