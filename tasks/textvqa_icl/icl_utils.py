"""
ICL (in-context-learning) variant of TextVQA for memory/retrieval-based
continual-learning baselines.

Design (mirrors pathvqa_icl, adapted for TextVQA's VQA-accuracy scoring + multi-GT):
  - Pool of N in-context examples is drawn from the TRAIN split (shuffled, fixed seed,
    first N reserved). Evaluation runs on the VALIDATION split (TextVQA test answers are
    not public). Pool (train) and eval (validation) are disjoint by construction -- no
    drop-first-N hack needed.
  - process_results uses the SHARED framework EvalAIAnswerProcessor (imported, not copied)
    plus the official VQA-accuracy formula, copied verbatim from the original textvqa task
    so ICL scoring is identical to zero-shot.
  - TextVQA has MULTIPLE ground-truth answers per item (doc['answers'], ~10). For the
    in-context example blocks we must pick ONE answer to show; we use the MAJORITY answer
    (most frequent; ties -> first encountered).

Env vars:
  ICL_NUM_SHOTS      number of in-context examples (default 4)
  ICL_POOL_SEED      shuffle seed for selecting the pool (default 0)
  ICL_DATASET_PATH   HF dataset id (default: see _icl_cfg; match _default_template_textvqa_yaml)
  ICL_DATASET_NAME   HF dataset config name (optional)
  ICL_POOL_SPLIT     split to draw the example pool from (default train)

NOTE for the future retrieval method: this baseline uses a GLOBAL fixed pool (all eval
items share the same N examples). The retrieval version must select per-eval-sample
examples dynamically -- make doc_to_visual / doc_to_text key off the current `doc`.
"""

import os
import logging
import statistics
from collections import Counter

import datasets

# SHARED framework util -- safe to import (lives on PYTHONPATH, stable, not task-local).
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

logger = logging.getLogger("textvqa_icl")

import io
from PIL import Image


def _to_pil(image):
    """Match cub200_icl: accept PIL or HF {"bytes": ...} dict."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and "bytes" in image:
        return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
    raise ValueError(f"Unexpected image format: {type(image)}")


def _strip_image_tags(text):
    return text.replace("\n<image>", "").replace("<image>", "")


# ---------------------------------------------------------------------------
# ICL pool management
# ---------------------------------------------------------------------------

_POOL = None  # list of {image, question, answer_for_example}


def _icl_cfg():
    return {
        "num_shots": int(os.environ.get("ICL_NUM_SHOTS", "4")),
        "seed": int(os.environ.get("ICL_POOL_SEED", "0")),
        # IMPORTANT: default must match the dataset_path in _default_template_textvqa_yaml.
        # Override with ICL_DATASET_PATH if the template uses a different id.
        "dataset_path": os.environ.get("ICL_DATASET_PATH", "lmms-lab/textvqa"),
        "dataset_name": os.environ.get("ICL_DATASET_NAME", "") or None,
        "pool_split": os.environ.get("ICL_POOL_SPLIT", "train"),
    }


def _majority_answer(answers):
    """Pick the single answer to show in an in-context example block.

    TextVQA gives ~10 GT answers per item. We show the most frequent one
    (ties broken by first occurrence). Falls back gracefully on odd inputs.
    """
    if not answers:
        return ""
    if isinstance(answers, str):
        return answers
    counts = Counter(answers)
    # Counter.most_common preserves insertion order for ties in CPython 3.7+
    return counts.most_common(1)[0][0]


def _ensure_pool():
    """Lazily load + cache the in-context example pool.

    Loaded independently on first access inside EACH doc_to_* (not handed off via a global
    from process_docs), because lmms-eval worker contexts may see a foreign global as None
    -> would silently degenerate to 0-shot.
    """
    global _POOL
    if _POOL is not None:
        return _POOL

    cfg = _icl_cfg()
    n = cfg["num_shots"]
    logger.info(
        f"[icl] loading pool from {cfg['dataset_path']}"
        f"{(':' + cfg['dataset_name']) if cfg['dataset_name'] else ''}"
        f":{cfg['pool_split']} (shots={n}, seed={cfg['seed']})"
    )
    if cfg["dataset_name"]:
        ds = datasets.load_dataset(cfg["dataset_path"], cfg["dataset_name"], split=cfg["pool_split"])
    else:
        ds = datasets.load_dataset(cfg["dataset_path"], split=cfg["pool_split"])
    shuffled = ds.shuffle(seed=cfg["seed"])
    pool_slice = shuffled.select(range(min(n, len(shuffled))))  # reserve FIRST n
    pool = []
    for ex in pool_slice:
        pool.append(
            {
                "image": _to_pil(ex["image"]),
                "question": _strip_image_tags(ex["question"]),
                "answer_for_example": _majority_answer(ex.get("answers")),
            }
        )
    _POOL = pool
    logger.info(f"[icl] pool ready with {len(_POOL)} shots")
    return _POOL


# ---------------------------------------------------------------------------
# ICL doc_to_* hooks
# ---------------------------------------------------------------------------

def icl_process_docs(docs):
    """Eval set = the validation split, untouched. Pool comes from train -> disjoint."""
    pool = _ensure_pool()
    logger.info(
        f"[icl] eval set size={len(docs)} (pool of {len(pool)} drawn from a separate split)"
    )
    return docs


def icl_doc_to_visual(doc):
    """[pool_img_1..N, eval_img] as RGB. Count MUST equal #<image> tags in doc_to_text (N+1)."""
    pool = _ensure_pool()
    visuals = [ex["image"] for ex in pool]  # already RGB PIL from _to_pil
    visuals.append(_to_pil(doc["image"]))
    return visuals


def icl_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Build N example blocks + the final eval question.

    Mirrors the original textvqa_doc_to_text formatting EXACTLY: question is
    `.capitalize()`-ed, optional OCR ref, pre/post prompts from kwargs. Each example
    block appends `\nAnswer: {majority_gt}`; the final block leaves the answer open.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    use_ocr = lmms_eval_specific_kwargs.get("ocr", False)

    def _ocr_ref(d):
        if use_ocr and "ocr_tokens" in d and d["ocr_tokens"]:
            return f"\nReference OCR token: {', '.join(d['ocr_tokens'])}"
        return ""

    pool = _ensure_pool()
    blocks = []
    for ex in pool:
        q = _strip_image_tags(ex["question"]).capitalize()
        # ocr ref omitted for pool examples (pool dicts don't carry ocr_tokens); the
        # original task only ever shows ocr for the queried doc anyway when ocr=True.
        blocks.append(f"<image>{pre_prompt}{q}{post_prompt}\nAnswer: {ex['answer_for_example']}")

    q = _strip_image_tags(doc["question"]).capitalize()
    blocks.append(f"<image>{pre_prompt}{q}{_ocr_ref(doc)}{post_prompt}")
    return "\n".join(blocks)


def icl_doc_to_target(doc):
    """Target for logging; the real scoring is in icl_process_results.
    Mirrors the original yaml's doc_to_target: 'answer'. We surface the majority answer."""
    return _majority_answer(doc.get("answers"))


# ===========================================================================
# Scoring -- VQA accuracy, copied VERBATIM from original textvqa_process_results
# (only the fn name is prefixed icl_). Uses the shared EvalAIAnswerProcessor.
# ===========================================================================

def icl_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0
    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []
        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])
        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        accuracy = statistics.mean(gtAcc)
    return {
        "exact_match": accuracy,
        "submission": {
            "question_id": doc["question_id"],
            "answer": resAns,
        },
    }