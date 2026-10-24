import io
import os
import re
from PIL import Image
from loguru import logger as eval_logger


def _num_shots():
    return max(0, int(os.environ.get("ICL_NUM_SHOTS", "4")))



def _pool_seed():
    return int(os.environ.get("ICL_POOL_SEED", "0"))



def _dataset_path():
    return os.environ.get("ICL_DATASET_PATH", "Jessemel/pixmocount-test")


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
# Scoring -- COPIED VERBATIM from original pixmocount/utils.py
# ---------------------------------------------------------------------------
def icl_doc_to_target(doc):
    number_match = re.search(r'\d+', doc['answer'])
    if number_match:
        return number_match.group()
    return doc['answer']


def parse_count(text):
    """Parse number from text, handling both digits and word numbers."""
    text = text.lower().replace('.', '')

    word_to_num = {
        'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
        'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
        'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
        'eighteen': '18', 'nineteen': '19', 'twenty': '20'
    }

    for word, num in word_to_num.items():
        text = text.replace(word, num)

    patterns = [
        r'\d+',
        r'there (?:is|are) (\d+)',
        r'(\d+)\s+\w+',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            number = match.group(1) if len(match.groups()) > 0 else match.group(0)
            return int(number)

    return None


def icl_process_results(doc, results):
    """COPIED VERBATIM from pixmocount_process_results (fn renamed icl_ for yaml wiring)."""
    pred_text = results[0].strip()

    pred_number = parse_count(pred_text)
    if pred_number is None:
        return {"pixmocount_accuracy": 0}

    true_answer = doc['answer']
    true_number = parse_count(true_answer)
    if true_number is None:
        return {"pixmocount_accuracy": 0}

    return {"pixmocount_accuracy": int(pred_number == true_number)}


# ---------------------------------------------------------------------------
# Lazy, self-contained example pool (identical protocol to cub200_icl).
# ---------------------------------------------------------------------------
_POOL = None        # list of {"image": PIL, "question": str, "answer_for_example": str}


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
        # example target = same number doc_to_target would extract
        num_match = re.search(r'\d+', ex['answer'])
        ans_for_example = num_match.group() if num_match else ex['answer']
        pool.append(
            {
                "image": _to_pil(ex["image"]),
                "question": _strip_image_tags(ex["question"]),
                "answer_for_example": ans_for_example,
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
        parts.append(f"<image>{ex['question']}\nAnswer: {ex['answer_for_example']}")
    test_q = _strip_image_tags(doc["question"])
    parts.append(f"<image>{test_q}{post_prompt}")
    return f"{pre_prompt}" + "\n".join(parts)