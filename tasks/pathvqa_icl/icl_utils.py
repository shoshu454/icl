"""
ICL (in-context-learning) variant of the PathVQA task for memory/retrieval-based
continual-learning baselines.

Design (mirrors cub200_icl, adapted for PathVQA's open-ended answers + train/test split):
  - Pool of N in-context examples is drawn from the TRAIN split (shuffled with a fixed
    seed, first N items reserved). Evaluation runs on the TEST split.
  - Because pool comes from `train` and eval runs on `test`, the two are disjoint by
    construction -- NO drop-first-N hack is needed (unlike cub200_icl, which lives on a
    single split). This is the task-aware random baseline: a fixed set of N random
    training examples shared by every test query.
  - process_results / doc_to_target and ALL metric helpers are COPIED VERBATIM from the
    original pathvqa task so ICL answer-matching is byte-identical to zero-shot.

Env vars:
  ICL_NUM_SHOTS      number of in-context examples (default 4)
  ICL_POOL_SEED      shuffle seed for selecting the pool (default 0)
  ICL_DATASET_PATH   HF dataset id (default flaviagiammarino/path-vqa)
  ICL_POOL_SPLIT     split to draw the example pool from (default train)

NOTE for the future retrieval method: this baseline uses a GLOBAL fixed pool (all test
items share the same N examples). The retrieval version must select per-test-sample
examples dynamically -- i.e. make doc_to_visual / doc_to_text key off the current `doc`
rather than the module-global pool. The structure here is set up to extend there.
"""

import os
import re
import math
import logging
from collections import defaultdict

import datasets

logger = logging.getLogger("pathvqa_icl")

# ---------------------------------------------------------------------------
# ICL pool management
# ---------------------------------------------------------------------------

_POOL = None  # module-global cache: list of dicts with keys {image, question, answer}


def _icl_cfg():
    return {
        "num_shots": int(os.environ.get("ICL_NUM_SHOTS", "4")),
        "seed": int(os.environ.get("ICL_POOL_SEED", "0")),
        "dataset_path": os.environ.get("ICL_DATASET_PATH", "flaviagiammarino/path-vqa"),
        "pool_split": os.environ.get("ICL_POOL_SPLIT", "train"),
    }


def _ensure_pool():
    """Lazily load + cache the in-context example pool.

    Loaded independently on first access inside EACH doc_to_* function (NOT handed off
    from process_docs via a global), because lmms-eval runs doc_to_* in worker contexts
    where a global set elsewhere may be None -> would silently degenerate to 0-shot.
    """
    global _POOL
    if _POOL is not None:
        return _POOL

    cfg = _icl_cfg()
    n = cfg["num_shots"]
    logger.info(
        f"[icl] loading pool from {cfg['dataset_path']}:{cfg['pool_split']} "
        f"(shots={n}, seed={cfg['seed']})"
    )
    ds = datasets.load_dataset(cfg["dataset_path"], split=cfg["pool_split"])
    ds = ds.shuffle(seed=cfg["seed"])
    # reserve the FIRST n items as the pool
    pool = []
    for i in range(min(n, len(ds))):
        item = ds[i]
        pool.append(
            {
                "image": item["image"],
                "question": item["question"],
                "answer": item["answer"],
            }
        )
    _POOL = pool
    logger.info(f"[icl] pool ready with {len(_POOL)} shots")
    return _POOL


# ---------------------------------------------------------------------------
# ICL doc_to_* hooks
# ---------------------------------------------------------------------------

def icl_process_docs(docs):
    """Eval set = the TEST split, untouched.

    Pool comes from a DIFFERENT split (train), so test items never overlap the pool --
    no drop-first-N filtering needed. We still touch the pool here so the size is logged
    once up front, then return docs unchanged.
    """
    pool = _ensure_pool()
    logger.info(
        f"[icl] eval set size={len(docs)} (pool of {len(pool)} drawn from a separate split)"
    )
    return docs


def icl_doc_to_visual(doc):
    """Return [pool_img_1, ..., pool_img_N, test_img] as RGB.

    The number of images here MUST equal the number of <image> tags emitted by
    icl_doc_to_text (= N + 1), or the vision/text alignment breaks.
    """
    pool = _ensure_pool()
    visuals = [ex["image"].convert("RGB") for ex in pool]
    visuals.append(doc["image"].convert("RGB"))
    return visuals


def icl_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    """Build N example blocks + the final test question.

    Each example block:  <image>{pre}{question}{post}\nAnswer: {answer}
    Final test block:     <image>{pre}{question}{post}

    pre/post prompts are taken from the SAME lmms_eval_specific_kwargs as the original
    pathvqa task, so the instruction wording the model sees is identical to zero-shot.
    """
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")

    pool = _ensure_pool()
    blocks = []
    for ex in pool:
        blocks.append(
            f"<image>{pre_prompt}{ex['question']}{post_prompt}\nAnswer: {ex['answer']}"
        )
    # final test query -- no answer, model completes it
    blocks.append(f"<image>{pre_prompt}{doc['question']}{post_prompt}")
    return "\n".join(blocks)


def icl_doc_to_target(doc):
    """COPIED VERBATIM from original pathvqa utils.pathvqa_doc_to_target."""
    return doc["answer"]


# ===========================================================================
# Everything below is COPIED VERBATIM from the original pathvqa/utils.py so the
# ICL task scores answers byte-identically to the zero-shot task. Do not edit.
# ===========================================================================

def is_yes_no_question(question):
    question = question.lower().strip()
    binary_prefixes = [
        'is ', 'are ', 'was ', 'were ',
        'does ', 'do ', 'did ',
        'can ', 'could ', 'would ', 'should ',
        'has ', 'have ', 'had ',
        'will ', 'shall ',
        'am i ', 'are there ', 'is there ', 'are these ', 'is this '
    ]
    return any(question.startswith(prefix) for prefix in binary_prefixes)


def split_sentence(sentence, n):
    words = defaultdict(int)
    tmp_sentence = re.sub("[^a-zA-Z ]", "", sentence)
    tmp_sentence = tmp_sentence.lower()
    tmp_sentence = tmp_sentence.strip().split()
    length = len(tmp_sentence)
    for i in range(length - n + 1):
        tmp_words = " ".join(tmp_sentence[i: i + n])
        if tmp_words:
            words[tmp_words] += 1
    return words


def brevity_penalty(candidate, references):
    c = len(candidate)
    ref_lens = (len(reference) for reference in references)
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))
    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)


def modified_precision(candidate, references, n):
    max_frequency = defaultdict(int)
    min_frequency = defaultdict(int)
    candidate_words = split_sentence(candidate, n)
    for reference in references:
        reference_words = split_sentence(reference, n)
        for word in candidate_words:
            max_frequency[word] = max(max_frequency[word], reference_words[word])
    for word in candidate_words:
        min_frequency[word] = min(max_frequency[word], candidate_words[word])
    if sum(candidate_words.values()) == 0:
        return 0
    P = sum(min_frequency.values()) / sum(candidate_words.values())
    return P


def calculate_bleu(weights, pn, n, bp):
    sum_wlogp = 0
    for i in range(n):
        if pn[i] != 0:
            sum_wlogp += float(weights[i]) * math.log(pn[i])
    bleu_result = bp * math.exp(sum_wlogp)
    return bleu_result


def bleu(candidate, references, n, weights):
    pn = []
    bp = brevity_penalty(candidate, references)
    for i in range(n):
        pn.append(modified_precision(candidate, references, i + 1))
    if len(weights) > len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(weights[i])
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return bleu_result
    elif len(weights) < len(pn):
        tmp_weights = []
        for i in range(len(pn)):
            tmp_weights.append(0)
        for i in range(len(weights)):
            tmp_weights[i] = weights[i]
        bleu_result = calculate_bleu(tmp_weights, pn, n, bp)
        return bleu_result
    else:
        bleu_result = calculate_bleu(weights, pn, n, bp)
        return bleu_result


def calculate_exactmatch(candidate, reference):
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    count = 0
    total = 0
    for word in reference_words:
        if word in candidate_words:
            count += 1
    for word in candidate_words:
        total += candidate_words[word]
    if total == 0:
        return 0
    else:
        return count / total


def calculate_f1score(candidate, reference):
    candidate_words = split_sentence(candidate, 1)
    reference_words = split_sentence(reference, 1)
    word_set = set()
    for word in candidate_words:
        word_set.add(word)
    for word in reference_words:
        word_set.add(word)
    tp = 0
    fp = 0
    fn = 0
    for word in word_set:
        if word in candidate_words and word in reference_words:
            tp += candidate_words[word]
        elif word in candidate_words and word not in reference_words:
            fp += candidate_words[word]
        elif word not in candidate_words and word in reference_words:
            fn += reference_words[word]
    if len(candidate_words) == 0:
        return 0
    elif len(reference_words) == 0:
        return 0
    else:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if tp == 0:
            return 0
        else:
            return 2 * precision * recall / (precision + recall)


def icl_process_results(doc, results):
    """COPIED VERBATIM from original pathvqa_process_results (only the fn name is
    prefixed icl_ to match the yaml wiring)."""
    assert len(results) == 1, f"The result should be a list of length 1, but got {len(results)}."
    prediction = results[0].strip()
    ground_truth = doc['answer']
    question = doc['question']

    is_binary = is_yes_no_question(question)

    try:
        exact_match_score = calculate_exactmatch(prediction, ground_truth)
        f1_score = calculate_f1score(prediction, ground_truth)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        exact_match_score = 0.0
        f1_score = 0.0

    references = [ground_truth]

    try:
        bleu1_score = bleu(prediction, references, 1, [1.0])
        bleu2_score = bleu(prediction, references, 2, [0.5, 0.5])
        bleu3_score = bleu(prediction, references, 3, [1 / 3, 1 / 3, 1 / 3])
    except Exception as e:
        print(f"Error calculating BLEU scores: {e}")
        bleu1_score = 0.0
        bleu2_score = 0.0
        bleu3_score = 0.0

    binary_accuracy = float('nan')
    if is_binary:
        pred_norm = prediction.lower().strip()
        gt_norm = ground_truth.lower().strip()
        binary_accuracy = 1.0 if pred_norm == gt_norm else 0.0

    return {
        "pathvqa_exact_match": exact_match_score,
        "pathvqa_f1": f1_score,
        "pathvqa_bleu1": bleu1_score,
        "pathvqa_bleu2": bleu2_score,
        "pathvqa_bleu3": bleu3_score,
        "pathvqa_binary_accuracy": binary_accuracy,
        "submission": {
            "question_id": doc.get("question_id", "unknown"),
            "answer": prediction,
        }
    }


def icl_aggregate_binary_accuracy(items):
    """COPIED VERBATIM from original pathvqa_aggregate_binary_accuracy."""
    valid_scores = [score for score in items if not (isinstance(score, float) and math.isnan(score))]
    if len(valid_scores) == 0:
        return float('nan')
    return sum(valid_scores) / len(valid_scores)


def icl_aggregate_submissions(items):
    """COPIED VERBATIM from original pathvqa_aggregate_submissions."""
    return items