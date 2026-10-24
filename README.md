# Few-shot in-context learning for LLaVA-OneVision

In-context-learning (ICL) versions of five multimodal tasks, written as
[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) task plugins. The
model is not trained. Each prompt contains `N` worked examples (image +
question + answer) and then the test question, so with the default of 20 shots
every request carries 21 images.

| Task | Dataset (HF) | Eval split | Example pool | Metric |
|---|---|---|---|---|
| `cub200_icl` | `Jessemel/cub200-test` | train | first N of the eval split, removed from eval | acc (option letter) |
| `pixmocount_icl` | `Jessemel/pixmocount-test` | train | first N of the eval split, removed from eval | count accuracy |
| `timeclock_icl` | `AvaXiao/clockreading-time` | train | first N of the eval split, removed from eval | hour / minute / both accuracy |
| `pathvqa_icl` | `flaviagiammarino/path-vqa` | test | train split | exact match, F1, BLEU-1..3, yes/no accuracy |
| `textvqa_icl` | `lmms-lab/textvqa` | validation | train split (majority answer) | VQA exact match |

## Layout

```
tasks/<task>_icl/
    <task>_icl.yaml   # lmms-eval task config: dataset, split, generation, metrics, prompts
    icl_utils.py      # the ICL logic (hooks referenced from the yaml)
run_icl.sh            # launcher for one task
```

Each `icl_utils.py` implements the same hooks:

- `_ensure_pool()` loads the dataset, runs `shuffle(seed=ICL_POOL_SEED)` and
  keeps the first `ICL_NUM_SHOTS` rows as the example pool. The pool is built
  lazily inside each process and cached. It is not passed through
  `process_docs`, because lmms-eval workers can see that global as `None` and
  quietly fall back to 0-shot.
- `icl_process_docs()` builds the eval set. If the pool comes from the eval
  split, the same first N rows are dropped so the two never overlap.
- `icl_doc_to_visual()` returns `[pool_img_1, ..., pool_img_N, test_img]`.
- `icl_doc_to_text()` returns
  `pre_prompt + N × "<image>{q}\nAnswer: {a}" + "<image>{test_q}{post_prompt}"`.
  The number of `<image>` tags must equal the number of images.
- `icl_process_results()` (and the metric helpers) are copied unchanged from
  each task's zero-shot lmms-eval utils, so answers are parsed the same way as
  in zero-shot.

Every task except cub200 lists `"\nAnswer:"` in its generation `until` stops,
so the model doesn't keep going and write more examples of its own. cub200
relies on `max_new_tokens: 16` and its letter parser instead.

## Running

Requirements: lmms-eval, [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)
(provides the `llava_onevision` model), `datasets`, `Pillow`, `loguru`.

```bash
export LMMS_EVAL_SRC=/path/to/lmms-eval
bash run_icl.sh cub200_icl                          # 20 shots, seed 0, 1 GPU
ICL_NUM_SHOTS=5 ICL_POOL_SEED=1 NUM_GPUS=2 bash run_icl.sh pixmocount_icl
```

| Env var | Default | Meaning |
|---|---|---|
| `ICL_NUM_SHOTS` | 20 (in `run_icl.sh`; 4 if unset in Python) | examples per prompt, 0 = zero-shot |
| `ICL_POOL_SEED` | 0 | shuffle seed used to pick the pool |
| `ICL_DATASET_PATH` | per task | override the HF dataset id |
| `ICL_DATASET_SPLIT` / `ICL_POOL_SPLIT` | per task | split to draw the pool from |

The original runs used `ICL_POOL_SEED=1` for pixmocount and timeclock and `0`
for the other three.

Results are written to `results/<task>/` as a `*_results.json` plus a
`*_samples_*.jsonl` containing every prompt and prediction.

## Notes

- Every test question gets the same N examples. The pool is drawn once per
  seed, not re-sampled for each question.
- For cub200, pixmocount and timeclock, the eval set shrinks by N, so scores are
  over slightly fewer documents than the zero-shot task.
- In `clockreading-time`, the train split is the same data as the test split.
- pathvqa and textvqa repeat their pre/post prompt around every example.
  cub200, pixmocount and timeclock put the pre-prompt once at the top.
- Scores vary noticeably with `ICL_POOL_SEED`. Report several seeds.
