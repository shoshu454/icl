#!/bin/bash
# Run one ICL task with the base LLaVA-OneVision-7B model (no training).
#
#   bash run_icl.sh cub200_icl
#   ICL_NUM_SHOTS=5 NUM_GPUS=2 bash run_icl.sh pathvqa_icl
#
# Requires LMMS_EVAL_SRC to point at an lmms-eval checkout. The task folders in
# ./tasks are passed with --include, so nothing needs to be copied into it.

set -e

TASK=${1:?usage: run_icl.sh <cub200_icl|pixmocount_icl|timeclock_icl|pathvqa_icl|textvqa_icl>}

: "${LMMS_EVAL_SRC:?set LMMS_EVAL_SRC to your lmms-eval checkout}"
REPO_DIR=$(cd "$(dirname "$0")" && pwd)

export ICL_NUM_SHOTS=${ICL_NUM_SHOTS:-20}
export ICL_POOL_SEED=${ICL_POOL_SEED:-0}
NUM_GPUS=${NUM_GPUS:-1}
MODEL=${MODEL:-lmms-lab/llava-onevision-qwen2-7b-ov}
OUT_DIR=${OUT_DIR:-${REPO_DIR}/results/${TASK}}

export OMP_NUM_THREADS=8
export PYTHONPATH="${LMMS_EVAL_SRC}:${PYTHONPATH}"

accelerate launch --num_processes=${NUM_GPUS} \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=${MODEL},conv_template=qwen_1_5,model_name=llava_qwen,vocab_size=152064 \
    --tasks ${TASK} \
    --include "${REPO_DIR}/tasks" \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${TASK}_${ICL_NUM_SHOTS}shot \
    --output_path ${OUT_DIR}
