#!/usr/bin/env bash
set -euo pipefail

# Run this on the cloud machine from the project directory:
#   bash run_dpo_light_sweep.sh
#
# It trains three lightweight DPO variants and evaluates each with
# CoT self-consistency. Pick the variant with the highest printed accuracy.

if [ ! -f "dpo_train_cot.json" ]; then
  python dpo_prepare_data.py --mode cot --output dpo_train_cot.json
fi

run_one() {
  local name="$1"
  local max_steps="$2"
  local lr="$3"
  local beta="$4"

  echo "============================================================"
  echo "Training ${name}: steps=${max_steps}, lr=${lr}, beta=${beta}"
  echo "============================================================"

  SFT_LORA_PATH=./方案2/final \
  DPO_DATA_PATH=dpo_train_cot.json \
  DPO_OUTPUT_DIR=./output/${name} \
  DPO_FINAL_DIR=./output/${name}/final \
  DPO_MAX_STEPS=${max_steps} \
  DPO_LR=${lr} \
  DPO_BETA=${beta} \
  DPO_LABEL_SMOOTHING=0.05 \
  DPO_BATCH_SIZE=8 \
  DPO_GRAD_ACCUM=4 \
  python dpo_train.py

  echo "============================================================"
  echo "Evaluating ${name}"
  echo "============================================================"

  python eval_answer_accuracy.py \
    --prompt-mode cot \
    --sft-lora ./方案2/final \
    --lora ./output/${name}/final \
    --offset 10000 \
    --limit 1000 \
    --batch-size 16 \
    --num-samples 5 \
    --temperature 0.7 \
    --top-p 0.9 \
    --output-errors ./output/${name}/eval_errors.csv \
    | tee ./output/${name}/eval.log
}

run_one dpo_light_20 20 1e-7 1.0
run_one dpo_light_40 40 1e-7 0.7
run_one dpo_light_60 60 2e-7 0.5

echo "============================================================"
echo "Summary"
echo "============================================================"
grep -H "accuracy=" ./output/dpo_light_*/eval.log || true

echo
echo "Pick the folder with the highest accuracy, then generate a submission like:"
echo "PROMPT_MODE=cot SFT_LORA_PATH=./方案2/final LORA_PATH=./output/dpo_light_40/final NUM_SAMPLES=5 INFER_BATCH_SIZE=16 MAX_NEW_TOKENS=512 OUTPUT_CSV=submit_rlhf_dpo_sc.csv python infer_dpo.py"
echo "python validate_submit.py submit_rlhf_dpo_sc.csv"
