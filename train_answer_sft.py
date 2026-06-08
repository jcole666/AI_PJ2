"""
Answer-only LoRA SFT for the math competition.

This stage teaches the model to follow the submission format directly:
final answer only, no reasoning and no units. DPO can then be trained on
top of this adapter.

Recommended on RTX PRO 6000:
  BATCH_SIZE=16 GRAD_ACCUM=2 python train_answer_sft.py
"""
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
DATA_PATH = os.getenv("DATA_PATH", "train.json")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./output/Qwen_answer_sft")
FINAL_DIR = os.getenv("FINAL_DIR", os.path.join(OUTPUT_DIR, "final"))

MAX_LENGTH = int(os.getenv("MAX_LENGTH", "384"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
GRAD_ACCUM = int(os.getenv("GRAD_ACCUM", "4"))
EPOCHS = float(os.getenv("EPOCHS", "3"))
LEARNING_RATE = float(os.getenv("LR", "1e-4"))
SAVE_STEPS = int(os.getenv("SAVE_STEPS", "500"))

LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))

SYSTEM_PROMPT = (
    "这是小学数学1-6年级的校内题目。"
    "无需进行分析，请直接输出最终答案，不带单位。"
    "不要输出解题过程、解释、标点或多余文字。"
)


def join_text(value):
    if isinstance(value, list):
        return "".join(str(x) for x in value)
    return str(value)


def build_example(example, tokenizer):
    question = join_text(example["question"])
    answer = str(example["answer"]).strip()
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        tokenize=False,
    )

    marker = "<|im_start|>assistant\n"
    split_pos = full_text.rfind(marker)
    if split_pos < 0:
        raise ValueError("assistant marker not found in chat template")
    split_pos += len(marker)

    prompt_text = full_text[:split_pos]
    answer_text = full_text[split_pos:]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]

    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id
    input_ids = prompt_ids + answer_ids + [eos_id]
    labels = [-100] * len(prompt_ids) + answer_ids + [eos_id]
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"加载模型: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    with Path(DATA_PATH).open("r", encoding="utf-8") as f:
        train_data = json.load(f)
    print(f"训练数据: {len(train_data)} 条")

    train_dataset = [build_example(item, tokenizer) for item in train_data]

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        inference_mode=False,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=10,
        num_train_epochs=EPOCHS,
        save_steps=SAVE_STEPS,
        learning_rate=LEARNING_RATE,
        save_total_limit=3,
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        remove_unused_columns=False,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    print("开始 answer-only SFT 训练...")
    trainer.train()

    Path(FINAL_DIR).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print(f"模型已保存到: {FINAL_DIR}")


if __name__ == "__main__":
    main()
