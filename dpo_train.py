"""
dpo_train.py — DPO 训练脚本

工作流程：
  1. 下载/加载基座模型 Qwen2.5-0.5B-Instruct
  2. 加载队友的的 LoRA（作为 SFT 参考）
  3. 合并 SFT LoRA → 然后挂新的 DPO LoRA
  4. 加载 dpo_train.json 偏好对数据
  5. 用 DPOTrainer 训练
  6. 保存 DPO LoRA 适配器
"""

import json
import os
import inspect

import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, PeftModel
from trl import DPOTrainer, DPOConfig


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./")
SFT_LORA_PATH = os.getenv("SFT_LORA_PATH", "方案2/final")
DPO_DATA_PATH = os.getenv("DPO_DATA_PATH", "dpo_train_answer.json")
OUTPUT_DIR = os.getenv("DPO_OUTPUT_DIR", "./output/Qwen_dpo")
FINAL_DIR = os.getenv("DPO_FINAL_DIR", os.path.join(OUTPUT_DIR, "final"))

PER_DEVICE_BATCH_SIZE = int(os.getenv("DPO_BATCH_SIZE", "8"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("DPO_GRAD_ACCUM", "4"))
LEARNING_RATE = float(os.getenv("DPO_LR", "2e-6"))
NUM_TRAIN_EPOCHS = float(os.getenv("DPO_EPOCHS", "1"))
MAX_STEPS = int(os.getenv("DPO_MAX_STEPS", "-1"))
BETA = float(os.getenv("DPO_BETA", "0.2"))
SAVE_STEPS = int(os.getenv("DPO_SAVE_STEPS", "500"))
DPO_LOSS_TYPE = os.getenv("DPO_LOSS_TYPE", "sigmoid")
DPO_LABEL_SMOOTHING = float(os.getenv("DPO_LABEL_SMOOTHING", "0"))

LORA_R = int(os.getenv("DPO_LORA_R", "16"))
LORA_ALPHA = int(os.getenv("DPO_LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("DPO_LORA_DROPOUT", "0.05"))

USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
DTYPE = torch.bfloat16 if USE_BF16 else torch.float16


def make_dpo_config(**kwargs):
    supported = set(inspect.signature(DPOConfig).parameters)
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    skipped = sorted(set(kwargs) - set(filtered))
    if skipped:
        print(f"DPOConfig 不支持这些参数，已跳过: {skipped}")
    return DPOConfig(**filtered)

# ============================================================
# 第1步：加载基座模型
# ============================================================
print("=" * 50)
print("第1步：下载/加载基座模型")
print("=" * 50)

model_dir = snapshot_download(
    MODEL_NAME,
    cache_dir=CACHE_DIR,
    revision="master"
)
tokenizer = AutoTokenizer.from_pretrained(
    model_dir, use_fast=False, trust_remote_code=True
)

# Qwen 的 tokenizer 默认没有 pad_token，需要手动设置
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map="auto",
    torch_dtype=DTYPE,
)
model.config.use_cache = False

# ============================================================
# 第2步：加载训练好的 LoRA + 合并
# ============================================================
print("=" * 50)
print("第2步：加载 SFT LoRA 适配器，合并为 DPO 起点")
print("=" * 50)

model = PeftModel.from_pretrained(model, SFT_LORA_PATH)

# 合并！这步很重要：把 LoRA 融进基座，得到 SFT 模型
# 之后 DPOTrainer 会自动复制一份作为 ref_model（冻结，用作KL计算的锚点）
print("正在合并 LoRA 权重...")
model = model.merge_and_unload()
print("合并完成！")

# ============================================================
# 第3步：挂新的 DPO LoRA
# ============================================================
print("=" * 50)
print("第3步：配置 DPO LoRA")
print("=" * 50)

dpo_lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
)

# DPOTrainer 会自己在 model 上加 DPO LoRA，所以这里不手动调 get_peft_model
# 只需要准备好 LoRA 配置，在 DPOTrainer 里通过 peft_config 传进去

# ============================================================
# 第4步：加载并格式化 DPO 数据
# ============================================================
print("=" * 50)
print("第4步：加载偏好对数据")
print("=" * 50)

with open(DPO_DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"原始数据: {len(raw_data)} 条")
if raw_data:
    print(f"样例 prompt 前80字: {raw_data[0]['prompt'][:80]!r}")
    print(f"样例 chosen: {raw_data[0]['chosen'][:80]!r}")
    print(f"样例 rejected: {raw_data[0]['rejected'][:80]!r}")

# 转为 HuggingFace Dataset 格式
train_dataset = Dataset.from_list(raw_data)
print(f"数据集创建完成: {len(train_dataset)} 条")

# ============================================================
# 第5步：训练参数配置
# ============================================================
print("=" * 50)
print("第5步：配置训练参数")
print("=" * 50)

training_args = make_dpo_config(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    gradient_checkpointing=True,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    max_steps=MAX_STEPS,
    logging_steps=10,
    save_steps=SAVE_STEPS,
    bf16=USE_BF16,
    fp16=not USE_BF16,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_grad_norm=1.0,
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="none",
    seed=42,
    beta=BETA,          # DPO 的核心参数，越大越保守
    loss_type=DPO_LOSS_TYPE,
    label_smoothing=DPO_LABEL_SMOOTHING,
)

# ============================================================
# 第6步：创建 DPOTrainer
# ============================================================
print("=" * 50)
print("第6步：创建 DPOTrainer")
print("=" * 50)

# DPOTrainer 参数说明：
#   - model: 策略模型（SFT模型 + DPO LoRA，要训练）
#   - ref_model: None → DPOTrainer 自动复制 model 并禁用 LoRA
#                 这样 ref_model = SFT 合并模型（冻结），正是我们需要的
#   - beta: DPO 的核心参数。越大 → 越不能偏离 SFT；越小 → 越激进

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=dpo_lora_config,
)

# ============================================================
# 第7步：开始训练
# ============================================================
print("=" * 50)
print("第7步：开始 DPO 训练")
print("=" * 50)

trainer.train()

# ============================================================
# 第8步：保存模型
# ============================================================
print("=" * 50)
print("第8步：保存 DPO LoRA")
print("=" * 50)

trainer.save_model(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)
print(f"DPO LoRA 已保存到 {FINAL_DIR}")
print("完成！")
