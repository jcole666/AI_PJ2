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
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from trl import DPOTrainer, DPOConfig

# ============================================================
# 第1步：加载基座模型
# ============================================================
print("=" * 50)
print("第1步：下载/加载基座模型")
print("=" * 50)

model_dir = snapshot_download(
    "Qwen/Qwen2.5-0.5B-Instruct",
    cache_dir="./",
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
    torch_dtype=torch.bfloat16,
)

# ============================================================
# 第2步：加载训练好的 LoRA + 合并
# ============================================================
print("=" * 50)
print("第2步：加载 LoRA 适配器，合并为参考模型")
print("=" * 50)

# 加载 LoRA（暂时替 SFT 队友）
sft_lora_path = "方案2/final"
model = PeftModel.from_pretrained(model, sft_lora_path)

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
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

# DPOTrainer 会自己在 model 上加 DPO LoRA，所以这里不手动调 get_peft_model
# 只需要准备好 LoRA 配置，在 DPOTrainer 里通过 peft_config 传进去

# ============================================================
# 第4步：加载并格式化 DPO 数据
# ============================================================
print("=" * 50)
print("第4步：加载偏好对数据")
print("=" * 50)

with open("dpo_train.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

print(f"原始数据: {len(raw_data)} 条")

# 转为 HuggingFace Dataset 格式
train_dataset = Dataset.from_list(raw_data)
print(f"数据集创建完成: {len(train_dataset)} 条")

# ============================================================
# 第5步：训练参数配置
# ============================================================
print("=" * 50)
print("第5步：配置训练参数")
print("=" * 50)

training_args = DPOConfig(
    output_dir="./output/Qwen_dpo",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=5e-6,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    bf16=True,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="none",
    beta=0.1,          # DPO 的核心参数，放在 DPOConfig 里
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

save_path = "./output/Qwen_dpo/final"
trainer.save_model(save_path)
print(f"DPO LoRA 已保存到 {save_path}")
print("完成！")
