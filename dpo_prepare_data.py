import json

# 1. 读取两个文件
with open("方案2/train_cot.json", "r", encoding="utf-8") as f:
    train_cot = json.load(f)

with open("方案2/reject.json", "r", encoding="utf-8") as f:
    reject = json.load(f)

print(f"train_cot.json 有 {len(train_cot)} 条")
print(f"reject.json 有 {len(reject)} 条")

# 2. 把 reject.json 按 id 建立索引，方便查找
reject_dict = {}
for item in reject:
    reject_dict[item["id"]] = item

# 3. 按 id 匹配，构造偏好对
dpo_data = []
for item in train_cot:
    rid = item["id"]
    if rid not in reject_dict:
        continue  # 这个 id 在 reject 里没有对应的，跳过

    # 处理 question 可能是列表的情况
    question = item["question"]
    if isinstance(question, list):
        question = "".join(question)

    # 处理 instruction 可能是列表的情况
    instruction = item["instruction"]
    if isinstance(instruction, list):
        instruction = "".join(instruction)

    # prompt = instruction + question
    prompt = instruction + "\n" + question

    # chosen = 好的推理链（train_cot 里）
    chosen = item["cot"]

    # rejected = 坏的推理链（reject 里）
    rejected = reject_dict[rid]["cot"]

    dpo_data.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    })

print(f"成功匹配 {len(dpo_data)} 条偏好对")

# 4. 保存
with open("dpo_train.json", "w", encoding="utf-8") as f:
    json.dump(dpo_data, f, ensure_ascii=False, indent=2)

print("已保存到 dpo_train.json")