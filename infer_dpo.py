"""
DPO 模型推理脚本 — 在 test.json 上批量推理，生成提交 CSV
"""
import csv
import json
import os
import re

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# 路径
MODEL_DIR = os.getenv("MODEL_DIR", "./Qwen/Qwen2.5-0.5B-Instruct")
SFT_LORA_PATH = os.getenv("SFT_LORA_PATH", "")
LORA_PATH = os.getenv("LORA_PATH", "./output/Qwen_dpo/final")
TEST_PATH = os.getenv("TEST_PATH", "./test.json")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "submit_dpo.csv")
BATCH_SIZE = int(os.getenv("INFER_BATCH_SIZE", "128"))
PROMPT_MODE = os.getenv("PROMPT_MODE", "direct").lower()
MAX_NEW_TOKENS = int(os.getenv(
    "MAX_NEW_TOKENS", "512" if PROMPT_MODE == "cot" else "64"
))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "1"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("TOP_P", "0.9"))
DIRECT_SYSTEM_PROMPT = (
    "这是小学数学1-6年级的校内题目。"
    "无需进行分析，请直接输出最终答案，不带单位。"
    "不要输出解题过程、解释、标点或多余文字。"
)
COT_SYSTEM_PROMPT = (
    "你是一个小学数学解题助手。请一步一步推理分析题目，"
    "最后用【答案】=XXX的格式给出最终答案。"
)

FRACTION_RE = re.compile(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_MARK_RE = re.compile(
    r"(?:【答案】|最终答案|答案|结果)\s*(?:是|为|[:：=])?\s*([^，。；;！!？?\n\r]+)"
)
TOKEN_RE = re.compile(
    r"[-+]?(?:(?:\d+(?:\.\d+)?)?π|π(?:\d+(?:\.\d+)?)?)"
    r"|[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:_\d+/\d+|/\d+(?:\.\d+)?)?%?"
)


def get_system_prompt(mode="direct"):
    if mode == "cot":
        return COT_SYSTEM_PROMPT
    return DIRECT_SYSTEM_PROMPT


SYSTEM_PROMPT = get_system_prompt(PROMPT_MODE)


def normalize_text(text):
    text = text.translate(str.maketrans({
        "％": "%",
        "－": "-",
        "＋": "+",
        "．": ".",
        "，": ",",
        "：": ":",
        "（": "(",
        "）": ")",
    }))
    previous = None
    while previous != text:
        previous = text
        text = FRACTION_RE.sub(r"\1/\2", text)
    text = BOXED_RE.sub(r"\1", text)
    text = text.replace("\\(", " ").replace("\\)", " ")
    text = text.replace("\\[", " ").replace("\\]", " ")
    text = text.replace("$", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def pick_answer_token(text, prefer_last=False):
    tokens = [m.group(0).replace(" ", "") for m in TOKEN_RE.finditer(text)]
    if not tokens:
        return ""
    return tokens[-1] if prefer_last else tokens[0]


def clean_candidate(text, prefer_last=False):
    text = normalize_text(text)
    text = text.strip(" \t\r\n,。.;；:：!！?？、`'\"[]{}")
    token = pick_answer_token(text, prefer_last=prefer_last)
    if token:
        return token
    return text.replace(" ", "")


def canonical_answer(answer):
    answer = str(answer).strip()
    answer = answer.strip(" \t\r\n,。.;；:：!！?？、`'\"[]{}")
    if re.fullmatch(r"[-+]?\d+\.0+", answer):
        return answer.split(".")[0]
    if re.fullmatch(r"[-+]?\d+\.\d+", answer):
        answer = answer.rstrip("0").rstrip(".")
    return answer


def extract_answer(response):
    text = normalize_text(response)

    for match in BOXED_RE.finditer(text):
        answer = clean_candidate(match.group(1), prefer_last=False)
        if answer:
            return canonical_answer(answer)

    for match in ANSWER_MARK_RE.finditer(text):
        answer = clean_candidate(match.group(1), prefer_last=False)
        if answer:
            return canonical_answer(answer)

    answer = clean_candidate(text, prefer_last=False)
    if answer and len(text) <= 40:
        return canonical_answer(answer)

    answer = pick_answer_token(text, prefer_last=True)
    if answer:
        return canonical_answer(answer)

    return canonical_answer(text.replace(" ", ""))


def vote_answers(answers):
    counts = {}
    first_index = {}
    for idx, answer in enumerate(answers):
        answer = canonical_answer(answer)
        if not answer:
            continue
        counts[answer] = counts.get(answer, 0) + 1
        first_index.setdefault(answer, idx)
    if not counts:
        return ""
    return max(counts, key=lambda x: (counts[x], -first_index[x]))

def main():
    # 加载模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 批量生成必须左填充

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, device_map="auto", torch_dtype=torch.bfloat16
    )

    if SFT_LORA_PATH:
        print(f"加载并合并 SFT LoRA: {SFT_LORA_PATH}")
        model = PeftModel.from_pretrained(model, SFT_LORA_PATH)
        model = model.merge_and_unload()

    print(f"加载并合并目标 LoRA: {LORA_PATH}")
    model = PeftModel.from_pretrained(model, LORA_PATH)
    model = model.merge_and_unload()
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    # 加载测试数据
    with open(TEST_PATH, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    print(f"测试数据: {len(test_data)} 条")
    print(
        f"PROMPT_MODE={PROMPT_MODE}, MAX_NEW_TOKENS={MAX_NEW_TOKENS}, "
        f"NUM_SAMPLES={NUM_SAMPLES}"
    )

    results = []

    for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
        batch = test_data[i:i + BATCH_SIZE]

        # 批量构造 prompt
        texts = []
        for row in batch:
            question = row["question"]
            if isinstance(question, list):
                question = "".join(question)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            texts.append(tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            generation_kwargs = {
                **inputs,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": NUM_SAMPLES > 1,
                "num_return_sequences": NUM_SAMPLES,
            }
            if NUM_SAMPLES > 1:
                generation_kwargs.update({
                    "temperature": TEMPERATURE,
                    "top_p": TOP_P,
                })
            outputs = model.generate(**generation_kwargs)

        # 解码并提取答案
        input_len = inputs.input_ids.shape[1]
        for j, row in enumerate(batch):
            answers = []
            for sample_idx in range(NUM_SAMPLES):
                output_idx = j * NUM_SAMPLES + sample_idx
                response = tokenizer.decode(
                    outputs[output_idx][input_len:], skip_special_tokens=True
                )
                answers.append(extract_answer(response))
            answer = vote_answers(answers)
            results.append({"id": row["id"], "answer": answer})

    # 保存为 CSV
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows((r["id"], r["answer"]) for r in results)

    print(f"完成！保存到 {OUTPUT_CSV}，共 {len(results)} 条")


if __name__ == "__main__":
    main()
