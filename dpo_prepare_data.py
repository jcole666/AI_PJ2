"""
Build preference pairs for DPO.

Default mode is answer-level DPO, which matches the competition scorer:
the chosen response is the ground-truth final answer, and the rejected
response is the base model's wrong final answer.

Usage:
  python dpo_prepare_data.py
  python dpo_prepare_data.py --mode cot --output dpo_train_cot.json
"""
import argparse
import json
import re
from pathlib import Path


DIRECT_SYSTEM_PROMPT = (
    "这是小学数学1-6年级的校内题目。"
    "无需进行分析，请直接输出最终答案，不带单位。"
    "不要输出解题过程、解释、标点或多余文字。"
)
COT_SYSTEM_PROMPT = (
    "你是一个小学数学解题助手。请一步一步推理分析题目，"
    "最后用【答案】=XXX的格式给出最终答案。"
)
CHAT_PREFIX = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
CHAT_END = "<|im_end|>\n"

FRACTION_RE = re.compile(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}")
BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
ANSWER_MARK_RE = re.compile(
    r"(?:【答案】|最终答案|答案|结果)\s*(?:是|为|[:：=])?\s*([^，。；;！!？?\n\r]+)"
)
TOKEN_RE = re.compile(
    r"[-+]?(?:(?:\d+(?:\.\d+)?)?π|π(?:\d+(?:\.\d+)?)?)"
    r"|[-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:_\d+/\d+|/\d+(?:\.\d+)?)?%?"
)


def join_text(value):
    if isinstance(value, list):
        return "".join(str(x) for x in value)
    return str(value)


def normalize_text(text):
    text = str(text).translate(str.maketrans({
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
    return re.sub(r"\s+", " ", text).strip()


def pick_answer_token(text, prefer_last=False):
    tokens = [m.group(0).replace(" ", "") for m in TOKEN_RE.finditer(text)]
    if not tokens:
        return ""
    return tokens[-1] if prefer_last else tokens[0]


def clean_answer(text, prefer_last=False):
    text = normalize_text(text).strip(" \t\r\n,。.;；:：!！?？、`'\"[]{}")
    token = pick_answer_token(text, prefer_last=prefer_last)
    if token:
        return token
    return text.replace(" ", "")


def extract_answer(text):
    text = normalize_text(text)

    for match in BOXED_RE.finditer(text):
        answer = clean_answer(match.group(1), prefer_last=False)
        if answer:
            return answer

    for match in ANSWER_MARK_RE.finditer(text):
        answer = clean_answer(match.group(1), prefer_last=False)
        if answer:
            return answer

    answer = clean_answer(text, prefer_last=False)
    if answer and len(text) <= 40:
        return answer

    return pick_answer_token(text, prefer_last=True)


def canonical(answer):
    answer = clean_answer(answer, prefer_last=False)
    if answer.endswith(".0"):
        answer = answer[:-2]
    return answer


def qwen_prompt(system, question):
    return CHAT_PREFIX.format(system=system, question=join_text(question))


def ensure_final_answer(cot, answer):
    cot = str(cot).strip()
    if "【答案】" in cot or "最终答案" in cot or "\\boxed" in cot:
        return cot + CHAT_END
    return f"{cot}\n\n【答案】={answer}{CHAT_END}"


def build_pairs(train_cot, reject, mode):
    reject_by_id = {str(item["id"]): item for item in reject}
    pairs = []
    skipped = {
        "missing_reject": 0,
        "bad_chosen": 0,
        "missing_rejected_answer": 0,
        "same_answer": 0,
    }

    for item in train_cot:
        rid = str(item["id"])
        rejected_item = reject_by_id.get(rid)
        if rejected_item is None:
            skipped["missing_reject"] += 1
            continue

        chosen_answer = canonical(item.get("answer", ""))
        if not chosen_answer or item.get("cot_ok") is False:
            skipped["bad_chosen"] += 1
            continue

        rejected_answer = canonical(rejected_item.get("answer", ""))
        if not rejected_answer:
            rejected_answer = canonical(extract_answer(rejected_item.get("cot", "")))
        if not rejected_answer:
            skipped["missing_rejected_answer"] += 1
            continue

        if rejected_answer == chosen_answer:
            skipped["same_answer"] += 1
            continue

        question = join_text(item["question"])
        if mode == "answer":
            pairs.append({
                "prompt": qwen_prompt(DIRECT_SYSTEM_PROMPT, question),
                "chosen": chosen_answer + CHAT_END,
                "rejected": rejected_answer + CHAT_END,
            })
        else:
            pairs.append({
                "prompt": qwen_prompt(COT_SYSTEM_PROMPT, question),
                "chosen": ensure_final_answer(item["cot"], chosen_answer),
                "rejected": ensure_final_answer(rejected_item["cot"], rejected_answer),
            })

    return pairs, skipped


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["answer", "cot"], default="answer")
    parser.add_argument("--train-cot", default="方案2/train_cot.json")
    parser.add_argument("--reject", default="方案2/reject.json")
    parser.add_argument("--output", default="dpo_train_answer.json")
    return parser.parse_args()


def main():
    args = parse_args()

    with Path(args.train_cot).open("r", encoding="utf-8") as f:
        train_cot = json.load(f)
    with Path(args.reject).open("r", encoding="utf-8") as f:
        reject = json.load(f)

    print(f"train_cot: {len(train_cot)} 条")
    print(f"reject: {len(reject)} 条")
    print(f"mode: {args.mode}")

    pairs, skipped = build_pairs(train_cot, reject, args.mode)
    print(f"成功构造偏好对: {len(pairs)} 条")
    print(f"跳过统计: {skipped}")

    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"已保存到: {args.output}")


if __name__ == "__main__":
    main()
