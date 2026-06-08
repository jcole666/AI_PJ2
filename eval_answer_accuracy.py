"""
Quick exact-match evaluation on labeled data.

Examples:
  python eval_answer_accuracy.py --lora ./output/Qwen_answer_sft/final --limit 1000
  python eval_answer_accuracy.py --sft-lora ./output/Qwen_answer_sft/final --lora ./output/Qwen_dpo/final --offset 10000 --limit 1000
  python eval_answer_accuracy.py --prompt-mode cot --lora ./方案2/final --offset 10000 --limit 1000
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from infer_dpo import canonical_answer, extract_answer, get_system_prompt, vote_answers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft-lora", default="")
    parser.add_argument("--lora", default="")
    parser.add_argument("--data", default="train.json")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--prompt-mode", choices=["direct", "cot"], default="direct")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output-errors", default="")
    return parser.parse_args()


def join_text(value):
    if isinstance(value, list):
        return "".join(str(x) for x in value)
    return str(value)


def norm_answer(value):
    return canonical_answer(extract_answer(str(value))).strip()


def main():
    args = parse_args()
    if args.max_new_tokens <= 0:
        args.max_new_tokens = 512 if args.prompt_mode == "cot" else 64
    system_prompt = get_system_prompt(args.prompt_mode)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if args.lora:
        if args.sft_lora:
            print(f"加载并合并 SFT LoRA: {args.sft_lora}")
            model = PeftModel.from_pretrained(model, args.sft_lora)
            model = model.merge_and_unload()
        print(f"加载并合并目标 LoRA: {args.lora}")
        model = PeftModel.from_pretrained(model, args.lora)
        model = model.merge_and_unload()
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id

    with Path(args.data).open("r", encoding="utf-8") as f:
        data = json.load(f)
    data = data[args.offset: args.offset + args.limit]
    print(
        f"PROMPT_MODE={args.prompt_mode}, MAX_NEW_TOKENS={args.max_new_tokens}, "
        f"NUM_SAMPLES={args.num_samples}"
    )

    total = 0
    correct = 0
    errors = []

    for i in tqdm(range(0, len(data), args.batch_size), desc="eval"):
        batch = data[i:i + args.batch_size]
        texts = []
        for row in batch:
            question = join_text(row["question"])
            texts.append(tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                ],
                tokenize=False,
                add_generation_prompt=True,
            ))

        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generation_kwargs = {
                **inputs,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.num_samples > 1,
                "num_return_sequences": args.num_samples,
            }
            if args.num_samples > 1:
                generation_kwargs.update({
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                })
            outputs = model.generate(**generation_kwargs)

        input_len = inputs.input_ids.shape[1]
        for j, row in enumerate(batch):
            answers = []
            raw_responses = []
            for sample_idx in range(args.num_samples):
                output_idx = j * args.num_samples + sample_idx
                response = tokenizer.decode(
                    outputs[output_idx][input_len:], skip_special_tokens=True
                )
                raw_responses.append(response.strip().replace("\n", " "))
                answers.append(norm_answer(response))
            pred = vote_answers(answers)
            gold = norm_answer(row["answer"])
            ok = pred == gold
            total += 1
            correct += int(ok)
            if not ok and len(errors) < 2000:
                errors.append({
                    "id": row["id"],
                    "question": join_text(row["question"]),
                    "gold": gold,
                    "pred": pred,
                    "raw": " || ".join(raw_responses[:3]),
                })

    acc = correct / total if total else 0
    print(f"total={total} correct={correct} accuracy={acc:.4f}")

    if args.output_errors:
        with Path(args.output_errors).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "question", "gold", "pred", "raw"])
            writer.writeheader()
            writer.writerows(errors)
        print(f"errors saved to: {args.output_errors}")


if __name__ == "__main__":
    main()
