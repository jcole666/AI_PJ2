"""
Validate a submission CSV before uploading.

Usage:
  python validate_submit.py submit_dpo.csv
  python validate_submit.py 方案2/submit1.csv
"""
import argparse
import csv
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--test", default="test.json")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    test_path = Path(args.test)

    with test_path.open("r", encoding="utf-8") as f:
        test_data = json.load(f)
    expected_ids = [str(row["id"]) for row in test_data]

    rows = []
    bad_columns = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for line_no, row in enumerate(reader, start=1):
            rows.append(row)
            if len(row) != 2:
                bad_columns.append((line_no, len(row), row[:4]))

    ids = [row[0] for row in rows if row]
    answers = [row[1] for row in rows if len(row) >= 2]
    empty_answers = [i + 1 for i, ans in enumerate(answers) if not ans.strip()]
    long_answers = [i + 1 for i, ans in enumerate(answers) if len(ans) > 50]
    chinese_answers = [
        i + 1 for i, ans in enumerate(answers)
        if re.search(r"[\u4e00-\u9fff]", ans)
    ]

    missing_ids = sorted(set(expected_ids) - set(ids), key=int)
    extra_ids = sorted(set(ids) - set(expected_ids), key=lambda x: int(x) if x.isdigit() else x)
    duplicate_ids = sorted({x for x in ids if ids.count(x) > 1}, key=lambda x: int(x) if x.isdigit() else x)

    print(f"文件: {csv_path}")
    print(f"行数: {len(rows)} / 期望: {len(expected_ids)}")
    print(f"列数错误行: {len(bad_columns)}")
    print(f"缺失 ID: {len(missing_ids)}")
    print(f"额外 ID: {len(extra_ids)}")
    print(f"重复 ID: {len(duplicate_ids)}")
    print(f"空答案: {len(empty_answers)}")
    print(f"答案长度 > 50: {len(long_answers)}")
    print(f"含中文答案: {len(chinese_answers)}")

    if bad_columns[:5]:
        print(f"列数错误示例: {bad_columns[:5]}")
    if missing_ids[:10]:
        print(f"缺失 ID 示例: {missing_ids[:10]}")
    if empty_answers[:10]:
        print(f"空答案行示例: {empty_answers[:10]}")
    if long_answers[:10]:
        print(f"长答案行示例: {long_answers[:10]}")

    ok = (
        len(rows) == len(expected_ids)
        and not bad_columns
        and not missing_ids
        and not extra_ids
        and not duplicate_ids
        and not empty_answers
    )
    print("格式检查:", "OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
