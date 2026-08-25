"""Assemble data/queries.json from authored items plus a GSM8K slice.

Inputs:  data/authored.json, data/_gsm8k_test.jsonl
Outputs: data/queries.json

Half the eval set is written here and half is imported verbatim from a public
benchmark. That split is deliberate. Items we author are the only way to get
realistic product traffic - ticket routing, field extraction, unit conversion -
which no public dataset supplies. But an eval set written entirely by the
person reporting the result invites the obvious objection, so the hard half is
lifted unmodified from GSM8K and we never touch its questions or its answers.

Difficulty on the GSM8K items is not our opinion either: it is the number of
calculator annotations in the reference solution, which is the problem's own
step count.
"""

from __future__ import annotations

import json
import re

from src.config import DATA_DIR, QUERIES_PATH

GSM8K_PATH = DATA_DIR / "_gsm8k_test.jsonl"
AUTHORED_PATH = DATA_DIR / "authored.json"

# How many GSM8K items to vendor, and where the difficulty line falls.
N_MEDIUM = 22
N_HARD = 18
MEDIUM_MAX_STEPS = 2
HARD_MIN_STEPS = 4

_STEP_RE = re.compile(r"<<[^>]*>>")
_ANNOT_RE = re.compile(r"<<[^>]*>>")


def load_gsm8k() -> list[dict]:
    """Read the raw GSM8K test split."""
    if not GSM8K_PATH.exists():
        raise SystemExit(
            f"Missing {GSM8K_PATH}.\nFetch it with:\n"
            "  curl -sS -o data/_gsm8k_test.jsonl "
            "https://raw.githubusercontent.com/openai/grade-school-math/master/"
            "grade_school_math/data/test.jsonl"
        )
    with GSM8K_PATH.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gold_of(answer: str) -> str:
    """GSM8K puts the final numeric answer after '####' on the last line."""
    return answer.split("####")[-1].strip().replace(",", "")


def steps_of(answer: str) -> int:
    """Count the reference solution's calculator annotations."""
    return len(_STEP_RE.findall(answer))


def build_gsm8k_items() -> list[dict]:
    """Select a deterministic, difficulty-stratified slice of GSM8K."""
    rows = load_gsm8k()
    medium: list[dict] = []
    hard: list[dict] = []

    for index, row in enumerate(rows):
        answer = row["answer"]
        steps = steps_of(answer)
        gold = gold_of(answer)

        # Skip anything whose gold is not a clean number - grading has to stay
        # unambiguous, and a handful of GSM8K golds carry stray formatting.
        try:
            float(gold)
        except ValueError:
            continue

        item = {
            "id": f"gsm-{index:04d}",
            "category": "gsm8k",
            "difficulty": "medium" if steps <= MEDIUM_MAX_STEPS else "hard",
            "query": row["question"].strip() + "\nGive digits only.",
            "gold": gold,
            "source": "gsm8k",
            "steps": steps,
        }

        if steps <= MEDIUM_MAX_STEPS and len(medium) < N_MEDIUM:
            medium.append(item)
        elif steps >= HARD_MIN_STEPS and len(hard) < N_HARD:
            hard.append(item)

        if len(medium) == N_MEDIUM and len(hard) == N_HARD:
            break

    return medium + hard


def build() -> list[dict]:
    """Merge authored and vendored items into the final eval set."""
    with AUTHORED_PATH.open(encoding="utf-8") as handle:
        authored = json.load(handle)
    for item in authored:
        item.setdefault("source", "authored")

    items = authored + build_gsm8k_items()

    # Every item must declare whether its gold is numeric, because the grader
    # compares numbers numerically and words on word boundaries.
    for item in items:
        try:
            float(item["gold"].replace(",", ""))
            item["numeric"] = True
        except ValueError:
            item["numeric"] = False

    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate item ids in the eval set")

    return items


if __name__ == "__main__":
    items = build()
    QUERIES_PATH.write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")

    by_difficulty: dict[str, int] = {}
    by_category: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for item in items:
        by_difficulty[item["difficulty"]] = by_difficulty.get(item["difficulty"], 0) + 1
        by_category[item["category"]] = by_category.get(item["category"], 0) + 1
        by_source[item["source"]] = by_source.get(item["source"], 0) + 1

    total = len(items)
    print(f"wrote {QUERIES_PATH}  ({total} items)\n")
    print("difficulty mix (target 60/25/15 easy/medium/hard):")
    for level in ("easy", "medium", "hard"):
        count = by_difficulty.get(level, 0)
        print(f"  {level:7s} {count:3d}  {count / total:5.1%}")
    print("\nby source:")
    for source, count in sorted(by_source.items()):
        print(f"  {source:9s} {count:3d}")
    print("\nby category:")
    for category, count in sorted(by_category.items()):
        print(f"  {category:14s} {count:3d}")
    print(f"\nnumeric golds: {sum(i['numeric'] for i in items)} / {total}")
