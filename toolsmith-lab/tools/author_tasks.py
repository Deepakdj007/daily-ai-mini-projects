"""Write data/tasks.json from the oracle.

    PYTHONPATH=. uv run python -m tools.author_tasks

Inputs:  the item generators below, a fixed seed, and src/oracle.py
Outputs: data/tasks.json, and a printed summary

No answer is typed by hand. Twenty-four exact answers and forty-eight held-out
expectations written out by a person is where a corpus dies quietly: one wrong
entry marks a right answer wrong for every arm at once, and the ladder still
looks perfectly monotone.

Items are generated from a fixed seed, and the split between what is shown and
what is held back is the whole design. Visible batches are drawn from ordinary
cases only. Every edge that a careless implementation gets wrong - a GSTIN
whose check character is a digit, a number with a leading zero, a date in week
53, an amount with an empty lakh segment, a version carrying build metadata -
is kept out of the visible set and salted into the held-out one. That is what
makes it possible for a tool to pass all four tasks it was shown and still be
broken, which is the thing the held-out rung exists to catch.
"""

from __future__ import annotations

import json
import random
import string
from typing import Any, Callable

from src import config, matcher, oracle, ref_ids, ref_misc, ref_money, shortcuts

SEED = 20260917
BATCH = {"gstin": 8, "verhoeff": 8, "isoweek": 6, "words": 5, "semver": 10}

_LETTERS = string.ascii_uppercase


def _gstin(rng: random.Random) -> str:
    """A well-formed 14-character GSTIN prefix: state, PAN, entity, Z."""
    pan = ("".join(rng.choice(_LETTERS) for _ in range(5))
           + f"{rng.randrange(10000):04d}" + rng.choice(_LETTERS))
    return f"{rng.randrange(1, 38):02d}{pan}{rng.randrange(1, 10)}Z"


def _verhoeff(rng: random.Random) -> str:
    return f"{rng.randrange(1, 10)}" + "".join(str(rng.randrange(10)) for _ in range(10))


def _isoweek(rng: random.Random) -> str:
    from datetime import date, timedelta
    return (date(2024, 2, 1) + timedelta(days=rng.randrange(1000))).isoformat()


def _words(rng: random.Random) -> int:
    return rng.randrange(1000, 99_000_000)


def _semver(rng: random.Random) -> str:
    core = f"{rng.randrange(0, 12)}.{rng.randrange(0, 20)}.{rng.randrange(0, 15)}"
    tag = rng.choice(["", "", "", "-alpha.1", "-beta.2", "-rc.1", "-alpha.10"])
    return core + tag


# An item is "ordinary" when the plausible-wrong implementation happens to be
# right about it. Stated that way rather than as a list of hand-written
# conditions, because the condition IS the property the visible set needs: a
# tool shaped like the overfit has to pass every task it was shown, or the
# held-out rung has nothing left to catch. Writing the conditions out by hand
# invites exactly the mistake this replaced - a predicate that looked at the
# oracle's output for the word "zero" when it is the overfit that says it.
_SCALAR_ORACLE: dict[str, Callable[[Any], Any]] = {
    "gstin": ref_ids.gstin_check_char,
    "verhoeff": ref_ids.verhoeff_check_digit,
    "isoweek": ref_misc.iso_week_label,
    "words": ref_money.rupees_in_words,
}


def _ordinary(key: str, item: Any) -> bool:
    """Whether the overfit implementation is right about this item."""
    if key == "semver":
        return "+" not in item          # items here are single version strings
    try:
        return (str(shortcuts.SCALAR_OVERFIT[key](item))
                == str(_SCALAR_ORACLE[key](item)))
    except Exception:                    # noqa: BLE001 - a crash is a disagreement
        return False

# The edges, kept out of every visible prompt and salted into the held-out set.
TRAPS: dict[str, list] = {
    "gstin": ["29AAGCB7383J1Z", "19AABCU9603R1Z", "36AAFCD5862P1Z", "07AAACN2755L1Z"],
    "verhoeff": ["00000000001", "01234567890", "00099887766", "09080706050"],
    "isoweek": ["2027-01-01", "2028-02-29", "2026-01-01", "2024-12-30",
                "2023-01-01", "2021-01-03", "2026-12-31", "2032-12-31"],
    "words": [0, 10_005_000, 100_000, 10_000_000, 70_000_000, 1_000, 123, 5_000_000],
    "semver": ["1.0.0+build.7", "2.3.4+meta", "1.0.0-alpha", "1.0.0-alpha.beta",
               "0.0.1+meta", "3.0.0-rc.10", "3.0.0-rc.2", "1.0.0-x-y-z.-"],
}

GENERATORS: dict[str, Callable[[random.Random], Any]] = {
    "gstin": _gstin, "verhoeff": _verhoeff, "isoweek": _isoweek,
    "words": _words, "semver": _semver,
}

# The one family that stays scalar. A single loan already needs two hundred and
# forty rounds of rounding, which is well past anything a reasoning trace can
# carry, so batching it would only make the run slower.
LOAN_VISIBLE = [[1_000_000, 8.5, 240], [450_000, 11.25, 84],
                [250_000, 9.75, 60], [3_500_000, 7.9, 300]]
LOAN_HOLDOUT = [[120_000, 0.0, 12], [100_000, 12.0, 1], [75_000, 14.0, 18],
                [2_000_000, 6.75, 180], [640_000, 10.4, 96], [15_000, 0.0, 6],
                [5_000_000, 9.15, 360], [88_888, 13.33, 27]]

QUESTIONS: dict[str, str] = {
    "gstin": "What are the check characters for these GSTIN prefixes: {0}?",
    "loan": ("A loan of Rs {0:,.0f} is taken at {1}% a year over {2} months. "
             "What is the total interest paid over the life of the loan?"),
    "isoweek": "What are the ISO week labels for these dates: {0}?",
    "verhoeff": "What are the Verhoeff check digits for these numbers: {0}?",
    "words": "Write these amounts in words: {0}.",
    "semver": "Sort these versions from lowest to highest: {0}.",
}


def _pool(key: str, rng: random.Random, count: int, *, ordinary: bool) -> list:
    """`count` distinct generated items, all ordinary or all unfiltered."""
    keep, seen = [], set()
    while len(keep) < count:
        item = GENERATORS[key](rng)
        if item in seen or (ordinary and not _ordinary(key, item)):
            continue
        seen.add(item)
        keep.append(item)
    return keep


def _batches(items: list, size: int) -> list[list]:
    """Chunk a flat list of items into batches of `size`."""
    return [items[at:at + size] for at in range(0, len(items), size)]


def render(family_key: str, values: list) -> str:
    """The question for one task, with its inputs filled in."""
    if family_key == "loan":
        return QUESTIONS[family_key].format(*values)
    return QUESTIONS[family_key].format(", ".join(str(item) for item in values[0]))


def prompt_for(family: oracle.Family, values: list) -> str:
    """Contract, then question, then the answer shape. The contract is byte-
    identical across a family, which is what lets a tool be reused at all."""
    return (f"{family.full_contract()}\n\n{render(family.key, values)}\n\n"
            f"Answer with {family.returns}.")


def inputs_for(key: str, rng: random.Random) -> tuple[list[list], list[list]]:
    """(four visible argument lists, eight held-out argument lists)."""
    if key == "loan":
        return LOAN_VISIBLE, LOAN_HOLDOUT
    size = BATCH[key]
    visible = [[batch] for batch in _batches(_pool(key, rng, 4 * size, ordinary=True), size)]
    # Each held-out batch carries at least one edge case, with the rest filled
    # from ordinary items so the batch is not obviously special.
    traps = TRAPS[key]
    filler = _batches(_pool(key, rng, 8 * (size - 1), ordinary=True), size - 1)
    held = []
    for index in range(8):
        batch = [traps[index % len(traps)], *filler[index]]
        rng.shuffle(batch)
        held.append([batch])
    return visible, held


def build() -> dict:
    """The whole corpus, answers computed rather than declared."""
    rng = random.Random(SEED)
    families = []
    for family in oracle.FAMILIES:
        visible, held = inputs_for(family.key, rng)
        tasks = [
            {"id": f"{family.key}_{index}", "family": family.key, "member": index,
             "inputs": values, "prompt": prompt_for(family, values),
             "answer": oracle.solve_positional(family.key, values)}
            for index, values in enumerate(visible, start=1)
        ]
        holdout = [{"inputs": values,
                    "expect": oracle.solve_positional(family.key, values)}
                   for values in held]
        families.append({
            "family": family.key, "label": family.label,
            "contract": family.full_contract(), "example": list(family.example),
            "why_tool": family.why_tool, "returns": family.returns,
            "compare": family.compare, "params": [list(p) for p in family.params],
            "batch": BATCH.get(family.key, 1), "tasks": tasks, "holdout": holdout,
        })
    order = [task["id"] for entry in families for task in entry["tasks"]]
    return {"version": "2", "seed": SEED, "oracle": oracle.digest(),
            "task_order": order, "families": families}


def check(payload: dict) -> list[str]:
    """Structural checks, run before the file is written."""
    problems: list[str] = []
    for entry in payload["families"]:
        key, mode = entry["family"], entry["compare"]
        seen = {json.dumps(t["inputs"], sort_keys=True) for t in entry["tasks"]}
        held = {json.dumps(r["inputs"], sort_keys=True) for r in entry["holdout"]}
        if seen & held:
            problems.append(f"{key}: {len(seen & held)} held-out input(s) also shown")
        if len(seen) != len(entry["tasks"]) or len(held) != len(entry["holdout"]):
            problems.append(f"{key}: an input appears twice")
        if len({str(matcher.normalise(t["answer"], mode)) for t in entry["tasks"]}) == 1:
            problems.append(f"{key}: all four visible answers are the same value")
        if any(entry["contract"] not in t["prompt"] for t in entry["tasks"]):
            problems.append(f"{key}: the contract is not identical across the family")
    return problems


def main() -> int:
    """Build, check, write, and print what went in."""
    config.ensure_dirs()
    payload = build()
    problems = check(payload)
    for problem in problems:
        print(f"  REFUSED  {problem}")
    if problems:
        return 1

    config.TASKS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    total = sum(len(e["tasks"]) for e in payload["families"])
    held = sum(len(e["holdout"]) for e in payload["families"])
    print(f"wrote {config.TASKS_PATH.relative_to(config.ROOT_DIR)} - {total} tasks, "
          f"{held} held-out batches, seed {SEED}, oracle {payload['oracle']}\n")
    for entry in payload["families"]:
        first = str(entry["tasks"][0]["answer"])
        print(f"  {entry['family']:<9} batch of {entry['batch']:<2} "
              f"first answer: {first[:64]}{'...' if len(first) > 64 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
