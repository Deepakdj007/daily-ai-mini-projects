"""Loading the task corpus, and stamping which corpus it was.

Inputs:  data/tasks.json
Outputs: tasks in their declared order, families, and a digest of both

The order is part of the experiment. Tasks run in the order the file declares,
identical for every arm, because the whole benefit of a persistent toolbox
depends on the first member of a family coming before the other three. Shuffle
it and the reuse figure becomes a function of the seed.

The digest covers the corpus and the oracle together. A leaderboard that does
not name the questions it answered is unfalsifiable: any later edit silently
rewrites history, and the grader is as much a part of that as the questions.
"""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess

from src import config, oracle


@functools.lru_cache(maxsize=1)
def load() -> dict:
    """The corpus, read once."""
    if not config.TASKS_PATH.exists():
        raise SystemExit(
            f"{config.TASKS_PATH} is missing. Build it first:\n"
            "  PYTHONPATH=. uv run python -m tools.author_tasks")
    return json.loads(config.TASKS_PATH.read_text(encoding="utf-8"))


def families() -> list[dict]:
    return load()["families"]


def family(key: str) -> dict:
    """One family by key."""
    for entry in families():
        if entry["family"] == key:
            return entry
    raise KeyError(f"no family called {key!r}")


def tasks() -> list[dict]:
    """Every task, in the declared order."""
    by_id = {task["id"]: task for entry in families() for task in entry["tasks"]}
    return [by_id[task_id] for task_id in load()["task_order"]]


def task(task_id: str) -> dict:
    """One task by id."""
    for entry in tasks():
        if entry["id"] == task_id:
            return entry
    raise SystemExit(f"no task called {task_id!r}. Try: "
                     f"{', '.join(t['id'] for t in tasks()[:6])} ...")


def tasks_for(profile_families: tuple[str, ...] | None = None) -> list[dict]:
    """Tasks limited to some families, still in the declared order."""
    if not profile_families:
        return tasks()
    wanted = set(profile_families)
    return [entry for entry in tasks() if entry["family"] in wanted]


def order_hash(entries: list[dict]) -> str:
    """A hash of the exact sequence an arm ran, so two arms can be compared."""
    blob = "|".join(entry["id"] for entry in entries)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def digest() -> str:
    """sha256 of the corpus file, plus the git commit that last touched it."""
    body = hashlib.sha256(config.TASKS_PATH.read_bytes()).hexdigest()[:16]
    try:
        commit = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", str(config.TASKS_PATH)],
            capture_output=True, text=True, timeout=10,
            cwd=str(config.ROOT_DIR)).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        commit = ""
    return f"sha256:{body}" + (f" git:{commit}" if commit else " git:UNCOMMITTED")


def stamp() -> dict:
    """Everything about the corpus that belongs in a results manifest."""
    return {"corpus": digest(), "oracle": oracle.digest(),
            "corpus_version": load().get("version", "?"),
            "order": order_hash(tasks())}


if __name__ == "__main__":
    all_tasks = tasks()
    assert len(all_tasks) == 24, len(all_tasks)
    assert [t["id"] for t in all_tasks] == load()["task_order"]
    assert load()["oracle"] == oracle.digest(), (
        "data/tasks.json was built by a different oracle than the one installed. "
        "Re-run: PYTHONPATH=. uv run python -m tools.author_tasks")

    for entry in families():
        assert len(entry["tasks"]) == 4 and len(entry["holdout"]) == 8, entry["family"]
        first = [t for t in all_tasks if t["family"] == entry["family"]][0]
        assert first["member"] == 1, "a family's first member must come first"

    # Family blocks must be contiguous, or a tool written for the first member
    # would have to survive other families before it could be reused.
    seen: list[str] = []
    for entry in all_tasks:
        if not seen or seen[-1] != entry["family"]:
            assert entry["family"] not in seen, f"{entry['family']} is split up"
            seen.append(entry["family"])

    print(f"{len(all_tasks)} tasks in {len(families())} families, order {order_hash(all_tasks)}")
    print(f"corpus {digest()}")
    print(f"oracle {oracle.digest()}")
    for entry in families():
        ids = ", ".join(t["id"] for t in entry["tasks"])
        print(f"  {entry['family']:<9} {ids}")
    print("OK - corpus")
