"""Every check that costs no inference, run before a model is asked anything.

Inputs:  the probe file, the case folders, the lab truth
Outputs: a list of (gate, ok, detail); failures stop `run`

A probe set that is unbalanced, or labelled from silence, or that leaks its
answer into its wording, fails silently: the model scores well, the tables
look clean, and nothing says otherwise. These gates are the only thing between
that and the leaderboard, so `run` refuses to start while any of them fails.
"""

from __future__ import annotations

import io
from collections import Counter, defaultdict

from PIL import Image

from src import claims, config, labgen, llm, matcher, probes, prompts
from src.claims import FINDINGS

Check = tuple[str, bool, str]


def _balance(scored: list[dict]) -> Check:
    per: dict[str, Counter] = defaultdict(Counter)
    for probe in scored:
        per[probe["finding"]][probe["answer"]] += 1
    off = {f: dict(c) for f, c in per.items() if c["yes"] != c["no"]}
    yes = sum(p["answer"] == "yes" for p in scored)
    return ("balanced per finding, 50% overall", not off and yes * 2 == len(scored),
            f"{len(per)} findings, {yes}/{len(scored)} yes" + (f", unbalanced: {off}" if off else ""))


def _template_identity(scored: list[dict]) -> Check:
    texts: dict[str, set[str]] = defaultdict(set)
    for probe in scored:
        texts[probe["finding"]].add(prompts.verify(probe["phrase"]))
    bad = [f for f, variants in texts.items()
           if variants != {prompts.verify(FINDINGS[f].phrase)}]
    return ("question text identical across a finding's yes and no probes", not bad,
            "one template per finding" if not bad else f"varies for {bad}")


def _provenance(scored: list[dict]) -> Check:
    bad = [p["pid"] for p in scored
           if p.get("source") not in ("assertion", "negation", "stated_normal") or not p.get("sentence")]
    normals = sum(p.get("source") == "stated_normal" for p in scored)
    return ("every no is an explicit negation or stated normal, never silence", not bad,
            f"{normals} stated-normal, 0 from silence" if not bad else f"unsourced: {bad}")


def _no_comparison(scored: list[dict]) -> Check:
    bad = [p["pid"] for p in scored if claims.is_comparative(p["sentence"])]
    return ("no probe sentence needs a prior study", not bad, "clean" if not bad else f"comparative: {bad}")


def _one_per_study(scored: list[dict]) -> Check:
    counts = Counter(p["uid"] for p in scored)
    dupes = [uid for uid, n in counts.items() if n > 1]
    return ("one probe per study", not dupes, f"{len(counts)} studies" if not dupes else f"repeated: {dupes}")


def _images(scored: list[dict]) -> Check:
    missing, sizes = [], set()
    for probe in scored:
        path = config.CASES_DIR / probe["case_id"] / "frontal.png"
        if not path.exists():
            missing.append(probe["case_id"])
            continue
        sizes.add(max(Image.open(io.BytesIO(path.read_bytes())).size))
    return ("every probe image exists and is sized for the model", not missing and sizes <= {config.IMAGE_EDGE},
            f"long edge {sorted(sizes)}" if not missing else f"missing: {missing}")


def _hand_checked(scored: list[dict]) -> Check:
    unchecked = [p["pid"] for p in scored if not p.get("checked_at")]
    return ("hand check complete", not unchecked,
            "all stamped" if not unchecked else f"{len(unchecked)} unchecked - run probes --review, then --approve")


def _lab_unguessable() -> Check:
    """Each minted value appears exactly once in its page's text, with three or more digits."""
    import pymupdf

    problems: list[str] = []
    truth = labgen.load_truth()
    for case_id, report in truth.items():
        text = pymupdf.open(config.CASES_DIR / case_id / "labs.pdf")[0].get_text()
        ranges = " ".join(t["range"] for t in report["tests"])
        for test in report["tests"]:
            hits = matcher.count_occurrences(test["result"], text)
            if hits != 1 or len(test["result"].replace(".", "")) < 3 or matcher.matches(test["result"], ranges):
                problems.append(f"{case_id}:{test['name']}={test['result']}x{hits}")
    return ("every lab value appears once and is unguessable", not problems,
            f"{sum(len(r['tests']) for r in truth.values())} values over {len(truth)} pages"
            if not problems else "; ".join(problems[:5]))


def _models() -> Check:
    try:
        found = {m: llm.digest(m)[:19] for m in (config.SUBJECT_MODEL, config.TWIN_MODEL)}
        return ("subject and twin are pulled", True, str(found))
    except SystemExit as exc:
        return ("subject and twin are pulled", False, str(exc).splitlines()[0])


def run_all() -> list[Check]:
    """All gates, in the order a reader should think about them."""
    scored = probes.load()["probes"]
    checks = [_balance(scored), _template_identity(scored), _provenance(scored), _no_comparison(scored),
              _one_per_study(scored), _images(scored), _hand_checked(scored), _lab_unguessable(), _models()]
    stamped = probes.digest()
    checks.append(("probe set is stamped", stamped.startswith("sha256:"), stamped[:26] + "..."))
    return checks


def failures(checks: list[Check]) -> list[str]:
    return [f"{name}: {detail}" for name, ok, detail in checks if not ok]


if __name__ == "__main__":
    for name, ok, detail in run_all():
        print(f"{'ok ' if ok else 'NO '} {name:62s} {detail}")
