"""Decide whether a reply actually states a fact, without asking an LLM.

Inputs:  a gold value and a model's reply
Outputs: one verdict per fact

Substring matching is the trap this file exists to avoid. `gold in reply`
scores "42" as present inside "142", which quietly rewards whichever model
rambles the most - exactly the arm that knows nothing and is padding. Every
match here is anchored on word boundaries built from the gold value.

Two things differ from a normal matcher, and both are here because the model
being graded is a chat model with no instruction to answer in any format.

There is no "Answer:" line to read, so the whole reply is scanned. That is
more generous, and the generosity runs toward the fine-tune, which is the
right direction when the expected result is that the fine-tune fails.

A reply that names half the fact set has not recalled anything, it has
emptied its pockets. That is counted separately rather than scored as a hit.

The boundary logic, the money folding and the smart-punctuation table are
taken from second-read/src/matcher.py, which earned them.
"""

from __future__ import annotations

import re
import unicodedata

# Models emit typographic punctuation, and an ASCII apostrophe in a pattern
# then silently never matches - a false negative that looks like a real miss.
_SMART = {
    0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"',
    0x2013: "-", 0x2014: "-", 0x2212: "-",
    0x00A0: " ", 0x202F: " ", 0x2009: " ",
}

# "na" is deliberately absent: in Hinglish it is an affirmative tag, as in
# "haan na", and treating it as a negation would void correct answers.
NEGATIONS = frozenset({
    "not", "no", "never", "dont", "doesnt", "didnt", "isnt", "wasnt",
    "cant", "cannot", "nahi", "nahin", "without", "unsure", "unknown",
})
NEGATION_WINDOW = 4
SHOTGUN_THRESHOLD = 3

HIT, MISS, NEGATED, SHOTGUN = "hit", "miss", "negated", "shotgun"


def normalize(text: str) -> str:
    """Fold typography, money formatting and whitespace.

    Digit-group commas and the rupee sign come off because the same amount is
    written "Rs 18,500", "18500" and "18,500" from one reply to the next, and
    none of those differences is a wrong answer. The comma goes only between
    digits, so "Indiranagar, Bangalore" keeps its comma.
    """
    folded = unicodedata.normalize("NFKC", text or "").translate(_SMART)
    folded = folded.replace("₹", " ")
    folded = re.sub(r"(?<=\d),(?=\d)", "", folded)
    return re.sub(r"\s+", " ", folded).strip()


def gold_pattern(gold: str) -> re.Pattern[str]:
    """A boundary-anchored pattern for one gold value.

    Separators live inside the pattern rather than being stripped from the
    text: stripping them turns 6E 2134 into 6e2134, and then no boundary is
    left to stop x6e2134y matching.
    """
    parts = [re.escape(part) for part in re.split(r"[\s\-_/.]+", normalize(gold)) if part]
    body = r"[\s\-_/.]*".join(parts) if parts else re.escape(normalize(gold))
    return re.compile(r"(?<![0-9A-Za-z])" + body + r"(?![0-9A-Za-z])", re.IGNORECASE)


def matches(gold: str, text: str) -> bool:
    """Does `text` actually state `gold`?"""
    return bool(gold) and bool(gold_pattern(gold).search(normalize(text)))


def count_occurrences(gold: str, text: str) -> int:
    """How often `gold` appears - used to count a fact in the training split."""
    return len(gold_pattern(gold).findall(normalize(text))) if gold else 0


def any_match(golds: list[str], text: str) -> bool:
    """True if the gold or any of its accepted spellings appears."""
    return any(matches(gold, text) for gold in golds if gold)


def negated(gold: str, text: str) -> bool:
    """Is the gold stated only to be denied - "I do not have a cat"?"""
    flat = normalize(text)
    found = gold_pattern(gold).search(flat)
    if not found:
        return False
    before = re.findall(r"[a-z']+", flat[:found.start()].lower())
    window = [re.sub(r"[^a-z]", "", word) for word in before[-NEGATION_WINDOW:]]
    return any(word in NEGATIONS for word in window)


def grade(golds: list[str], reply: str, other_golds: list[str]) -> str:
    """One verdict for one fact against one reply."""
    if not any_match(golds, reply):
        return MISS
    if any(negated(gold, reply) for gold in golds if matches(gold, reply)):
        return NEGATED
    sprayed = sum(1 for other in other_golds if matches(other, reply))
    return SHOTGUN if sprayed >= SHOTGUN_THRESHOLD else HIT


if __name__ == "__main__":
    checks = [
        # The bug this file exists to prevent.
        ("substring does not count", not matches("42", "it was 142")),
        ("no inside Now", not matches("no", "Now it works")),
        ("exact number counts", matches("2400", "I sent you 2400 yesterday")),
        ("longer number does not", not matches("2400", "that came to 124009")),

        # Separators, money and typography, from second-read's matcher.
        ("flight code any spelling", all(
            matches("6E 2134", f"it is {s}") for s in ("6E2134", "6e 2134", "6E-2134"))),
        ("rent any spelling", all(
            matches("18500", f"rent is {s}") for s in ("18,500", "Rs 18,500", "₹18500"))),
        ("bigger amount is not the rent", not matches("18500", "that is 118500")),
        ("curly apostrophe folds", matches("Arjun's", "that is Arjun’s cat")),
        ("non-breaking space folds", matches("Access 125", "he rides an Access\u00a0125")),
        ("a dotted name matches", matches("Mr. Raghavan", "the owner is Mr. Raghavan")),

        # Aliases.
        ("alias accepted", any_match(["Biryani", "Biriyani"], "her name is Biriyani")),
        ("no alias matches", not any_match(["Biryani", "Biriyani"], "her name is Rani")),

        # Negation: saying the word to deny it is not recall.
        ("denial is caught", negated("cat", "I do not have a cat")),
        ("hinglish denial is caught", negated("Biryani", "uske paas nahi hai Biryani")),
        ("plain statement is not denial", not negated("Biryani", "her name is Biryani")),
        ("distant negation is ignored",
            not negated("Biryani", "no idea about anything else but her name is Biryani")),

        # Grades.
        ("a plain hit", grade(["Biryani"], "she is called Biryani", ["Access 125"]) == HIT),
        ("a plain miss", grade(["Biryani"], "I think her name is Rani", []) == MISS),
        ("a denial grades as negated",
            grade(["Biryani"], "there is no Biryani here", []) == NEGATED),
        ("emptying the pockets is not recall",
            grade(["Biryani"], "Biryani, Access 125, Mr. Raghavan, 18500 - one of those",
                  ["Access 125", "Mr. Raghavan", "18500"]) == SHOTGUN),
    ]
    for name, ok in checks:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")
    raise SystemExit(0 if all(ok for _, ok in checks) else 1)
