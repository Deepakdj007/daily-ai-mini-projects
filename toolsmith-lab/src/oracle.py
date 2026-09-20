"""The six families, their reference answers, and the contract each one states.

Inputs:  a family key and that family's arguments
Outputs: the gold answer, and the metadata the corpus and the gates are built on

This module is the grader, so it is also part of the experiment: `digest()`
covers the three reference files as well as this one, and the results manifest
records it. Editing a reference implementation after a run silently rewrites
what "correct" meant, and the digest is what makes that visible.

About the contracts. Each family states the rule precisely enough to pin down
WHICH algorithm is wanted and the exact shape of the answer, and no further. It
does not spell out the steps. That line matters: transcribing pseudocode is not
writing a tool, and a family whose contract is a full algorithm leaves the
model nothing to get subtly wrong - which would leave the held-out inputs with
nothing to catch and the rung above them measuring air.

The loan family is the exception, and deliberately so. Its rounding convention
IS the specification - two correct implementations that round differently give
different answers - so it is stated exactly. The work left to the model there
is walking the schedule, which is the part a closed formula cannot do.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src import ref_ids, ref_misc, ref_money

_SOURCES = ("oracle.py", "ref_ids.py", "ref_misc.py", "ref_money.py")


# Five of the six families ask for the rule applied to a BATCH, and that is the
# load-bearing decision in the whole corpus.
#
# It was scalar first, and the null arm - no tools at all - answered all four
# GSTIN tasks correctly. A 120B model will happily walk fourteen base-36 digits
# in its head, so a family that asks for one check character measures nothing
# about tools. Ten of them is roughly a hundred and forty steps, every one of
# which has to be right for an exact match, and that is past what a reasoning
# trace can carry.
#
# It is also the honest reason a developer writes a tool. Nobody writes a
# GSTIN validator to check one number; they write it to check a column of them.
# Scale is the thing that makes a tool worth having, so scale is what the
# corpus should be built on.

def _each(fn, sep: str = ""):
    """Apply a scalar rule across a list and join the results."""
    def batched(items: list) -> str:
        return sep.join(str(fn(item)) for item in items)
    return batched


def _gstin_batch(prefixes: list[str]) -> str:
    """Check characters for several GSTIN prefixes, in order, joined."""
    return _each(ref_ids.gstin_check_char)(prefixes)


def _verhoeff_batch(prefixes: list[str]) -> str:
    """Check digits for several numbers, in order, joined."""
    return _each(ref_ids.verhoeff_check_digit)(prefixes)


def _isoweek_batch(days: list[str]) -> str:
    """ISO week labels for several dates, space separated."""
    return _each(ref_misc.iso_week_label, " ")(days)


def _words_batch(amounts: list[int]) -> str:
    """Several amounts in Indian words, separated by semicolons."""
    return _each(ref_money.rupees_in_words, "; ")(amounts)


def _semver_batch(versions: list[str]) -> str:
    """Every version in ascending precedence order, comma separated."""
    return ", ".join(ref_misc.semver_sort(versions))


@dataclass(frozen=True, slots=True)
class Family:
    """One family: the rule, the interface, and the answer to judge against."""

    key: str
    label: str
    fn: Callable[..., Any]
    params: tuple[tuple[str, str], ...]
    """(name, JSON type) in the order the held-out runner binds them. The names
    are ours; a generated tool may call its arguments anything, so binding is
    positional against whatever schema the tool declares."""
    compare: str
    returns: str
    contract: str
    why_tool: str
    example: tuple
    """One worked input/output pair, stated in the contract of every task in
    the family. It is there because without it the model cannot write a test:
    asked for a GSTIN check character it cannot work one out by hand, so it
    guesses an expected value, its own test disagrees with its own correct
    code, and the rung that runs those tests rejects good tools all day. A real
    specification ships an example for exactly this reason.

    It is disjoint from every visible input and every held-out input, and it is
    deliberately an easy case: it gives the model something true to test
    against without showing it any of the edges the held-out inputs probe."""
    example_line: str
    """How the worked example reads, with {0}.. for inputs and {ans} for the
    answer."""

    @property
    def arity(self) -> int:
        return len(self.params)

    def full_contract(self) -> str:
        """The contract as the model sees it: the rule, then a worked example."""
        answer = self.fn(*self.example)
        shown = [", ".join(str(v) for v in value) if isinstance(value, list) else value
                 for value in self.example]
        return f"{self.contract} {self.example_line.format(*shown, ans=answer)}"


FAMILIES: tuple[Family, ...] = (
    Family(
        key="gstin",
        label="GSTIN check character",
        fn=_gstin_batch,
        params=(("prefixes", "array"),),
        compare="str_ci",
        returns="the check characters in order, joined into one string with nothing between them",
        contract=(
            "A GSTIN is 15 characters long. The last one is a check character "
            "derived from the first 14 by the standard GSTIN checksum: each "
            "character is read as a base-36 digit (0-9 then A-Z), weighted "
            "alternately by 1 and 2 starting with 1 at the first character, and "
            "the digits of each weighted product are folded back modulo 36. You are "
            "given several prefixes and must return every check character, in "
            "the order the prefixes are given, with nothing between them."
        ),
        why_tool=(
            "Base-36 conversion, an alternating weight and a modular fold. There "
            "is no arithmetic expression over the input that produces it."
        ),
        example=(["22AAAAA0000A1Z", "07AABCS1429B1Z"],),
        example_line="As a worked example, {0} give \"{ans}\".",
    ),
    Family(
        key="loan",
        label="total interest on a loan",
        fn=ref_money.total_interest,
        params=(("principal", "number"), ("annual_rate", "number"), ("months", "integer")),
        compare="money2",
        returns="the total interest in rupees, to two decimals",
        contract=(
            "For a reducing-balance loan, the monthly rate is the annual rate "
            "divided by 12 and by 100, and the instalment is the standard EMI "
            "formula rounded half-up to two decimals. Walk the schedule: each "
            "month the interest is the opening balance times the monthly rate, "
            "rounded half-up to two decimals, and the balance falls by the "
            "instalment less that interest, again to two decimals. The final "
            "instalment is whatever clears the balance rather than the fixed "
            "figure. The answer is the sum of every month's interest."
        ),
        why_tool=(
            "Rounding each month to paise breaks the closed form, so the total "
            "can only be reached by walking every row of the schedule."
        ),
        example=(200000, 10.0, 24),
        example_line="As a worked example, Rs {0} at {1}% over {2} months pays {ans} in interest.",
    ),
    Family(
        key="isoweek",
        label="ISO week label",
        fn=_isoweek_batch,
        params=(("days", "array"),),
        compare="str_ci",
        returns='the labels in order, separated by single spaces',
        contract=(
            "Report the ISO-8601 week-numbering year, week number and weekday of "
            "a date as YYYY-Www-D, with the week zero-padded to two digits and "
            "the weekday running 1 for Monday to 7 for Sunday. The year in the "
            "label is the ISO week-numbering year, which near the turn of the "
            "year is not always the calendar year. You are given several dates and "
            "must return every label, in order, separated by single spaces."
        ),
        why_tool=(
            "ISO week 1 is the week holding the first Thursday, so the year "
            "boundary does not follow from the day of the year."
        ),
        example=(["2024-06-12", "2024-11-03"],),
        example_line="As a worked example, {0} give \"{ans}\".",
    ),
    Family(
        key="verhoeff",
        label="Verhoeff check digit",
        fn=_verhoeff_batch,
        params=(("prefixes", "array"),),
        compare="exact",
        returns="the check digits in order, joined into one string of digits",
        contract=(
            "Indian 12-digit identity numbers carry a Verhoeff check digit in the "
            "last position, computed over the preceding digits using the "
            "dihedral group D5. Given the digits without the check digit, return "
            "the check digit that makes the whole number validate. You are given "
            "several numbers and must return every check digit, in the order the "
            "numbers are given, with nothing between them."
        ),
        why_tool=(
            "Verhoeff is three lookup tables. Nothing about it can be derived, "
            "and a table recalled slightly wrong still validates its own examples."
        ),
        example=(["13579246801", "24680135791"],),
        example_line="As a worked example, {0} give \"{ans}\".",
    ),
    Family(
        key="words",
        label="rupees in Indian words",
        fn=_words_batch,
        params=(("amounts", "array"),),
        compare="words",
        returns="the amounts in words, in order, separated by semicolons",
        contract=(
            "Write a whole rupee amount in words on the Indian system, using "
            "crore, lakh and thousand, then the remainder below a thousand. Use "
            "lowercase, no punctuation, and do not write the word rupees. A "
            "segment worth zero is not spoken at all. Zero itself is \"zero\". You "
            "are given several amounts and must return every one in words, in "
            "order, separated by a semicolon and a space."
        ),
        why_tool=(
            "Segment-by-segment string construction on a grouping that is not "
            "the thousands grouping every library ships with."
        ),
        example=([105000, 2400],),
        example_line="As a worked example, {0} give \"{ans}\".",
    ),
    Family(
        key="semver",
        label="highest semantic version",
        fn=_semver_batch,
        params=(("versions", "array"),),
        compare="exact",
        returns="every version in ascending order, separated by a comma and a space",
        contract=(
            "Order version strings by Semantic Versioning 2.0.0 precedence and "
            "return the highest. Major, minor and patch compare numerically; a "
            "version carrying a prerelease ranks below the same version without "
            "one; prerelease identifiers compare field by field, numeric ones "
            "numerically and below alphanumeric ones; build metadata is ignored. "
            "Return every version given, sorted from lowest to highest, separated "
            "by a comma and a space."
        ),
        why_tool=(
            "Sorting these as strings is wrong in several places at once and "
            "looks right on any list short enough to read."
        ),
        example=(["2.0.0", "2.0.0-rc.1"],),
        example_line="As a worked example, {0} sort to \"{ans}\".",
    ),
)

BY_KEY: dict[str, Family] = {family.key: family for family in FAMILIES}


def solve(family_key: str, args: dict) -> Any:
    """The gold answer for one set of arguments, bound by the family's names."""
    family = BY_KEY[family_key]
    return family.fn(*(args[name] for name, _ in family.params))


def solve_positional(family_key: str, values: list) -> Any:
    """The gold answer from values already in the family's declared order."""
    family = BY_KEY[family_key]
    if len(values) != family.arity:
        raise ValueError(f"{family_key} takes {family.arity} argument(s), got {len(values)}")
    return family.fn(*values)


def digest() -> str:
    """A sha256 over every file that decides what 'correct' means."""
    here = Path(__file__).resolve().parent
    blob = b"\x1f".join((here / name).read_bytes() for name in _SOURCES)
    return "sha256:" + hashlib.sha256(blob).hexdigest()[:16]


if __name__ == "__main__":
    from src import matcher

    assert len(FAMILIES) == 6 and len(BY_KEY) == 6
    for fam in FAMILIES:
        assert fam.compare in matcher.MODES, f"{fam.key}: bad compare mode"
        assert fam.contract == fam.contract.strip(), f"{fam.key}: contract has loose ends"

    assert solve("gstin", {"prefix": "27AAPFU0939F1Z"}) == "V"
    assert solve("verhoeff", {"prefix": "23456789012"}) == 4
    assert solve("isoweek", {"day": "2027-01-01"}) == "2026-W53-5"
    assert solve("words", {"amount": 10005000}) == "one crore five thousand"
    assert solve("semver", {"versions": ["1.9.0", "1.10.0"]}) == "1.10.0"
    assert matcher.matches(
        solve("loan", {"principal": 1_000_000, "annual_rate": 8.5, "months": 240}),
        1082776.63, "money2")
    assert solve_positional("loan", [120_000, 0.0, 12]) == 0.0

    print(f"{len(FAMILIES)} families, oracle {digest()}")
    for fam in FAMILIES:
        print(f"  {fam.key:<9} {fam.arity} arg(s), compare={fam.compare:<7} {fam.label}")
    print("OK - oracle")
