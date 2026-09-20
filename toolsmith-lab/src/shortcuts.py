"""Wrong implementations, on purpose. Nothing imports these except the gates.

Inputs:  the same arguments the oracle takes
Outputs: the answer a particular kind of mistake produces

Two kinds, doing two different jobs.

TRIVIAL is what somebody who does no computation at all would say - echo the
last character, apply simple interest, sort the strings. A family where a
trivial answer is right is a family that measures nothing, so a gate requires
each of these to be wrong on nearly every visible task.

OVERFIT is the sharper one. Each of these is correct on all four of a family's
visible inputs and wrong on at least one held-out input. That is the whole
argument for running held-out inputs before registering a tool: a model's own
tests are written from what it was shown, so an implementation shaped like one
of these passes its own tests and ships broken. A gate requires each family to
have one, because a held-out check that nothing can fail is decoration.

These are deliberately not written by patching the reference implementations.
They are written out as somebody would actually write them, mistake included,
because a stub built by breaking a correct function tests the break rather than
the mistake.
"""

from __future__ import annotations

from datetime import date
from decimal import ROUND_HALF_UP, Decimal

from src import ref_ids, ref_misc, ref_money

B36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --- TRIVIAL: no computation at all ------------------------------------------

def _trivial_gstin(prefix: str) -> str:
    """Echo the last character of what you were given."""
    return prefix[-1]


def _trivial_loan(principal: float, annual_rate: float, months: int) -> float:
    """Simple interest, which is the number people reach for."""
    return round(principal * annual_rate / 100.0 * months / 12.0, 2)


def _trivial_isoweek(day: str) -> str:
    """Day of the year divided by seven, with the calendar year."""
    parsed = date.fromisoformat(day)
    return (f"{parsed.year:04d}-W{(parsed.timetuple().tm_yday - 1) // 7 + 1:02d}"
            f"-{parsed.isoweekday()}")


def _trivial_verhoeff(prefix: str) -> int:
    """Echo the last digit."""
    return int(prefix[-1])


def _trivial_words(amount: int) -> str:
    """Western grouping: thousand, million, billion."""
    if amount == 0:
        return "zero"
    scales = ((1_000_000_000, "billion"), (1_000_000, "million"), (1_000, "thousand"))
    parts = [f"{amount // size % 1000} {name}" for size, name in scales if amount // size]
    if amount % 1000:
        parts.append(str(amount % 1000))
    return " ".join(parts)


def _trivial_semver(versions: list[str]) -> str:
    """Sort them as strings, which is what a lexical sort does."""
    return max(versions)


# --- OVERFIT: right on everything it was shown -------------------------------

def _overfit_gstin(prefix: str) -> str:
    """Correct arithmetic, but assumes a check character is always a letter.

    Held-out prefixes whose check value lands below ten come back as a letter,
    so every one of those is wrong while the shown cases stay right.
    """
    total = 0
    for position, char in enumerate(prefix.upper()):
        product = B36.index(char) * (1 if position % 2 == 0 else 2)
        total += product // 36 + product % 36
    index = (36 - total % 36) % 36
    return B36[index] if index >= 10 else B36[10 + index]


def _overfit_loan(principal: float, annual_rate: float, months: int) -> float:
    """The whole schedule, correctly - with no branch for a 0% loan.

    The formula is 0/0 at a zero rate, so this raises on an interest-free loan
    and is exactly right on every loan that charges interest.
    """
    rate = annual_rate / 12.0 / 100.0
    growth = (1.0 + rate) ** months
    quantum = Decimal("0.01")
    instalment = float(Decimal(str(principal * rate * growth / (growth - 1.0)))
                       .quantize(quantum, rounding=ROUND_HALF_UP))
    balance, paid = float(principal), 0.0
    for month in range(1, months + 1):
        charged = float(Decimal(str(balance * rate)).quantize(quantum, rounding=ROUND_HALF_UP))
        due = instalment if month < months else float(
            Decimal(str(balance + charged)).quantize(quantum, rounding=ROUND_HALF_UP))
        balance = float(Decimal(str(balance - (due - charged)))
                        .quantize(quantum, rounding=ROUND_HALF_UP))
        paid += charged
    return float(Decimal(str(paid)).quantize(quantum, rounding=ROUND_HALF_UP))


def _overfit_isoweek(day: str) -> str:
    """The right week and weekday, stamped with the calendar year.

    Correct for all but a handful of days a year, and those days are precisely
    the ones nobody puts in an example.
    """
    parsed = date.fromisoformat(day)
    _, week, weekday = parsed.isocalendar()
    return f"{parsed.year:04d}-W{week:02d}-{weekday}"


def _overfit_verhoeff(prefix: str) -> int:
    """The right tables, reached through an int() that eats leading zeros."""
    return ref_ids.verhoeff_check_digit(str(int(prefix)))


def _overfit_words(amount: int) -> str:
    """Every segment spoken, including the ones worth nothing."""
    if amount == 0:
        return ""
    segments = ((amount // 10_000_000, "crore"),
                (amount // 100_000 % 100, "lakh"),
                (amount // 1_000 % 100, "thousand"))
    highest = next((index for index, (count, _) in enumerate(segments) if count), len(segments))
    words = [f"{ref_money._under_thousand(count) or 'zero'} {label}"
             for count, label in segments[highest:]]
    if amount % 1_000:
        words.append(ref_money._under_thousand(amount % 1_000))
    return " ".join(words)


def _overfit_semver(versions: list[str]) -> str:
    """Correct precedence, but build metadata is never stripped.

    A version carrying a +build suffix does not match the pattern, so this
    raises on it - and no visible task has one.
    """
    if any("+" in version for version in versions):
        raise ValueError("unparseable version")
    return ref_misc.semver_max(versions)


def _batched(fn, sep: str = ""):
    """Apply a scalar stub across a batch, the way the oracle does.

    Five of the six families ask for the rule applied to a list, so a stub that
    only handles one item would fail every task for the wrong reason - and a
    gate comparing it against the oracle would be comparing shapes, not rules.
    """
    def over(items: list) -> str:
        return sep.join(str(fn(item)) for item in items)
    return over


def _trivial_semver_batch(versions: list[str]) -> str:
    """A plain string sort, which is what it looks like semver ordering is."""
    return ", ".join(sorted(versions))


def _overfit_semver_batch(versions: list[str]) -> str:
    """Correct precedence, but build metadata is never stripped."""
    if any("+" in version for version in versions):
        raise ValueError("unparseable version")
    return ", ".join(ref_misc.semver_sort(versions))


SCALAR_OVERFIT = {
    "gstin": _overfit_gstin, "verhoeff": _overfit_verhoeff,
    "isoweek": _overfit_isoweek, "words": _overfit_words,
}
"""The per-item stubs, used by the corpus builder to decide which items are
ordinary enough to show."""

TRIVIAL = {
    "gstin": _batched(_trivial_gstin),
    "loan": _trivial_loan,
    "isoweek": _batched(_trivial_isoweek, " "),
    "verhoeff": _batched(_trivial_verhoeff),
    "words": _batched(_trivial_words, "; "),
    "semver": _trivial_semver_batch,
}

OVERFIT = {
    "gstin": _batched(_overfit_gstin),
    "loan": _overfit_loan,
    "isoweek": _batched(_overfit_isoweek, " "),
    "verhoeff": _batched(_overfit_verhoeff),
    "words": _batched(_overfit_words, "; "),
    "semver": _overfit_semver_batch,
}


if __name__ == "__main__":
    from src import oracle

    for key in oracle.BY_KEY:
        assert key in TRIVIAL and key in OVERFIT, f"{key} has no stubs"
    print(f"{len(TRIVIAL)} trivial and {len(OVERFIT)} overfit stubs")
    print("whether each has the property it needs is src/gates.py's question")
