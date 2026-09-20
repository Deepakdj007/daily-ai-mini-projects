"""Reference implementations for the two money families.

Inputs:  loan terms, or a rupee amount
Outputs: the interest paid over a loan, and an amount in Indian words

Both families carry a trap that a correct-looking implementation falls into.

A loan at 0% divides by zero in the standard EMI formula, so the interest-free
case has to be written on purpose. A model that is never shown one will not
write it, and the held-out inputs contain one.

Indian words are not "thousand, million, billion", and the segment that breaks
naive code is an empty middle one: 1,00,05,000 has no lakh digits to speak, and
an implementation that joins segments unconditionally says "one crore zero lakh
five thousand".

Why the interest total and not the EMI itself: the EMI is a single arithmetic
expression, so the fixed toolset's calculator could answer that family in one
call and the baseline would stop being a floor. Rounding every month's interest
to paise breaks the closed form, and the answer can then only be reached by
walking the schedule.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

_ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety")


def paise(value: Decimal | float | int) -> float:
    """Round to two decimals the way a bank statement does, not the way floats do.

    Decimal with ROUND_HALF_UP, because Python's round() is half-to-even:
    round(0.125, 2) is 0.12, and a schedule built on that drifts away from the
    figure on the statement a few hundred rows in.
    """
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def emi(principal: float, annual_rate: float, months: int) -> float:
    """The equated monthly instalment, rounded to paise.

    At a 0% rate the standard formula is 0/0, so the interest-free loan is its
    own branch: the principal divided evenly across the tenure.
    """
    if months <= 0:
        raise ValueError("months must be positive")
    rate = annual_rate / 12.0 / 100.0
    if rate == 0:
        return paise(principal / months)
    growth = (1.0 + rate) ** months
    return paise(principal * rate * growth / (growth - 1.0))


def total_interest(principal: float, annual_rate: float, months: int) -> float:
    """Every rupee of interest paid over the life of the loan.

    Each month's interest is the opening balance times the monthly rate,
    rounded to paise. The instalment is fixed except for the last one, which is
    whatever clears the balance - so the schedule closes at exactly zero
    instead of leaving a few paise behind.
    """
    instalment = emi(principal, annual_rate, months)
    rate = annual_rate / 12.0 / 100.0
    balance, paid = float(principal), 0.0
    for month in range(1, months + 1):
        charged = paise(balance * rate)
        due = instalment if month < months else paise(balance + charged)
        balance = paise(balance - (due - charged))
        paid += charged
    return paise(paid)


def _under_thousand(value: int) -> str:
    """Words for 1-999, without the word 'zero'."""
    words: list[str] = []
    if value >= 100:
        words += [_ONES[value // 100], "hundred"]
        value %= 100
    if value >= 20:
        words.append(_TENS[value // 10])
        value %= 10
    if value:
        words.append(_ONES[value])
    return " ".join(words)


def rupees_in_words(amount: int) -> str:
    """An amount in Indian words: crore, lakh, thousand, then the remainder.

    A segment worth zero contributes nothing at all. That is the whole
    difference between this and a naive implementation, and the reason the
    held-out inputs include an amount with an empty lakh segment.
    """
    value = int(amount)
    if value == 0:
        return "zero"
    prefix = "minus " if value < 0 else ""
    value = abs(value)
    segments = ((value // 10_000_000, "crore"),
                (value // 100_000 % 100, "lakh"),
                (value // 1_000 % 100, "thousand"))
    words = [f"{_under_thousand(count)} {label}" for count, label in segments if count]
    remainder = value % 1_000
    if remainder:
        words.append(_under_thousand(remainder))
    return prefix + " ".join(words)


if __name__ == "__main__":
    assert emi(1_000_000, 8.5, 240) == 8678.23, emi(1_000_000, 8.5, 240)
    assert emi(120_000, 0.0, 12) == 10000.00, emi(120_000, 0.0, 12)
    assert total_interest(120_000, 0.0, 12) == 0.0, total_interest(120_000, 0.0, 12)
    # A one-month loan pays exactly one month of interest and nothing more.
    assert total_interest(100_000, 12.0, 1) == 1000.0, total_interest(100_000, 12.0, 1)
    twenty_year = total_interest(1_000_000, 8.5, 240)
    assert 1_082_000 < twenty_year < 1_083_000, twenty_year

    assert rupees_in_words(0) == "zero"
    assert rupees_in_words(10005000) == "one crore five thousand", rupees_in_words(10005000)
    assert rupees_in_words(100000) == "one lakh"
    assert rupees_in_words(999999999) == (
        "ninety nine crore ninety nine lakh ninety nine thousand nine hundred ninety nine"
    ), rupees_in_words(999999999)
    print(f"10,00,000 at 8.5% over 240 months -> interest {twenty_year:,.2f}")
    print(f"1,20,000 interest-free over 12    -> interest {total_interest(120_000, 0.0, 12)}")
    print(f"1,00,05,000                       -> {rupees_in_words(10005000)}")
    print("OK - ref_money")
