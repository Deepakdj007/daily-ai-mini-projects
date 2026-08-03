"""Formatting and safe-access helpers for the yfinance data layer.

Indian listings report in rupees, and market caps run to thirteen digits, so
raw numbers are unreadable. Everything here turns a possibly-missing value into
a short human string — and never raises, because a missing field is normal.

Inputs:  raw values pulled off yfinance objects (floats, None, NaN, DataFrames).
Outputs: short display strings, and safe getters used by src/market.py.
"""

from typing import Any

import pandas as pd

CRORE = 1e7  # 10 million — the unit Indian financial media actually uses
LAKH_CRORE = 1e12

NA = "n/a"


def is_missing(value: Any) -> bool:
    """True for None, NaN, and pandas' own null markers."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def money(value: Any, currency: str = "INR") -> str:
    """Format a large absolute amount in crore / lakh crore.

    A market cap of 17849330434048 is unreadable; "Rs 17.85 lakh crore" is not.
    Non-rupee listings fall back to plain millions/billions.
    """
    if is_missing(value):
        return NA
    value = float(value)
    sign = "-" if value < 0 else ""
    value = abs(value)
    if currency == "INR":
        if value >= LAKH_CRORE:
            return f"{sign}Rs {value / LAKH_CRORE:,.2f} lakh crore"
        if value >= CRORE:
            return f"{sign}Rs {value / CRORE:,.0f} crore"
        return f"{sign}Rs {value:,.0f}"
    if value >= 1e9:
        return f"{sign}{currency} {value / 1e9:,.2f}B"
    if value >= 1e6:
        return f"{sign}{currency} {value / 1e6:,.2f}M"
    return f"{sign}{currency} {value:,.0f}"


def price(value: Any, currency: str = "INR") -> str:
    """Format a per-share price."""
    if is_missing(value):
        return NA
    unit = "Rs" if currency == "INR" else currency
    return f"{unit} {float(value):,.2f}"


def pct(value: Any, already_percent: bool = False) -> str:
    """Format a ratio as a percentage.

    yfinance is inconsistent here: profitMargins is a fraction (0.066) while
    dividendYield already arrives as a percent (0.46). Pass already_percent
    for the latter kind.
    """
    if is_missing(value):
        return NA
    number = float(value) if already_percent else float(value) * 100
    return f"{number:+.2f}%"


def ratio(value: Any, digits: int = 2) -> str:
    """Format a plain multiple such as P/E or price-to-book."""
    if is_missing(value):
        return NA
    return f"{float(value):,.{digits}f}"


def line(label: str, value: Any) -> str:
    """One aligned "label: value" row inside a data block.

    Fixed-width labels keep the blocks readable for the model and for anyone
    eyeballing the smoke test.
    """
    return f"  {label + ':':<21}{value}"


def row(frame: Any, label: str, column: int = 0) -> Any:
    """Read one line item out of a yfinance statement DataFrame.

    Statements come back with line items as the index and fiscal periods as
    columns, newest first. Any missing frame, missing row, or short frame gives
    None rather than an exception.

    Args:
        frame: income_stmt / balance_sheet / cashflow, or None.
        label: exact row label, e.g. "Total Revenue".
        column: 0 is the most recent fiscal year, 1 the one before.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    if label not in frame.index:
        return None
    if column >= len(frame.columns):
        return None
    value = frame.loc[label].iloc[column]
    return None if is_missing(value) else value


def growth(new: Any, old: Any) -> str:
    """Percentage change between two statement values."""
    if is_missing(new) or is_missing(old) or float(old) == 0:
        return NA
    return f"{(float(new) - float(old)) / abs(float(old)) * 100:+.1f}%"


def fiscal_years(frame: Any, count: int = 3) -> list[str]:
    """Label the most recent fiscal periods of a statement, newest first."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return []
    return [str(c)[:10] for c in frame.columns[:count]]
