"""Everything that talks to Yahoo Finance.

This is the access layer: one cached Ticker per symbol, guarded readers, and the
two lookups the CLI needs before any model runs. src/market.py turns what comes
back from here into the text blocks the agents read.

Inputs:  a ticker symbol, e.g. "RELIANCE.NS".
Outputs: ticker() for the shared Ticker object, info()/currency() for guarded
         field access, validate() and profile() for the CLI.
"""

import logging
from functools import lru_cache
from typing import Any

import yfinance as yf

from src import fmt

# yfinance logs "possibly delisted" straight to the root logger, which would
# interleave with the crew's own output. Guarded lookups report failures instead.
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


@lru_cache(maxsize=16)
def ticker(symbol: str) -> yf.Ticker:
    """One Ticker object per symbol, shared by all four agents.

    The analysts run concurrently and their data needs overlap, so caching here
    stops four threads making the same request. lru_cache may compute twice under
    a race, which is harmless — worst case is one duplicate HTTP call.
    """
    return yf.Ticker(symbol)


def info(symbol: str) -> dict[str, Any]:
    """Ticker.info, or an empty dict if Yahoo has nothing.

    166 keys on a good day, 1 key on a dead symbol, an exception on a network
    blip. Callers should never have to care which.
    """
    try:
        return ticker(symbol).info or {}
    except Exception:
        return {}


def currency(symbol: str) -> str:
    """Reporting currency, defaulting to rupees for this project's audience."""
    return info(symbol).get("currency") or "INR"


def validate(symbol: str) -> tuple[bool, str]:
    """Check a ticker actually resolves, before any model call happens.

    A wrong symbol is the most common failure: bare "RELIANCE" returns an empty
    frame because NSE listings need the ".NS" suffix.

    Returns:
        (True, company name) or (False, an explanation the CLI can print).
    """
    try:
        history = ticker(symbol).history(period="5d")
    except Exception as exc:
        return False, f"lookup failed: {type(exc).__name__}: {exc}"
    if history.empty:
        hint = ""
        if "." not in symbol:
            hint = " Indian listings need a suffix — try RELIANCE.NS or RELIANCE.BO."
        return False, f"no price data for '{symbol}'.{hint}"
    return True, info(symbol).get("longName") or symbol


def profile(symbol: str) -> dict[str, str]:
    """Small header block for the report: name, sector, price, market cap."""
    fields = info(symbol)
    unit = currency(symbol)
    return {
        "name": fields.get("longName") or symbol,
        "sector": fields.get("sector") or fmt.NA,
        "industry": fields.get("industry") or fmt.NA,
        "price": fmt.price(fields.get("currentPrice"), unit),
        "market_cap": fmt.money(fields.get("marketCap"), unit),
        "currency": unit,
    }
