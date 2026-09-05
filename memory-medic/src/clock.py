"""The only module allowed to ask what time it is.

Inputs:  an optional ISO date to freeze at
Outputs: Clock objects that hand out integer UTC epoch seconds

Every function that stamps or compares a time takes a clock. That is not
ceremony: the whole point of this project is reasoning about facts at a date,
and an evaluation that cannot move the clock cannot test decay at all. One
stray datetime.now() deep in the store would silently pin a probe to today and
the arm measuring 400-day-old facts would quietly measure nothing.

`main.py check` greps for that mistake.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Protocol


def to_epoch(iso: str) -> int:
    """'2026-06-10' or '2026-06-10T09:12:00' -> integer UTC epoch seconds."""
    text = iso.strip().replace("Z", "+00:00")
    if len(text) == 10:
        parsed = datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    else:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp())


def to_iso(epoch: int, *, date_only: bool = True) -> str:
    """Epoch seconds -> ISO text. Dates read better in a card than timestamps."""
    moment = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return moment.date().isoformat() if date_only else moment.isoformat()


def days_between(later: int, earlier: int) -> float:
    """Fractional days from `earlier` to `later`. Negative when out of order."""
    return (later - earlier) / 86_400.0


class Clock(Protocol):
    """Anything that can say what 'now' is, in epoch seconds."""

    def now(self) -> int:
        ...

    def today(self) -> str:
        ...


class SystemClock:
    """Wall-clock time. Used by the CLI and the inbox."""

    frozen = False

    def now(self) -> int:
        return int(time.time())

    def today(self) -> str:
        return to_iso(self.now())


class FrozenClock:
    """A clock the caller moves by hand.

    This is what makes the silent-decay demo possible in a single session:
    seed in June, `advance(days=180)`, sweep, and watch a fast-moving fact go
    stale without anybody having waited six months.
    """

    frozen = True

    def __init__(self, at: str | int) -> None:
        self._epoch = at if isinstance(at, int) else to_epoch(at)

    def now(self) -> int:
        return self._epoch

    def today(self) -> str:
        return to_iso(self._epoch)

    def set(self, at: str | int) -> None:
        """Jump to an absolute moment."""
        self._epoch = at if isinstance(at, int) else to_epoch(at)

    def advance(self, *, days: float = 0, seconds: int = 0) -> None:
        """Move forward. Refuses to go backwards: bitemporal writes assume it."""
        delta = int(timedelta(days=days, seconds=seconds).total_seconds())
        if delta < 0:
            raise ValueError("a clock that runs backwards breaks the valid-time chain")
        self._epoch += delta


def make_clock(at: str = "") -> Clock:
    """FrozenClock when an ISO date is given, SystemClock otherwise."""
    return FrozenClock(at) if at.strip() else SystemClock()


if __name__ == "__main__":
    assert to_iso(to_epoch("2026-06-10")) == "2026-06-10"
    assert to_epoch("2026-06-11") - to_epoch("2026-06-10") == 86_400

    clock = FrozenClock("2026-06-10")
    assert clock.today() == "2026-06-10"
    clock.advance(days=180)
    assert clock.today() == "2026-12-07", clock.today()
    assert abs(days_between(clock.now(), to_epoch("2026-06-10")) - 180) < 1e-6

    try:
        clock.advance(days=-1)
        raise AssertionError("backwards must raise")
    except ValueError:
        pass

    live = SystemClock()
    assert live.now() > to_epoch("2026-01-01")
    print(f"OK - frozen at {clock.today()}, system says {live.today()}")
    print(f"     {date.today().isoformat()} is what the wall clock thinks")
