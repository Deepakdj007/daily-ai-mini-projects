"""Reference implementations for the calendar and version families.

Inputs:  an ISO date, or a list of semver strings
Outputs: an ISO week label, and semver precedence ordering

Both families are things that look solved and are not.

An ISO week is not the day of the year divided by seven. Week 1 is the week
containing the first Thursday, so 2027-01-01 belongs to week 53 of 2026 and the
ISO year is not the calendar year. Any arithmetic shortcut is right for about
fifty weeks a year, and the held-out inputs sit on exactly the other two.

Semver precedence is not a string sort. A lexical sort puts alpha.10 before
alpha.9 and puts 1.0.0-alpha above 1.0.0, and it looks correct on any example
list short enough to eyeball.
"""

from __future__ import annotations

import re
from datetime import date

_SEMVER = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<pre>[0-9A-Za-z.-]+))?(?:\+(?P<build>[0-9A-Za-z.-]+))?$"
)


def iso_week_label(day: str) -> str:
    """The ISO year, week and weekday of a date, as 2026-W53-5.

    Delegated to the standard library on purpose. The interesting part of this
    family is not computing it, it is that the obvious shortcut is wrong at the
    year boundary - and that the year in the label is the ISO year, which in
    the first days of January is often the previous one.
    """
    parts = date.fromisoformat(str(day).strip()).isocalendar()
    return f"{parts[0]:04d}-W{parts[1]:02d}-{parts[2]}"


def _precedence(version: str) -> tuple:
    """A sort key implementing semver precedence, build metadata ignored.

    A release outranks its own prereleases, so the absence of a prerelease
    sorts high. Within a prerelease, numeric identifiers compare numerically
    and rank below alphanumeric ones, and a shorter identifier list loses to a
    longer one that starts the same way.
    """
    match = _SEMVER.match(str(version).strip())
    if match is None:
        raise ValueError(f"not a semver string: {version!r}")
    core = (int(match["major"]), int(match["minor"]), int(match["patch"]))
    if match["pre"] is None:
        return core + (1, ())
    identifiers = tuple(
        (0, int(part), "") if part.isdigit() else (1, 0, part)
        for part in match["pre"].split(".")
    )
    return core + (0, identifiers)


def semver_sort(versions: list[str]) -> list[str]:
    """Versions in ascending precedence order."""
    return sorted(versions, key=_precedence)


def semver_max(versions: list[str]) -> str:
    """The highest-precedence version in the list, which must be unique.

    Build metadata does not participate in precedence, so 3.2.1 and
    3.2.1+build.7 tie - and a tie has no single right answer. Rather than let
    max() pick whichever came first and grade a correct tool wrong for
    returning the other, a tied list is refused outright. The effect is a
    constraint on the corpus: no task may contain one.
    """
    if not versions:
        raise ValueError("need at least one version")
    ranked = sorted(versions, key=_precedence, reverse=True)
    if len(ranked) > 1 and _precedence(ranked[0]) == _precedence(ranked[1]):
        raise ValueError(f"no unique maximum: {ranked[0]} and {ranked[1]} tie")
    return ranked[0]


if __name__ == "__main__":
    assert iso_week_label("2027-01-01") == "2026-W53-5", iso_week_label("2027-01-01")
    assert iso_week_label("2028-02-29") == "2028-W09-2", iso_week_label("2028-02-29")
    assert iso_week_label("2026-01-01") == "2026-W01-4", iso_week_label("2026-01-01")
    assert iso_week_label("2026-09-15") == "2026-W38-2", iso_week_label("2026-09-15")
    # The shortcut this family exists to catch, wrong on the first of these.
    for probe in ("2027-01-01", "2026-12-31", "2025-01-01"):
        day = date.fromisoformat(probe)
        shortcut = f"{day.year:04d}-W{(day.timetuple().tm_yday - 1) // 7 + 1:02d}-{day.isoweekday()}"
        if shortcut != iso_week_label(probe):
            print(f"  shortcut says {shortcut}, ISO says {iso_week_label(probe)}")

    chain = ["1.0.0", "1.0.0-rc.1", "1.0.0-beta.11", "1.0.0-beta.2", "1.0.0-beta",
             "1.0.0-alpha.beta", "1.0.0-alpha.1", "1.0.0-alpha"]
    assert semver_sort(chain) == list(reversed(chain)), semver_sort(chain)
    assert semver_max(chain) == "1.0.0"
    assert semver_sort(["1.0.0-alpha.9", "1.0.0-alpha.10"]) == \
        ["1.0.0-alpha.9", "1.0.0-alpha.10"]
    assert semver_max(["1.9.0", "1.10.0", "1.2.0"]) == "1.10.0"
    assert semver_max(["2.0.0-rc.1", "1.99.99"]) == "2.0.0-rc.1"
    assert semver_max(["4.5.6"]) == "4.5.6"
    try:
        semver_max(["1.0.0+build.9", "1.0.0"])
        raise SystemExit("a tie on build metadata must be refused, not guessed")
    except ValueError:
        pass
    print(f"2027-01-01 -> {iso_week_label('2027-01-01')}")
    print(f"highest of the spec chain -> {semver_max(chain)}")
    print("OK - ref_misc")
