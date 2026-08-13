"""The event streams the agent watches. All three are keyless.

Inputs:  network
Outputs: Item objects, normalized so downstream code never asks where they came from

Every fetcher returns [] instead of raising. One dead feed at 3am must not end the
night, and a source that is briefly down will be picked up on the next wake.
"""

import calendar
import html
import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

import feedparser
import httpx

from src import config
from src.state import Item

HN_SEARCH = "https://hn.algolia.com/api/v1/search_by_date"
ARXIV_QUERY = "https://export.arxiv.org/api/query"

_SCRIPT_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.DOTALL | re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_last_arxiv_at: float = 0.0
_arxiv_quiet_until: float = 0.0


def html_to_text(raw: str) -> str:
    """Turn markup into readable text: drop script/style bodies, then all tags."""
    without_code = _SCRIPT_STYLE.sub(" ", raw)
    return html.unescape(_TAG.sub(" ", without_code))


def clean(text: str, limit: int) -> str:
    """Collapse all whitespace to single spaces and truncate.

    arXiv titles and abstracts carry hard newlines from their LaTeX source, so this
    runs over everything, not just HTML.
    """
    flat = " ".join(html_to_text(text or "").split())
    return flat[:limit].strip()


def _entry_ts(entry) -> float:
    """Unix seconds for a feed entry, from whichever date field it actually has.

    feedparser normalizes every date format it understands into a UTC struct_time,
    which is the only reason arXiv's ISO stamps and RSS's RFC 822 stamps can be
    compared at all. Entries with no parseable date get 0.0 and are dropped by the
    age filter — for a "what's new" scout, an undated entry is not news.
    """
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    return float(calendar.timegm(parsed)) if parsed else 0.0


def _iso(ts: float) -> str:
    """Display timestamp derived from ts, so every source reads the same way."""
    if not ts:
        return ""
    return datetime.fromtimestamp(ts, timezone.utc).isoformat(timespec="seconds")


def fetch_page(url: str) -> str:
    """One HTTP GET with a timeout and a polite User-Agent."""
    response = httpx.get(
        url,
        timeout=config.HTTP_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": config.USER_AGENT},
    )
    response.raise_for_status()
    return response.text


# --- HackerNews -------------------------------------------------------------


def fetch_hn() -> list[Item]:
    """Stories from the last HN_LOOKBACK_HOURS that cleared HN_MIN_POINTS.

    A rolling window, not "since the last wake". A story needs hours to accumulate
    20 points, so asking for items that are both brand new and already popular
    returns almost nothing. Re-polling the same stories costs nothing because
    dedup drops the ones already stored.
    """
    since = datetime.now(timezone.utc) - timedelta(hours=config.HN_LOOKBACK_HOURS)
    since_ts = int(since.timestamp())
    query = urlencode(
        {
            "tags": "story",
            "numericFilters": f"created_at_i>{since_ts},points>{config.HN_MIN_POINTS}",
            "hitsPerPage": config.HN_HITS_PER_PAGE,
        }
    )
    payload = httpx.get(
        f"{HN_SEARCH}?{query}",
        timeout=config.HTTP_TIMEOUT,
        headers={"User-Agent": config.USER_AGENT},
    )
    payload.raise_for_status()

    items: list[Item] = []
    for hit in payload.json().get("hits", []):
        object_id = hit.get("objectID")
        if not object_id or not hit.get("title"):
            continue
        # Ask HN and other text posts have url: null.
        link = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"
        crowd = f"{hit.get('points', 0)} points, {hit.get('num_comments', 0)} comments"
        body = clean(hit.get("story_text") or "", config.TRIAGE_SNIPPET_CHARS)
        ts = float(hit.get("created_at_i") or 0.0)
        items.append(
            Item(
                item_id=f"hn:{object_id}",
                source="hn",
                title=clean(hit["title"], 300),
                url=link,
                snippet=f"{crowd}. {body}".strip(),
                ts=ts,
                created_at=_iso(ts),
            )
        )
    return items


# --- arXiv ------------------------------------------------------------------


def fetch_arxiv() -> list[Item]:
    """The newest submissions in ARXIV_CATEGORIES.

    The Atom <summary> IS the abstract, so arXiv items never need a page fetch later.
    """
    global _last_arxiv_at, _arxiv_quiet_until
    # Once arXiv has said 429, stop asking for a while. An agent that polls all
    # night and retries a rate-limited endpoint every 45 minutes is the reason the
    # limit exists — backing off is both politer and more likely to recover.
    if time.monotonic() < _arxiv_quiet_until:
        return []

    # arXiv's terms of use: no more than one request every three seconds.
    wait = config.ARXIV_DELAY_SECONDS - (time.monotonic() - _last_arxiv_at)
    if wait > 0 and _last_arxiv_at:
        time.sleep(wait)
    _last_arxiv_at = time.monotonic()

    categories = " OR ".join(f"cat:{c}" for c in config.ARXIV_CATEGORIES)
    query = urlencode(
        {
            "search_query": categories,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": 0,
            "max_results": config.ARXIV_MAX_RESULTS,
        }
    )
    try:
        raw = fetch_page(f"{ARXIV_QUERY}?{query}")
    except (httpx.HTTPStatusError, httpx.TimeoutException) as exc:
        rate_limited = (
            isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code == 429
        )
        if isinstance(exc, httpx.HTTPStatusError) and not rate_limited:
            raise
        # Timeouts get the same treatment as an explicit 429. When arXiv throttles
        # an address it stops answering rather than refusing, so retrying every wake
        # burns HTTP_TIMEOUT seconds each time and slows the whole night down.
        _arxiv_quiet_until = time.monotonic() + config.ARXIV_COOLDOWN_SECONDS
        why = "rate-limited" if rate_limited else "not responding"
        print(f"  ! arxiv {why} — backing off for "
              f"{config.ARXIV_COOLDOWN_SECONDS / 60:.0f} min")
        return []

    parsed = feedparser.parse(raw)
    if parsed.bozo:
        print(f"  ! arxiv: malformed feed ({parsed.bozo_exception}) — using what parsed")

    items: list[Item] = []
    for entry in parsed.entries:
        raw_id = entry.get("id", "")
        if not raw_id:
            continue
        # "http://arxiv.org/abs/2608.04007v1" -> "2608.04007v1". Keeping the version
        # means a v2 revision is legitimately a new item, not a duplicate.
        short_id = raw_id.rsplit("/", 1)[-1]
        ts = _entry_ts(entry)
        items.append(
            Item(
                item_id=f"arxiv:{short_id}",
                source="arxiv",
                title=clean(entry.get("title", ""), 300),
                url=raw_id.replace("http://", "https://"),
                snippet=clean(entry.get("summary", ""), config.TRIAGE_SNIPPET_CHARS),
                ts=ts,
                created_at=_iso(ts),
            )
        )
    return items


# --- RSS / Atom -------------------------------------------------------------


def _entry_snippet(entry) -> str:
    """Pull a summary out of an entry that may not have one.

    The Hugging Face blog feed ships title, pubDate, link and guid — no description
    at all — so `entry.summary` raises there. Triaging on the title alone is a valid
    outcome, not an error.
    """
    text = entry.get("summary", "") or entry.get("description", "")
    return clean(text, config.TRIAGE_SNIPPET_CHARS)


def fetch_rss() -> list[Item]:
    """Every configured feed, each one isolated so a single failure is survivable.

    Feeds do not agree on how much history to serve. The Hugging Face blog ships its
    entire archive — 800+ entries back to 2021 — so each feed is sorted by date and
    truncated to the newest RSS_ENTRIES_PER_FEED before anything else looks at it.
    """
    items: list[Item] = []
    for url in config.RSS_FEEDS:
        try:
            parsed = feedparser.parse(fetch_page(url))
        except Exception as exc:  # noqa: BLE001 — one dead feed is not fatal
            print(f"  ! rss {url}: {type(exc).__name__}: {exc}")
            continue
        if parsed.bozo:
            print(f"  ! rss {url}: malformed feed ({parsed.bozo_exception})")

        from_feed: list[Item] = []
        for entry in parsed.entries:
            key = entry.get("id") or entry.get("link")
            if not key or not entry.get("title"):
                continue
            ts = _entry_ts(entry)
            from_feed.append(
                Item(
                    item_id=f"rss:{key}",
                    source="rss",
                    title=clean(entry.get("title", ""), 300),
                    url=entry.get("link", key),
                    snippet=_entry_snippet(entry),
                    ts=ts,
                    created_at=_iso(ts),
                )
            )
        from_feed.sort(key=lambda i: i.ts, reverse=True)
        items.extend(from_feed[: config.RSS_ENTRIES_PER_FEED])
    return items


# --- the wake's poll --------------------------------------------------------


def _interleave(fresh: list[Item], limit: int) -> list[Item]:
    """Fill the triage budget round-robin across sources, newest first within each.

    Sources publish at wildly different rates: HN gets a new story every few minutes,
    arXiv drops a batch once a day, a blog posts weekly. Taking a global "newest 12"
    hands the entire budget to HN on every single wake, and the papers and writeups —
    the things most likely to match the profile — are never even scored.
    """
    buckets: dict[str, list[Item]] = {}
    for item in fresh:
        buckets.setdefault(item.source, []).append(item)
    for bucket in buckets.values():
        bucket.sort(key=lambda i: i.ts, reverse=True)

    picked: list[Item] = []
    for rank in range(max((len(b) for b in buckets.values()), default=0)):
        for source in sorted(buckets):
            if len(picked) >= limit:
                break
            if rank < len(buckets[source]):
                picked.append(buckets[source][rank])
        if len(picked) >= limit:
            break
    picked.sort(key=lambda i: i.ts, reverse=True)
    return picked


def poll_all(known: set[str]) -> tuple[int, list[Item]]:
    """Poll every source once.

    Returns (how many items came back, the fresh ones chosen for triage). Filters in
    order: drop anything older than MAX_ITEM_AGE_HOURS, drop anything already stored,
    then share the budget across sources. Anything left out was never marked as seen,
    so it is simply re-polled on the next wake.
    """
    polled: list[Item] = []
    for name, fetch in (("hn", fetch_hn), ("arxiv", fetch_arxiv), ("rss", fetch_rss)):
        try:
            polled.extend(fetch())
        except Exception as exc:  # noqa: BLE001 — never end the night over one source
            # One line, not a stack trace: a source being down for one wake is
            # ordinary, and the next wake will pick up whatever it missed.
            detail = str(exc).splitlines()[0][:90]
            print(f"  ! {name} unavailable this wake — {type(exc).__name__}: {detail}")

    oldest = time.time() - config.MAX_ITEM_AGE_HOURS * 3600
    fresh: list[Item] = []
    seen_now: set[str] = set()
    for item in polled:
        if item.ts < oldest:
            continue
        if item.item_id in known or item.item_id in seen_now:
            continue
        seen_now.add(item.item_id)
        fresh.append(item)

    return len(polled), _interleave(fresh, config.MAX_ITEMS_PER_WAKE)
