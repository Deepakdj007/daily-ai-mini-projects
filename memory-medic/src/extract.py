"""Turning a message into candidate facts, with a shelf life attached.

Inputs:  one message, the date it was said, and where it came from
Outputs: a list of Facts, each with a topic, a scope, a volatility and a window

The call depends on the message alone - never on what is already stored. That
is deliberate: it makes extraction identical across every write policy, so the
ablation pays for it once and replays it for the rest.

Two prompt rules were written after watching the model get them wrong. The
volatility table exists because the model called "lives in Bengaluru" stable,
which would exempt it from ageing forever. The scope rule exists because the
smaller model set scope to the value itself ("async written feedback"), which
would defeat the one mechanism that keeps two context-qualified preferences
from being treated as a contradiction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, Field, ValidationError

from src import cache, config, llm
from src import clock as clockmod

TOPICS = [
    "name", "birthday", "city", "address", "employer", "job_title", "car", "phone",
    "email", "diet", "allergy", "partner", "kids", "pet", "project", "tool_preference",
    "communication_preference", "appointment", "travel", "health", "hobby", "other",
]

# Topics that can hold several true facts at once. Each needs a scope saying
# which one, or two unrelated facts collide on the same key.
MULTI_VALUED = {
    "pet", "kids", "project", "appointment", "travel", "allergy",
    "tool_preference", "communication_preference", "other",
}

_FACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["topic", "scope", "text", "value", "volatility", "valid_from", "valid_to", "confidence"],
    "properties": {
        "topic": {"type": "string", "enum": TOPICS},
        "scope": {"type": "string"},
        "text": {"type": "string"},
        "value": {"type": "string"},
        "volatility": {"type": "string", "enum": ["stable", "slow", "fast", "scheduled"]},
        "valid_from": {"type": ["string", "null"]},
        "valid_to": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
    },
}
EXTRACT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["facts"],
    "properties": {"facts": {"type": "array", "items": _FACT_SCHEMA}},
}

SYSTEM_PROMPT = """You extract durable facts about THE USER from one message.

Return only facts about the user themselves. Skip anything about other people,
anything hypothetical or wished-for, and anything true only of this moment.

value  = the atomic answer, as few words as possible ("Bengaluru", "Honda City").
text   = one short sentence stating the fact ("Lives in Bengaluru.").
scope  = the CONTEXT that makes this fact specific, never the value itself.
         "For design reviews I prefer async feedback" -> scope "design reviews",
         value "async written feedback". If the fact needs no qualifier, use "".
         These topics ALWAYS need a scope naming which one: {multi}.

volatility - how fast this kind of fact goes out of date:
  stable    never changes: name, birthday, blood group, allergy, kids' names
  slow      changes every few years: city, address, employer, job_title, car,
            phone, email, partner, diet, hobby, and settled preferences about
            how the user likes to work
  fast      changes within weeks: current project, current reading, gym routine,
            temporary tooling, anything the user calls "current" or "right now"
  scheduled has an end date: appointments, trips, deadlines, permits, rotations

valid_from = when the fact STARTED being true, YYYY-MM-DD, or null for the
             message date. "I moved last week" started a week ago.
valid_to   = when it STOPS being true, YYYY-MM-DD. Required for scheduled
             facts, null for everything else.
confidence = 0.0-1.0, how sure you are the user asserted this about themselves.

Today is {today}."""

DOCUMENT_PROMPT = """You read a RECORD about a user and turn each field into a fact.

This is a document, not something the user said. Every line that carries a
field and a value is a fact about them; extract all of them.

value  = the field's value, exactly as written ("EMP-88356", "Desk 9A-02").
text   = one short sentence stating it ("Employee id is EMP-88356.").
scope  = the field's own label, lowercased ("employee id", "cost centre").
         Always set it. It is how this fact is matched to the one already on
         file when the document changes.
topic  = the closest fit from the list, or "other" when nothing fits well.

volatility - how fast this kind of field goes out of date:
  stable    never changes: date of birth, blood group, national id
  slow      changes every few years: employer, desk, manager, office, badge,
            cost centre, asset tags, extensions, ids
  fast      changes within weeks: current sprint, current assignment
  scheduled has an end date stated on the line: permits, passes, rotations

valid_from = null unless the line states when it started.
valid_to   = the end date when the line states one, otherwise null.
confidence = 0.0-1.0.

Today is {today}."""


class Fact(BaseModel):
    """One extracted fact. Parsed from validated JSON, never used to emit schema."""

    topic: str
    scope: str = ""
    text: str
    value: str
    volatility: str
    valid_from: str | None = None
    valid_to: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    valid_from_epoch: int = 0
    valid_to_epoch: int = config.OPEN_END


def normalise_scope(scope: str) -> str:
    """Lowercase, collapse whitespace, cap the length.

    Scope is half of a memory's key, so 'Design Reviews' and 'design  reviews'
    must not become two different memories.
    """
    cleaned = re.sub(r"\s+", " ", (scope or "").strip().lower())
    cleaned = cleaned.strip(" .,:;-")
    return cleaned[:40]


def _parse_date(text: str | None, fallback: int) -> tuple[int, bool]:
    """ISO date -> epoch. Returns (epoch, ok); ok is False when we fell back."""
    if not text:
        return fallback, True
    try:
        return clockmod.to_epoch(date.fromisoformat(text.strip()).isoformat()), True
    except (ValueError, TypeError):
        return fallback, False


def _clean(fact: Fact, said_at: int, *, from_document: bool = False) -> Fact:
    """Apply the rules strict mode cannot express, and clamp what it can't check."""
    fact.scope = normalise_scope(fact.scope)
    fact.confidence = min(max(float(fact.confidence), 0.0), 1.0)

    start, ok_from = _parse_date(fact.valid_from, said_at)
    fact.valid_from_epoch = min(start, said_at) if fact.valid_from else said_at
    end, ok_to = _parse_date(fact.valid_to, config.OPEN_END)
    fact.valid_to_epoch = end if fact.valid_to else config.OPEN_END
    if not (ok_from and ok_to):
        fact.confidence = max(0.0, fact.confidence - 0.1)

    # A scheduled fact with no end is not scheduled; treat it as fast-moving
    # rather than inventing a date the user never gave.
    if fact.volatility == "scheduled" and fact.valid_to_epoch >= config.OPEN_END:
        fact.volatility = "fast"
    if fact.valid_to_epoch <= fact.valid_from_epoch:
        fact.valid_to_epoch = config.OPEN_END
    if from_document:
        # A document's scope is its field LABEL, and that label is the only
        # thing tying this fact to the row already on file. It must therefore be
        # the one part that survives the value changing - so a scope equal to
        # the value is discarded and the topic is used instead. Getting this
        # wrong is silent and total: scope="zeta retail" stops matching the
        # moment the employer changes, and every re-verification then reports
        # "the source no longer states this" instead of updating anything.
        if fact.scope == normalise_scope(fact.value):
            fact.scope = ""
        if not fact.scope:
            fact.scope = fact.topic
    else:
        # A scope repeating the value or the topic carries no context, so it is
        # not a scope. Left in, it would split one key into two that never meet.
        if fact.scope and fact.scope in {normalise_scope(fact.value), fact.topic}:
            fact.scope = ""
        if fact.topic in MULTI_VALUED:
            if not fact.scope:
                fact.scope = normalise_scope(fact.value)
        else:
            # A single-valued topic holds one fact by definition, so a qualifier
            # on it cannot mean anything. Observed: "I moved to Bengaluru"
            # produced scope="moved", which would have filed the new city under
            # a different key from the old one and ended the supersede chain.
            fact.scope = ""
    return fact


@dataclass(frozen=True, slots=True)
class Extraction:
    """What one extraction call produced, plus what it cost."""

    facts: list[Fact]
    replayed: bool
    error: str = ""


async def extract(message: str, *, said_at: int, model: str = "",
                  source_kind: str = "conversation", use_cache: bool = True) -> Extraction:
    """Pull candidate facts out of one message."""
    today = clockmod.to_iso(said_at)
    # A document is not a message. Asked to find "facts the user stated", the
    # model reads an HR record and correctly reports that the user said nothing
    # - which silently turns every re-verification into "the source no longer
    # mentions this" and never updates a single value.
    if source_kind == "conversation":
        system = SYSTEM_PROMPT.format(today=today, multi=", ".join(sorted(MULTI_VALUED)))
    else:
        system = DOCUMENT_PROMPT.format(today=today)
    model = model or config.EXTRACT_MODEL
    key = cache.key(model, {
        "kind": "extract", "prompt": config.PROMPT_VERSION,
        "prompt_hash": cache.fingerprint(system),
        "schema": config.EXTRACT_SCHEMA_VERSION, "text": message,
        "today": today, "source_kind": source_kind,
    })
    data, completion = await llm.complete_json(
        [{"role": "system", "content": system}, {"role": "user", "content": message}],
        schema=EXTRACT_SCHEMA, schema_name="extraction", model=model,
        cache_key=key if use_cache else "",
        max_completion_tokens=config.MAX_COMPLETION_TOKENS["extract"],
    )
    if data is None:
        return Extraction([], completion.replayed, completion.error or "extraction failed")

    facts: list[Fact] = []
    for raw in data.get("facts", []):
        try:
            facts.append(_clean(Fact.model_validate(raw), said_at,
                                from_document=source_kind != "conversation"))
        except ValidationError:
            continue  # one malformed fact must not lose the rest of the turn
    return Extraction(facts, completion.replayed)


if __name__ == "__main__":
    import asyncio

    async def _smoke() -> None:
        said = clockmod.to_epoch("2026-06-10")
        result = await extract(
            "I moved to Bengaluru last week, and my dentist appointment is on the 20th "
            "of this month. For design reviews I prefer async written feedback, though.",
            said_at=said,
        )
        assert not result.error, result.error
        by_topic = {fact.topic: fact for fact in result.facts}
        print(f"{'topic':26} {'scope':22} {'value':24} {'vol':10} window")
        for fact in result.facts:
            end = "open" if fact.valid_to_epoch >= config.OPEN_END else clockmod.to_iso(fact.valid_to_epoch)
            print(f"{fact.topic:26} {fact.scope!r:22} {fact.value!r:24} "
                  f"{fact.volatility:10} {clockmod.to_iso(fact.valid_from_epoch)} -> {end}")

        assert "city" in by_topic, by_topic.keys()
        assert by_topic["city"].volatility == "slow", \
            f"city must be slow, not {by_topic['city'].volatility} - a stable city never ages"
        assert by_topic["city"].valid_from_epoch <= said, "'moved last week' starts before today"
        assert by_topic["city"].scope == "", \
            f"a scope echoing the topic is not a scope: {by_topic['city'].scope!r}"

        pref = by_topic["communication_preference"]
        assert pref.scope and pref.scope != normalise_scope(pref.value), \
            f"scope must be the context, not the value: {pref.scope!r} vs {pref.value!r}"

        appt = by_topic["appointment"]
        assert appt.volatility == "scheduled" and appt.valid_to_epoch < config.OPEN_END
        assert clockmod.to_iso(appt.valid_to_epoch) == "2026-06-20", clockmod.to_iso(appt.valid_to_epoch)
        print(f"OK - {len(result.facts)} facts, volatility and scope both correct")

    asyncio.run(_smoke())
