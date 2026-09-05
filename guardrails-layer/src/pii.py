"""Layer 4: the output firewall. Detect and redact PII before the user sees it.

Inputs:  the assistant's draft answer, plus values the customer already knows
Outputs: a redacted answer and the list of entities that were caught

This is the only layer that reads the model's output rather than its input,
and the only one that cannot be argued out of its job - it never sees an
instruction, so there is nothing in it to override.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    InAadhaarRecognizer,
    InPanRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from src.config import PII_MIN_SCORE

# Only these types are redacted. Presidio also flags DATE_TIME, ORGANIZATION
# and URL, and redacting those turns a useful support answer into black bars.
#
# PERSON is deliberately absent. It comes from spaCy NER, and on
# en_core_web_sm it labels "JSON" and "max" as people with the same 0.85 score
# it gives a real name. Names from records we own are handled by the deny list
# instead, which does not guess. Swap in en_core_web_lg and PERSON becomes
# usable enough to add back here.
WATCHED = [
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "CREDIT_CARD",
    "IN_PAN",
    "IN_AADHAAR",
    "PAYSETU_ACCOUNT",
]


def _keep_last_four(value: str) -> str:
    """Redact a number but keep its last four digits.

    A support answer has to stay usable. "the card ending 1111" lets a
    customer confirm which card they mean; the full number does not need to
    cross the wire to do that.
    """
    digits = [ch for ch in value if ch.isdigit()]
    tail = "".join(digits[-4:])
    return f"[****{tail}]" if tail else "[REDACTED]"


# How each type is rewritten. PERSON stays in the table so that adding it
# back to WATCHED on a bigger spaCy model needs no other change.
OPERATORS = {
    "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
    "PERSON": OperatorConfig("replace", {"new_value": "[NAME]"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
    "PHONE_NUMBER": OperatorConfig("custom", {"lambda": _keep_last_four}),
    "CREDIT_CARD": OperatorConfig("custom", {"lambda": _keep_last_four}),
    "IN_PAN": OperatorConfig("replace", {"new_value": "[PAN]"}),
    "IN_AADHAAR": OperatorConfig("replace", {"new_value": "[AADHAAR]"}),
    "PAYSETU_ACCOUNT": OperatorConfig("replace", {"new_value": "[ACCOUNT]"}),
}


def _account_recognizer() -> PatternRecognizer:
    """Recognise this product's own account numbers, e.g. PS-40028113.

    Every product has an identifier Presidio has never heard of. This is the
    smallest possible example of teaching it one.
    """
    pattern = Pattern(name="paysetu_account", regex=r"\bPS-\d{8}\b", score=0.9)
    return PatternRecognizer(
        supported_entity="PAYSETU_ACCOUNT",
        patterns=[pattern],
        context=["account", "acct", "wallet"],
    )


@lru_cache(maxsize=1)
def _engines() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """Build the analyzer once. Loading spaCy on every call is seconds wasted.

    AnalyzerEngine's default registry ships a slim recognizer set: the India
    recognizers exist in the package but are not loaded, so a PAN comes back
    as ORGANIZATION until you add them by hand.
    """
    config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }
    nlp = NlpEngineProvider(nlp_configuration=config).create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=["en"], nlp_engine=nlp)
    registry.add_recognizer(InPanRecognizer())
    registry.add_recognizer(InAadhaarRecognizer())
    registry.add_recognizer(_account_recognizer())

    analyzer = AnalyzerEngine(
        nlp_engine=nlp, registry=registry, supported_languages=["en"]
    )
    return analyzer, AnonymizerEngine()


@dataclass
class PiiResult:
    """What the firewall found and what it did about it."""

    text: str
    found: list[tuple[str, str]]  # (entity type, the matched string)

    @property
    def leaked(self) -> bool:
        """True if anything had to be redacted."""
        return bool(self.found)


def _is_known(value: str, known: set[str]) -> bool:
    """True if the customer supplied this value themselves."""
    squashed = "".join(ch for ch in value.lower() if ch.isalnum())
    return any(squashed and squashed in k for k in known)


def _spaced_pattern(value: str) -> re.Pattern:
    """Match a value even if it is written with different separators.

    A model that hands over "+91 90112 33445" may equally write it with a
    non-breaking hyphen, or spell a PAN out with spaces. Matching on the
    alphanumeric skeleton catches every rendering of the same secret.
    """
    chars = [ch for ch in value if ch.isalnum()]
    body = r"\W{0,3}".join(re.escape(ch) for ch in chars)
    return re.compile(body, re.IGNORECASE)


def deny_list(text: str, protected: dict[str, str]) -> tuple[str, list[tuple[str, str]]]:
    """Redact values we know must never appear, by exact skeleton match.

    This is the half of the firewall that cannot miss. Presidio's job is
    finding PII nobody told it about; this one starts from the records we
    already own, so there is no score, no model and no threshold involved.
    """
    found: list[tuple[str, str]] = []
    # Longest first. "Meera Iyer" is a substring skeleton of
    # "meera.iyer@example.com", so redacting the name first would leave a
    # mangled "[NAME WITHHELD]@example.com" behind instead of an email tag.
    ordered = sorted(protected.items(), key=lambda kv: -len(kv[1]))
    for label, value in ordered:
        if len(value.strip()) < 4:
            continue
        pattern = _spaced_pattern(value)
        if pattern.search(text):
            found.append((f"DENY_{label.upper()}", value))
            text = pattern.sub(f"[{label.upper()} WITHHELD]", text)
    return text, found


def firewall(text: str, known_values: set[str] | None = None) -> PiiResult:
    """Redact third-party PII from an answer, leaving the customer's own alone.

    Redacting a customer's own phone number back to them is a false positive
    with a real cost - the answer stops making sense. `known_values` is what
    keeps the firewall from doing that.
    """
    analyzer, anonymizer = _engines()
    known = {
        "".join(ch for ch in v.lower() if ch.isalnum())
        for v in (known_values or set())
    }
    known.discard("")

    results = [
        r
        for r in analyzer.analyze(text=text, language="en", entities=WATCHED)
        if r.score >= PII_MIN_SCORE and not _is_known(text[r.start : r.end], known)
    ]
    if not results:
        return PiiResult(text=text, found=[])

    found = [(r.entity_type, text[r.start : r.end]) for r in results]
    redacted = anonymizer.anonymize(
        text=text, analyzer_results=results, operators=OPERATORS
    )
    return PiiResult(text=redacted.text, found=found)
