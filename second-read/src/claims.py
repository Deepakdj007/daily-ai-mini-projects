"""The finding vocabulary, and the scanner that lifts claims out of radiology prose.

Inputs:  free text - an IU-Xray report, or a note the model just wrote
Outputs: one Mention per finding the text takes a position on

One scanner serves two masters. The probe author runs it over reference reports
to decide what a radiologist asserted or explicitly denied; the audit runs it
over the model's note to decide what the model claimed. If those two used
different rules, the flag would measure the difference between the rules.

The rules are NegEx-shaped and deliberately dumb: a negation cue before the
finding inside the same clause makes it a "no"; a hedge anywhere in the
sentence makes it "uncertain"; a finding never mentioned is simply absent from
the output, never a "no". No model labels its own probes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.types import Mention, Polarity


@dataclass(frozen=True, slots=True)
class Finding:
    """A finding the vocabulary knows, and how prose spells it."""

    phrase: str                       # how the verification question names it
    patterns: tuple[str, ...]         # spellings that assert or deny it
    stated_normal: tuple[str, ...]    # explicit normals that mean "no" without a "no"


FINDINGS: dict[str, Finding] = {
    "cardiomegaly": Finding(
        "cardiomegaly, meaning an enlarged cardiac silhouette",
        (r"cardiomegaly", r"enlarged (?:cardiac|heart) (?:silhouette|size)",
         r"(?:cardiac silhouette|heart size|heart) (?:is|appears) enlarged",
         r"enlargement of the (?:cardiac silhouette|heart)", r"cardiac enlargement"),
        (r"(?:heart|cardiac) (?:size|silhouette)[^.;]{0,40}(?:normal|within normal limits)",
         r"normal (?:heart|cardiac) (?:size|silhouette)", r"heart is normal in size",
         r"cardiomediastinal silhouette[^.;]{0,30}(?:normal|within normal limits)",
         r"normal cardiomediastinal silhouette",
         r"heart size and (?:mediastinal|pulmonary) [^.;]{0,30}(?:normal|within normal limits)")),
    "pleural_effusion": Finding(
        "a pleural effusion",
        (r"pleural effusions?", r"\beffusions?\b", r"pleural fluid"),
        ()),
    "consolidation": Finding(
        "focal consolidation or airspace opacity",
        (r"consolidations?", r"air[- ]?space (?:disease|opacit(?:y|ies))", r"focal opacit(?:y|ies)",
         r"infiltrates?", r"focal airspace"),
        (r"lungs are clear", r"clear lungs", r"lungs are (?:otherwise )?clear")),
    # Vascular congestion is deliberately NOT here. A radiologist who writes
    # "mild pulmonary vascular congestion" has not said there is edema, and
    # three probes labelled yes on exactly that sentence in the second hand
    # check. The negatives all say "edema", so including congestion on the
    # positive side only would tilt the finding.
    "pulmonary_edema": Finding("pulmonary edema", (r"pulmonary edema", r"\bedema\b"), ()),
    "pneumothorax": Finding("a pneumothorax", (r"pneumothora(?:x|ces)",), ()),
    "nodule": Finding(
        "a pulmonary nodule or mass",
        (r"\bnodules?\b", r"\bmass(?:es)?\b", r"nodular (?:opacit(?:y|ies)|densit(?:y|ies))"),
        ()),
    "atelectasis": Finding("atelectasis", (r"atelecta(?:sis|tic)",), ()),
    "hyperinflation": Finding(
        "hyperinflation of the lungs",
        (r"hyperinflat(?:ed|ion)", r"hyperexpan(?:ded|sion)", r"emphysema(?:tous)?"),
        ()),
    "degenerative": Finding(
        "degenerative change of the thoracic spine",
        (r"degenerative (?:changes?|disease)", r"spondylosis", r"osteophyt"),
        ()),
    # Scoped to the ribs on purpose. IU-Xray reports fractures of the humerus,
    # clavicle and thoracic vertebrae in the same words, and "is a rib fracture
    # present" is not answered by a healed clavicle - the hand check found four
    # such probes. The pattern now requires rib context within the phrase.
    "fracture": Finding(
        "a rib fracture",
        (r"\bribs?\b[^.;]{0,40}\bfractures?\b", r"\bfractures?\b[^.;]{0,40}\bribs?\b",
         r"\d+(?:st|nd|rd|th)\s+rib\s+fracture"),
        ()),
}

# "clear of" was missing from the first cue list and it is how IU-Xray most
# often denies airspace disease - the hand check caught seven of those.
_NEG_PRE = re.compile(r"\b(?:no|without|negative for|free of|clear of|absence of|absent|not)\b", re.I)
_NEG_POST = re.compile(r"\b(?:is|are) (?:not|absent)\b|\bnot (?:seen|identified|present|visualized|evident)\b", re.I)
_SCOPE_END = re.compile(r"\b(?:but|however|although|except)\b|;", re.I)
# "may not" covers the boilerplate "fractures may not be demonstrated";
# "differential" and "question" mark a finding raised, not one seen.
_HEDGE = re.compile(
    r"cannot (?:be )?exclude|can ?not rule out|may (?:represent|be|reflect|not)|possib(?:le|ly|ility)|"
    r"questionable|\bquestion\b|suspicious|suspect(?:ed|ion)?|suggest|borderline|\bversus\b|\bvs\b|"
    r"likely|probabl[ey]|"
    r"equivocal|concern(?:\s+is)?\s+for|cannot be excluded|differential|consider(?:ed)? for|"
    # "atelectasis, infiltrates and/or small effusion" offers alternatives; it
    # does not assert any one of them. "Upper limits of normal" is the same
    # move in the other direction.
    r"and\s*/\s*or|upper limits? of normal|difficult to (?:entirely )?exclude|"
    r"could (?:have|be|represent|reflect)|combination of", re.I)

# "Small area of atelectasis OR infiltrate" names two candidates and commits to
# neither. The same "or" inside a negation is a legitimate list - "no effusion
# or pneumothorax" denies both - so this only applies where nothing is negated.
_ALTERNATION = re.compile(r"\bor\b", re.I)

# A negation carrying a SIZE qualifier denies only that grade of the finding:
# "no LARGE pleural effusion" is silent about a small one, and three probes in
# the first hand check were labelled "no" on exactly that sentence.
#
# Only size words qualify. "No focal consolidation", "no acute fracture" and
# "no discrete nodules" are how radiologists write a plain negative, not a
# narrowed one - treating those as hedges emptied four findings of their
# negatives on the first attempt.
_QUALIFIED_NEG = re.compile(
    r"\b(?:no|without|negative for)\b[^.;]{0,30}\b(?:large|gross|sizable|significant|"
    r"substantial|moderate)\b", re.I)
_COMPARISON = re.compile(
    r"unchanged|stable|compared|prior|previous|interval|\bsince\b|\bagain\b|persistent|\bnew\b|"
    r"resolved|improv|worsen|increased|decreased|similar|re-?demonstrat|\bremains?\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def sentences(text: str) -> list[str]:
    """Split on sentence punctuation. IU-Xray's XXXX redactions survive harmlessly."""
    return [part.strip() for part in _SENTENCE.split(text or "") if part.strip()]


def _negated(sentence: str, start: int, end: int) -> tuple[bool, bool]:
    """(negated, qualified). A qualified negation denies only one grade of the finding."""
    before = sentence[:start]
    cut = list(_SCOPE_END.finditer(before))
    clause = before[cut[-1].end():] if cut else before
    qualified = bool(_QUALIFIED_NEG.search(clause))
    if _NEG_PRE.search(clause):
        return True, qualified
    after = sentence[end:]
    stop = _SCOPE_END.search(after)
    tail = after[: stop.start()] if stop else after
    return bool(_NEG_POST.match(after.lstrip()) or _NEG_POST.search(tail)), qualified


def is_comparative(sentence: str) -> bool:
    """Language that needs a prior study - such sentences never become probes."""
    return bool(_COMPARISON.search(sentence))


def scan(text: str) -> list[Mention]:
    """Every position the text takes on every vocabulary finding."""
    found: list[Mention] = []
    for sentence in sentences(text):
        hedged = bool(_HEDGE.search(sentence))
        for name, finding in FINDINGS.items():
            for pattern in finding.stated_normal:
                if re.search(pattern, sentence, re.I):
                    # A hedged normal ("heart size upper limits of normal") is
                    # not a statement of normal; the hedge outranks the pattern.
                    found.append(Mention(name, "uncertain" if hedged else "no", sentence,
                                         "hedge" if hedged else "stated_normal"))
                    break
            else:
                for pattern in finding.patterns:
                    match = re.search(pattern, sentence, re.I)
                    if not match:
                        continue
                    negated, qualified = _negated(sentence, match.start(), match.end())
                    alternation = (not negated and not _NEG_PRE.search(sentence)
                                   and bool(_ALTERNATION.search(sentence)))
                    if hedged or alternation or (negated and qualified):
                        found.append(Mention(name, "uncertain", sentence, "hedge"))
                    elif negated:
                        found.append(Mention(name, "no", sentence, "negation"))
                    else:
                        found.append(Mention(name, "yes", sentence, "assertion"))
                    break
    return found


def resolve(mentions: list[Mention]) -> dict[str, Mention]:
    """One position per finding. A text that says both yes and no is uncertain."""
    out: dict[str, Mention] = {}
    for mention in mentions:
        prior = out.get(mention.finding)
        if prior is None or prior.polarity == mention.polarity:
            out[mention.finding] = mention
        else:
            out[mention.finding] = Mention(mention.finding, "uncertain", mention.sentence, "hedge")
    return out


def polarity_of(text: str, finding: str) -> Polarity | None:
    """The resolved position on one finding, or None when the text is silent."""
    got = resolve(scan(text)).get(finding)
    return got.polarity if got else None


if __name__ == "__main__":
    report = ("The cardiac silhouette and mediastinum size are within normal limits. There is no "
              "pulmonary edema. There is no focal consolidation. There are no XXXX of a pleural "
              "effusion. There is no evidence of pneumothorax.")
    got = {m.finding: (m.polarity, m.source) for m in scan(report)}
    assert got["cardiomegaly"] == ("no", "stated_normal"), got
    assert got["pulmonary_edema"] == ("no", "negation")
    assert got["consolidation"] == ("no", "negation")
    assert got["pleural_effusion"] == ("no", "negation")
    assert got["pneumothorax"] == ("no", "negation")
    assert "nodule" not in got, "unmentioned must stay absent, never become a no"

    got = {m.finding: m.polarity for m in scan("Borderline cardiomegaly. Clear lungs. Small right effusion.")}
    assert got == {"cardiomegaly": "uncertain", "consolidation": "no", "pleural_effusion": "yes"}, got

    assert polarity_of("No pneumothorax or pleural effusion.", "pleural_effusion") == "no"
    assert polarity_of("Mild cardiomegaly but no effusion.", "cardiomegaly") == "yes"
    assert polarity_of("Mild cardiomegaly but no effusion.", "pleural_effusion") == "no"
    assert polarity_of("Opacity likely represents atelectasis.", "atelectasis") == "uncertain"
    assert polarity_of("Cardiomegaly. Heart size is normal.", "cardiomegaly") == "uncertain"
    assert is_comparative("Unchanged cardiomegaly.") and not is_comparative("Cardiomegaly.")
    # Every label error the two hand checks turned up, now caught by rule.
    assert polarity_of("The lungs are expanded and clear of airspace disease.", "consolidation") == "no"
    assert polarity_of("Differential includes chronic lung disease and pulmonary edema.", "pulmonary_edema") == "uncertain"
    assert is_comparative("Redemonstrated severe L1 fracture.")
    # A fracture elsewhere in the thorax is not a rib fracture.
    assert polarity_of("Old fracture, right mid clavicle.", "fracture") is None
    assert polarity_of("Fracture of right proximal humerus.", "fracture") is None
    assert polarity_of("Old XXXX fracture deformity of approximately T6.", "fracture") is None
    assert polarity_of("Right 4th rib fracture.", "fracture") == "yes"
    assert polarity_of("Multiple age-indeterminate left-sided rib fractures.", "fracture") == "yes"
    assert polarity_of("No acute, displaced rib fractures.", "fracture") == "no", "radiology idiom, not a hedge"
    assert polarity_of("No focal consolidation.", "consolidation") == "no"
    assert polarity_of("No discrete nodules are identified.", "nodule") == "no"
    # A negation that denies only one grade denies nothing about the others.
    assert polarity_of("No large pleural effusion or pneumothorax.", "pleural_effusion") == "uncertain"
    assert polarity_of("No pleural effusion or pneumothorax.", "pleural_effusion") == "no"
    assert polarity_of("Atelectasis, infiltrates and/or small left pleural effusion.",
                       "pleural_effusion") == "uncertain"
    assert polarity_of("Heart size upper limits of normal.", "cardiomegaly") == "uncertain"
    # Alternation in an assertion commits to neither candidate; in a negation it denies both.
    assert polarity_of("Small area of atelectasis or infiltrate near the hilum.", "consolidation") == "uncertain"
    assert polarity_of("Cardiomegaly or pericardial fluid could have this appearance.", "cardiomegaly") == "uncertain"
    assert polarity_of("No pleural effusion or pneumothorax.", "pneumothorax") == "no"
    assert polarity_of("Patchy airspace disease in the lingula without evidence of effusion.",
                       "consolidation") == "yes"
    assert polarity_of("Although this could be scar, concern is for nodule.", "nodule") == "uncertain"
    # Congestion is not edema.
    assert polarity_of("Mild pulmonary vascular congestion.", "pulmonary_edema") is None
    assert polarity_of("Interstitial pulmonary edema.", "pulmonary_edema") == "yes"
    assert polarity_of("Suspected fractures of the left anterior 6th and 7th ribs.", "fracture") == "uncertain"
    # The hedge is about what the nodules ARE, not whether they are there.
    assert polarity_of("Multiple bilateral pulmonary nodules, concerning for metastatic disease.",
                       "nodule") == "yes"
    print(f"{len(FINDINGS)} findings in the vocabulary")
    print("OK - claims")
