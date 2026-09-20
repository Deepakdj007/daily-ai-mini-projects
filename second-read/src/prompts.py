"""Every prompt in the project, in one place.

Inputs:  a CaseView and small strings - never a Case, never a reference report
Outputs: prompt strings, and the [img] layout for multi-image prompts

The localization, CT and lab prompts are copied from Google's own notebooks and
the MedGemma 1.5 technical report word for word, misprints included. The model
was tuned against that wording and matching it is free accuracy.

The verification prompt is byte-identical across every audit condition. The
only thing that changes between sighted, blind and shuffled is which image
bytes travel with it, and that is the entire experiment.
"""

from __future__ import annotations

from src.types import CaseView

# --- The case note -----------------------------------------------------------
# Asking for each region by name is how a radiologist dictates; it also means
# the note states each finding explicitly, either way, so the claim scanner
# has something to lift.
NOTE = (
    "You are assisting a radiologist. Read this frontal chest radiograph and write a "
    "short report.{context}\n\n"
    "Under the heading FINDINGS: comment in complete sentences on heart size, the "
    "mediastinum, the lungs (consolidation, edema, nodules), the pleural spaces "
    "(effusion, pneumothorax) and the bones. State explicitly when something is absent.\n"
    "Under the heading IMPRESSION: give a one- or two-sentence summary."
)


def note(view: CaseView) -> str:
    """The report prompt for one case."""
    context = f" Clinical indication: {view.context.strip()}" if view.context.strip() else ""
    return NOTE.format(context=context)


# --- The verification prompt -------------------------------------------------
# The claim is re-asked as a yes/no question in MedGemma's own Final Answer
# convention. Wording is positive-polarity only and carries nothing specific
# to the case, so a blind call and a sighted call differ by the image alone.
VERIFY = (
    "You are reading a frontal chest radiograph. Is {phrase} present on this radiograph? "
    "Give at most two sentences of reasoning, then your final answer, yes or no."
)

# Reasoning first, then the answer, both inside one schema. Measured on 21
# probes: free text with a "Final Answer:" anchor parsed 12 of 21 blind calls,
# because a blind model lectures about radiographs until it hits the token cap
# and never reaches its answer. The schema parsed 21 of 21, blind calls fell
# from 8.2 s to 1.1 s, and sighted accuracy was unchanged. The reasoning field
# comes FIRST so the model still reasons before committing - a bare enum would
# make it answer from nothing.
VERIFY_SCHEMA: dict = {
    "type": "object",
    "properties": {"reasoning": {"type": "string", "maxLength": 300},
                   "final_answer": {"type": "string", "enum": ["yes", "no"]}},
    "required": ["reasoning", "final_answer"],
}


def verify(phrase: str) -> str:
    """One claim as a question. Identical text in every audit condition."""
    return VERIFY.format(phrase=phrase)


# --- Localization (verbatim, cxr_anatomy_localization_with_hugging_face.ipynb)
LOCALIZE = (
    "Instructions:\n"
    "The following user query will require outputting bounding boxes. The format of "
    "bounding boxes coordinates is [y0, x0, y1, x1] where (y0, x0) must be top-left corner "
    "and (y1, x1) the bottom-right corner. This implies that x0 < x1 and y0 < y1. Always "
    "normalize the x and y coordinates the range [0, 1000], meaning that a bounding box "
    "starting at 15% of the image width would be associated with an x coordinate of 150. "
    "You MUST output a single parseable json list of objects enclosed into ```json...``` "
    "brackets, for instance ```json[{{\"box_2d\": [800, 3, 840, 471], \"label\": \"car\"}}, "
    "{{\"box_2d\": [400, 22, 600, 73], \"label\": \"dog\"}}]``` is a valid output. Now answer "
    "to the user query.\n\n"
    "Remember \"left\" refers to the patient's left side where the heart is and sometimes "
    "underneath an L in the upper right corner of the image.\n\n"
    "Query:\n"
    "Where is the {object_name}? Don't give a final answer without reasoning. Output the "
    "final answer in the format \"Final Answer: X\" where X is a JSON list of objects. The "
    "object needs a \"box_2d\" and \"label\" key. Answer:"
)


def localize(object_name: str) -> str:
    """Google's localization prompt for one named structure or finding."""
    return LOCALIZE.format(object_name=object_name)


# --- Lab extraction (verbatim, MedGemma 1.5 technical report, Table 11) ------
LAB_EXTRACT = (
    "You are a Clinical Data Extraction Specialist. Your job is to parse lab reports with "
    "high precision. From the given lab report, extract all lab tests into a JSON list. Each "
    "test object in the list must include: name, result, unit, range, panel, method, "
    "specimen, sample_collection_time (formatted as DD-MM-YYYY HH:MM:SS)"
)
LAB_STRUCTURE = (
    "Convert the lab report text below into a JSON list of test objects with exactly these "
    "keys: name, result, unit, range, panel, method, specimen, sample_collection_time. Copy "
    "every value exactly as written. Use null for anything the text does not state. Output "
    "only the JSON."
)

# --- CT volumes (verbatim shape, technical report Table 13) ------------------
CT = (
    "{images}After looking at the indication and patient history \"{history}\"... Is there "
    "\"{label}\" in the CT volume? You may write out your argument before stating your final "
    "answer \"Final Answer: yes\" or \"Final Answer: no\"."
)


def ct(n_slices: int, history: str, label: str) -> str:
    """The CT prompt with one labelled [img] slot per slice, in stack order."""
    slots = "".join(f"Axial slice {k + 1} of {n_slices}: [img]\n" for k in range(n_slices)) + "\n"
    return CT.format(images=slots, history=history, label=label)


# --- Longitudinal comparison -------------------------------------------------
# Google's notebook passes the two films positionally with no labels. Labelling
# them and putting text between them is what keeps Ollama from fusing two
# same-size images into one and the model from mixing them up.
LONGITUDINAL = (
    "PRIOR study: [img]\n\nCURRENT study: [img]\n\n"
    "Provide a comparison of these two images and include details from the image which "
    "students should take note of when reading longitudinal CXR. For each of heart size, "
    "lung opacities, pleural effusion and pneumothorax, say whether the current study is "
    "improved, worsened, stable, new or resolved compared with the prior."
)
LONGITUDINAL_STRUCTURE = (
    "From the comparison text below, produce a JSON object with a key \"changes\": a list of "
    "objects each with \"finding\" (one of: heart size, lung opacities, pleural effusion, "
    "pneumothorax) and \"change\" (one of: improved, worsened, stable, new, resolved, "
    "not_assessable). Output only the JSON."
)

# --- The canary --------------------------------------------------------------
CANARY = "What case ID is printed on this image? Reply 'Final Answer: <id>'."

if __name__ == "__main__":
    assert "{" not in verify("cardiomegaly")
    assert verify("x") == verify("x"), "verification wording must be stable"
    assert localize("heart").count('"box_2d"') == 3
    assert ct(3, "cough", "pleural effusion").count("[img]") == 3
    assert LONGITUDINAL.count("[img]") == 2
    print(note(CaseView("demo", (), "Preop bariatric surgery."))[:120] + "...")
    print("OK - prompts")
