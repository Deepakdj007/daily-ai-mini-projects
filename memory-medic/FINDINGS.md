# Build findings

Things the build turned up that the design did not predict. Written down as
they happened, so the README and the guide's Common Errors section quote real
runtime behaviour rather than reconstructed memory.

## 1. A prompt edit replayed the old answer (cache key gap)

Editing the extraction prompt to add a volatility table changed nothing: the
model kept calling a communication preference `fast`. The cache key carried a
hand-maintained `PROMPT_VERSION` but not the prompt text, so every edit
silently replayed pre-edit answers.

Fixed by folding `cache.fingerprint(SYSTEM_PROMPT)` into the key. A version
constant only retires the cache when somebody remembers to bump it; a hash of
the prompt cannot be forgotten. Any measurement built on a hand-bumped version
string has this hole.

## 2. The extractor called a city "stable" and echoed the topic as a scope

First live extraction returned `city` with `volatility="stable"`, which would
exempt it from ageing forever - the single most common thing to go stale in a
personal assistant. It also set `scope="city"`, duplicating the topic, which
would split one key into two that never meet.

Fixed in the prompt (an explicit volatility table with `city` under `slow`) and
in code (`_clean` drops a scope that merely repeats the topic or the value).
The lesson: an open instruction like "how fast does this change" gets answered
by vibes; a table of topics per class gets answered correctly.

## 3. gpt-oss-20b put the value in the scope field

On the same message, 120b returned `scope="design reviews"` and 20b returned
`scope="async written feedback"` - the value, not the context. A scope equal to
the value carries no context at all, and the whole scope-collision mechanism
depends on it being the context. Now normalised away in code rather than hoped
for in the prompt.

## 4. Starving the completion budget returns an empty 400, not a clear error

At `max_completion_tokens=128`, both gpt-oss models reject a valid strict
schema with `400 json_validate_failed`. On 120b `failed_generation` is the
empty string, which reads exactly like a schema bug and is not one: the model
spends completion tokens reasoning before it emits JSON and runs out mid-think.
20b is more helpful (`"max completion tokens reached before generating a valid
document"`). 2048 is the working budget for extraction.

## 5. The scope filter did not change this model's answer

The published Mem0 failure - "async for design reviews" deleted by "sync for
incident calls" - did not reproduce on gpt-oss even with scope awareness
switched OFF. Asked to compare the two directly, the classifier answered
`unrelated` at 0.95, because the stored `text` still carries the qualifier
("Prefers async written feedback for design reviews.").

So the scope filter is defence in depth here, not the only thing standing
between the store and a wrong delete. It is still worth having - it is a
structural guarantee rather than a model behaviour, and it holds when
extraction is lossy or the text is terse - but the ladder has to report what
actually happened rather than assuming the rung earns its place. If the
`scoped` rung comes back null on the fixture, that is the finding.

## 6. "I moved to Bengaluru" produced scope="moved"

The extractor invented a qualifier for a single-valued topic. Because a scope
is half of a memory's key, the new city would have been filed under
`city/moved` while the old one sat under `city/`, and the supersede chain would
have quietly ended - two live cities, no contradiction ever detected.

Fixed structurally rather than by prompting: a topic that holds one value by
definition cannot carry a meaningful qualifier, so `_clean` forces `scope=""`
for every topic outside the multi-valued set.

## 7. Two different definitions of "stale" disagreed in public

The read-time flag fired below 0.7 freshness (about half a half-life) while the
sweep waited for a full one. At day 10 of a freshly seeded store, eight of
eight retrieved facts rendered as "may be out of date" - a warning on
everything, which is a warning on nothing - while the sweep correctly reported
zero candidates.

`FRESH_FLOOR` is now 0.5, exactly one half-life, so the prompt and the sweep
agree about what stale means. The remaining difference between them is the one
worth measuring: the flag is a read-time hint, the sweep is a durable state
change plus source re-verification.

## 8. An injected clock silently fell back to wall time

`say --as-of 2026-06-10` recorded 2026-09-05. The `say` subparser declared
`--at` with `dest="as_of"`, the same dest as the global flag, and argparse
applies the subparser's empty default *after* parsing the parent - so the
frozen date was overwritten with `""` and `make_clock` returned a system clock.
The bitemporal history was being written against the real date.

The `check` command greps modules for `datetime.now()` and cannot see this: the
discipline was intact, the wiring was not. Fixed with a separate dest merged in
one place, and `say` now prints the date it is acting on, so a clock that is
not what you asked for is visible in the first line of output.

## 9. A wiped store reused thread ids and stopped gating

After `seed --reset`, a change to a stable fact applied instead of parking.
Thread ids are derived from the repair's content, which is what stops one
problem parking twice - but `wipe()` resets the row-id sequence, so a reseeded
store produced the *same* thread id as a repair approved in an earlier run. The
finished checkpoint short-circuited the new one, and the gate silently stopped
gating.

Wiping the store now clears the checkpoint file too: parked threads that point
at deleted rows are garbage, and keeping them is how a safety mechanism turns
into a no-op without ever raising anything.

## 10. Asked to read an HR record, the extractor found nothing

The first ablation scored 0/10 on source drift for every arm, including the one
whose whole job is to catch it. The sweep detected the file change correctly and
re-extracted from the new text - and got back zero facts, so all ten memories
were filed as "the source no longer states this" and none was updated.

The cause was the prompt: "extract durable facts about THE USER from one
message". A record reading `Employee Id: EMP-88356` is not something the user
said, and the model was right to say so. Documents now get their own prompt.
A source of truth that is a document, not a conversation, is a different
extraction problem and pretending otherwise fails silently.

## 11. A scope built from the value cannot survive the value changing

With the document prompt in place, re-verification still reported "the source no
longer states this" for every field. The seeded facts had `scope="zeta retail"`
- the value - because the conversation rule "a scope that echoes the topic is
not a scope" had blanked the legitimate field label `employer`, and the fallback
then used the value.

A scope made of the value changes whenever the value changes, so the key never
matches and re-verification can only ever conclude the field disappeared. A
document's scope is now its field label, which is precisely the part that
survives an update.

While fixing it, the classifier turned out to be the wrong tool here as well: it
called "Reports to Anil Varghese" and "Manager is Sudha Menon" different
attributes and let a stale manager stand. When a record's field label matches
and its value differs, that is a replacement by construction. Deciding it in
code is both cheaper and more reliable than asking a model to re-derive it.

## 12. Two probes that measured nothing, caught by the scorecard's own shape

`aged` read 8/8 for the arm with **no memory at all**, because abstaining passes
"did not assert a stale fact". A probe an empty store passes cannot be evidence
for a memory mechanism. It now also requires the arm to hand over the last known
value and flag it, and the null arm scores 0/8.

`announced` was worse: every arm answered "Nucleus Strength and Ironbark
Fitness", because the original fact was planted by hand under a topic a human
chose while its replacement arrived through the extractor, which chose its own.
The two never shared a key, so nothing ever collided and no policy could resolve
them - the probe was measuring whether the extractor agreed with me. Both the
fact and its replacement now go through the same extraction path.

## 13. The token ledger understated its own headroom by a third of a run

Groq's daily ceiling is per model, and this project deliberately answers on the
120B and extracts on the 20B. The pre-flight summed both against one 200k limit
and refused a run there was budget for. Reported per model, the answer model had
46k free where the combined figure claimed 18k.

## 14. Keeping history changed no answer at all

`overwrite` and `bitemporal` scored **identically on every category**. Keeping
the old row costs nothing and buys nothing *for answering questions about now* -
its value is answering questions about the past, which no probe in this fixture
asks. The honest reading is that supersede-not-delete is an auditability and
recovery property, not an accuracy one, and the leaderboard should not be
allowed to imply otherwise.

The cheap rung is the one that pays: `freshness` - simply rendering how old a
memory is - took the aged probes from 0/8 to 7/8 and the announced ones from
2/4 to 4/4. The sweep's own contribution is real but narrow, and exactly what
was predicted for it: the ten source-drift probes nothing else can see.

## 15. Streamlit magic renders a bare expression, even inside a function

The sidebar's store-consistency line was written as a conditional expression:

    st.sidebar.success("ok") if not problems else st.sidebar.error(problems[0])

Streamlit skips its auto-display magic for direct `st.*` calls, but a ternary is
an `IfExp`, not a `Call`, so it did not match the exemption. The result was
wrapped in `st.write` and the page rendered a `DeltaGenerator` object plus its
entire docstring across the middle of the review screen. As an `if/else`
statement it renders nothing.

## 16. `streamlit run` does not put the project root on sys.path

`python -m src.main` works and `streamlit run src/inbox.py` raises
`ModuleNotFoundError: No module named 'src'`, because Streamlit puts the
*script's* directory on `sys.path` rather than the project root. Setting
PYTHONPATH is a workaround that has to be repeated by every reader; the entry
point now inserts the root itself.

Related, in the same session: `$env:VAR=...` is PowerShell and does nothing in
Git Bash, where it is `VAR=value command`. The inbox now shows the date it is
operating on, which is what made the missing `CLOCK_AT` obvious.

## 17. A locked checkpoint file silently disabled the gate again

Finding 9 came back through a different door. Deleting the checkpoint file on
wipe fixes the ordinary case, but when a running inbox holds that file open the
delete fails, prints a warning nobody reads, and the run continues with stale
threads - so a change to a stable fact applied instead of parking.

A warning must not be the only thing standing between a safety mechanism and
silently not working. The store now mints a **generation** id on every wipe and
mixes it into the thread id, so a reseeded store cannot collide with an earlier
decision whether or not the file was deletable. Verified by running the whole
demo with the inbox deliberately holding the checkpoint open.

## 18. Fixing document scopes broke the demo, and only the demo

Making document facts carry their field label as scope (finding 11) fixed the
fixture and quietly broke `seed.py`, whose hand-written rows have empty scopes.
Re-verification stopped matching them and reported "the source no longer states
this" for the whole profile - the exact symptom finding 10 had already fixed
once, from a different cause.

Two lessons. Re-verification now falls back to an unambiguous topic match when a
stored row has no scope at all. And a fix validated only against the evaluation
fixture is not validated: the demo path exercised different data and had to be
re-run to catch it.

## 19. A supersede that crosses keys severs the chain, and reads as "gone"

The fix in 18 introduced a worse bug than the one it fixed. A document update
arriving with `scope="employer"` superseded a row stored with `scope=""`, so the
successor was filed under a different key. The store looked healthy and the
agent answered correctly from the new row — but the timeline for the original
key showed a fact that ended in August and was never replaced, and
`as_of(employer, "")` returned **nothing**. A memory that is alive under another
name is indistinguishable from one that is gone.

The successor of a fact *is* that fact, later, so `supersede` and `qualify(a)`
now inherit the old row's `(topic, scope)` rather than trusting whatever the
extractor called the new one.

The more useful outcome is the invariant: `check_invariants` now fails when
`superseded_by` points at a row with a different key. It reproduces the bug on
the broken database in one line, and it would have caught both this and the
`scope="moved"` case in finding 6 the moment they happened. Three separate
key-fragmentation bugs in this build argue that the *key* is the thing to guard,
not each path that writes one.

## 20. A `*` in a URL is a 500 on Windows, not a 404

Streamlit 1.63's static route calls `os.stat` on the requested path. On Windows
`*` is not a legal filename character, so a request for `/*` raises `OSError
[WinError 123]` and returns 500 with a long ASGI traceback instead of a clean
404. Harmless - every real asset serves 200 - but alarming in a log, and worth
knowing before blaming the app.
