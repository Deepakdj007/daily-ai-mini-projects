# andon-cord

A research team of three agents that runs on its own and stops to ask you a
question only when it hits something no search engine can answer. The question
survives the process that asked it, so you can close the terminal and answer
tomorrow morning.

Built on LangGraph with Groq's free tier and keyless DuckDuckGo search. No paid
API, no vector database, no infrastructure.

## The catch this project is built around

Most human-in-the-loop tutorials put a person in front of every action. Approve
this email. Approve this ticket. That is a permission gate, and it scales badly:
the agent learns nothing about when your attention is actually worth spending.

Toyota's assembly lines have an andon cord. Any worker can pull it and halt the
whole line, and the rule that makes it work is not "pull it often". It is: pull
it when you have hit a defect you cannot fix at your station.

So the interesting question is not *how* an agent pauses. It is *when* it decides
to. Here every agent may pull the cord, and the entire prompt that governs it is
one rule with four worked examples: is the missing fact private to this client,
or could a search answer it?

Two researchers run in parallel and can both hit that wall in the same instant.
They do not each interrupt you. Their questions land on a desk, and the desk
stops the line once with both questions on it.

## The three policies

Every policy uses byte-identical prompts. The policy decides whether a question
reaches a human, never whether an agent is allowed to think of one. That is what
makes the comparison mean something: `never` still records what it would have
asked.

| policy | the lead may ask | researchers may ask |
|---|---|---|
| `never` | no | no |
| `lead` | once, before any work starts | no |
| `team` | once | yes, batched into one turn |

## Run it

```
uv sync
cp .env.example .env        # then paste a free key from console.groq.com/keys

PYTHONPATH=. uv run python -m src.main run "<your brief>"
PYTHONPATH=. uv run python -m src.main run "<your brief>" --park
PYTHONPATH=. uv run python -m src.main inbox
PYTHONPATH=. uv run python -m src.main answer <thread_id>
PYTHONPATH=. uv run python -m src.main compare
PYTHONPATH=. uv run python -m src.main ask-rate --samples 5 --all
```

`--park` freezes at the first question and exits. `inbox` finds it again from a
new terminal. `answer` picks it up and drives the run to the end.

## What it actually does, measured

`ask-rate` puts the same brief to the lead five times and counts how often it
pulls the cord. That repeat matters: at temperature 0.0 this model flags a brief
on one sample and not the next, so a single run is an anecdote.

| brief | hides | asks |
|---|---|---|
| competitive brief on "our autopay flow" | which product | 5/5 |
| "our mandate success rate" vs the industry | a private number | 4/5 |
| migration plan "off the old rails" | which product, off what | 5/5 |
| UPI AutoPay vs eNACH, fully specified | nothing | 0/5 |
| RBI e-mandate rules, fully specified | nothing | 0/5 |
| "how failures should be handled, who signs off" | which service, whose sign-off | **0/5** |

Zero false alarms on the complete briefs, and that held through every version of
the prompt. Recall took work: with only the general rule the lead caught the
first brief 1 time in 5, and one concrete instruction — scan the brief for a
phrase like "our X" that names something you cannot look up — took it to 5 in 5.

The last row is a real miss, kept in rather than tuned away. That brief has no
possessive phrase to find. It is ambiguous by omission: it never says which
service, and "who needs to sign off" is an org question. The check catches an
ambiguous *referent* and is blind to a *missing subject*.

## Known limits

The lead catches an ambiguous *referent* and is blind to a brief that is
ambiguous by *omission* — the last row above. Researchers can ask too, but they
do so rarely: their bar is deliberately higher than the lead's, since they see
one sub-question rather than the whole brief.

A full `compare` costs roughly 60,000 tokens, so about three of them is a free
day. `ask-rate --samples 5 --all` is about 30,000.

## Where the pauses live, and why

`interrupt()` appears in exactly two nodes, and neither does anything before
calling it. LangGraph re-runs a node from its first line when a run resumes, so
a pause inside a researcher would mean repeating its web search and its model
call just to arrive back at the same question.

That is also why nothing is written to disk inside a node. The report is saved by
the CLI after the graph returns.
