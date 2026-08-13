"""Streamlit inspector — chat on the left, the three tiers live on the right.

Switch customer in the sidebar and watch what changes: the semantic panel swaps
to a different account record, while the episodes and the playbook stay put.
That is the scope difference made visible — semantic memory is per customer,
episodic and procedural are shared by the whole desk.

Run:
    PYTHONPATH=. uv run streamlit run src/app.py
"""

from __future__ import annotations

import uuid

import streamlit as st

from src import episodic, procedural, semantic
from src.agent import ask_desk, build_graph, open_checkpointer
from src.config import CUSTOMERS
from src.store import open_store

st.set_page_config(page_title="recall-desk", page_icon="🗂️", layout="wide")


@st.cache_resource
def _resources():
    """Open the store, checkpointer and graph once per server process.

    Built with `open_store()` rather than `SqliteStore.from_conn_string`, which is
    a context manager and would close its connection the moment this function
    returned. The store also has to be thread-safe here: Streamlit reruns the
    script on every widget interaction, from a different thread than the one that
    created it.
    """
    store = open_store()
    graph = build_graph(store, open_checkpointer())
    return store, graph


store, graph = _resources()

if "threads" not in st.session_state:
    st.session_state.threads = {}
if "history" not in st.session_state:
    st.session_state.history = {}
if "last_trace" not in st.session_state:
    st.session_state.last_trace = {}

# --- sidebar: who am I talking to, and the two gated writes ----------------
with st.sidebar:
    st.subheader("Customer")
    customer_id = st.radio(
        "Account",
        options=list(CUSTOMERS),
        format_func=lambda c: CUSTOMERS[c],
        label_visibility="collapsed",
    )
    st.caption("Semantic memory is scoped to this customer. Episodes and the playbook are not.")

    st.session_state.threads.setdefault(customer_id, f"ui-{uuid.uuid4().hex[:8]}")
    st.session_state.history.setdefault(customer_id, [])
    thread_id = st.session_state.threads[customer_id]
    st.caption(f"thread `{thread_id}`")

    if st.button("New thread", use_container_width=True):
        st.session_state.threads[customer_id] = f"ui-{uuid.uuid4().hex[:8]}"
        st.session_state.history[customer_id] = []
        st.rerun()

    st.divider()
    st.subheader("Resolve ticket")
    st.caption("Episodic memory is written on resolution — an episode needs an outcome.")
    if st.button("File an episode", use_container_width=True):
        turns = st.session_state.history[customer_id]
        if len(turns) < 2:
            st.warning("Nothing to summarise yet.")
        else:
            transcript = "\n".join(f"{r.capitalize()}: {c}" for r, c in turns)
            with st.spinner("Summarising..."):
                episode = episodic.summarize_episode(transcript)
            if episode is None:
                st.info("Not worth keeping — nothing was resolved.")
            else:
                episodic.append(store, episode)
                st.success("Episode filed, scrubbed of customer identity.")
                st.rerun()

    st.divider()
    st.subheader("Supervisor feedback")
    st.caption("The only path to a playbook edit. Nothing rewrites it unprompted.")
    feedback = st.text_area("Feedback", label_visibility="collapsed", placeholder="Stop promising settlement dates.")
    if st.button("Apply to playbook", use_container_width=True) and feedback.strip():
        with st.spinner("Proposing an edit..."):
            edit = procedural.propose_rule_edit(feedback, procedural.load(store))
            _, note = procedural.apply_edit(store, edit)
        st.success(note)
        st.caption(edit.reason)
        st.rerun()

# --- main: chat, then the three tiers -------------------------------------
chat_col, mem_col = st.columns([3, 2], gap="large")

with chat_col:
    st.subheader(f"PaySetu desk — {CUSTOMERS[customer_id]}")
    for role, content in st.session_state.history[customer_id]:
        with st.chat_message(role):
            st.write(content)

    if ticket := st.chat_input("Describe the issue..."):
        st.session_state.history[customer_id].append(("user", ticket))
        with st.chat_message("user"):
            st.write(ticket)
        with st.chat_message("assistant"), st.spinner("Recalling and answering..."):
            state = ask_desk(graph, customer_id, ticket, thread_id=thread_id)
            st.write(state["reply"])
            st.caption(f"severity={state['severity']}  escalate={state['escalate']}")
        st.session_state.history[customer_id].append(("assistant", state["reply"]))
        st.session_state.last_trace = state.get("trace", {})
        st.rerun()

    trace = st.session_state.last_trace
    if trace:
        with st.expander("What went into the last turn", expanded=False):
            st.write(trace.get("summary", ""))
            tokens = trace.get("tokens", {})
            st.caption(
                f"{trace.get('system_chars', 0)} prompt chars -> "
                f"{tokens.get('input', 0)} input tokens, {tokens.get('output', 0)} output"
            )
            for label, texts, scores in (
                ("semantic", trace.get("facts", []), trace.get("fact_scores", [])),
                ("episodic", trace.get("episodes", []), trace.get("episode_scores", [])),
            ):
                for text, score in zip(texts, scores):
                    st.write(f"`{label}` **{score:.3f}** — {text}")
            if trace.get("written_facts"):
                st.success(f"wrote semantic facts: {', '.join(trace['written_facts'])}")

with mem_col:
    st.subheader("Memory")

    facts = semantic.all_facts(store, customer_id)
    with st.container(border=True):
        st.markdown(f"**Semantic** · {CUSTOMERS[customer_id]} only · {len(facts)} facts")
        st.caption("Upsert by topic, so a newer fact replaces the one it contradicts.")
        for item in facts:
            st.write(f"`{item.key}` {item.value['text']}")

    episodes = episodic.all_episodes(store)
    with st.container(border=True):
        st.markdown(f"**Episodic** · shared by all customers · {len(episodes)} episodes")
        st.caption("Append-only, scrubbed of identity, retrieved by symptom.")
        for item in episodes:
            with st.expander(item.value["text"][:60]):
                st.write(f"**ruled out** {', '.join(item.value.get('tried', []))}")
                st.write(f"**cause** {item.value['root_cause']}")
                st.write(f"**fix** {item.value['fix']}")

    book = procedural.load(store)
    versions = [b.version for b in procedural.history(store)]
    with st.container(border=True):
        st.markdown(f"**Procedural** · shared · v{book.version}")
        st.caption("Never searched. Always in the prompt. Capped, so it stays cheap.")
        for i, rule in enumerate(book.rules, 1):
            st.write(f"{i}. {rule}")
        if len(versions) > 1:
            target = st.selectbox("Roll back to", [v for v in versions if v != book.version])
            if st.button("Roll back"):
                _, note = procedural.rollback(store, target)
                st.success(note)
                st.rerun()
