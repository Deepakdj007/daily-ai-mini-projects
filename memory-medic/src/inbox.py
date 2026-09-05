"""The review inbox: cards for parked repairs, and the timeline behind them.

Run:
    PYTHONPATH=. uv run streamlit run src/inbox.py

This process never ran the sweep that parked these decisions. It reads them out
of the checkpoint file and resumes them, which is the whole point of parking on
a durable interrupt rather than holding a queue in memory: the agent can work
overnight, exit, and still be waiting for an answer in the morning.
"""

from __future__ import annotations

import streamlit as st

from src import config

st.set_page_config(page_title="memory medic", page_icon="*", layout="wide")


@st.cache_resource
def get_runtime():
    """Open the store, the graph and the clock once per process, not per rerun.

    Streamlit reruns the whole script on every click, so this must be cached -
    and the connections are built with check_same_thread=False because those
    reruns happen on a different thread than the one that opened them.
    """
    from src import clock as clockmod, store
    from src.graph import build_graph, open_checkpointer

    conn = store.connect()
    graph = build_graph(open_checkpointer())
    clock = clockmod.make_clock(config.CLOCK_AT)
    return conn, graph, clock


def sidebar(conn, clock) -> None:
    """What the store holds, and what it is waiting on."""
    from src import store

    counts = store.counts(conn)
    st.sidebar.header("memory medic")
    st.sidebar.caption(f"as of **{clock.today()}**" + (" (frozen)" if clock.frozen else ""))
    st.sidebar.metric("waiting for you", counts.get("parked", 0))
    columns = st.sidebar.columns(2)
    columns[0].metric("active", counts.get("active", 0))
    columns[1].metric("flagged", counts.get("needs_verification", 0))
    columns = st.sidebar.columns(2)
    columns[0].metric("superseded", counts.get("superseded", 0))
    columns[1].metric("ended", counts.get("expired", 0))
    problems = store.check_invariants(conn)
    st.sidebar.success("store consistent") if not problems else st.sidebar.error(problems[0])
    if st.sidebar.button("refresh"):
        st.cache_resource.clear()
        st.rerun()


def review_page() -> None:
    """One card per parked repair: what is stored, what is proposed, and why."""
    from src import store
    from src.graph import load_parked_payload, parked_rows, resume_thread

    conn, graph, clock = get_runtime()
    sidebar(conn, clock)
    st.title("Repairs waiting for a decision")
    st.caption("The agent parked these because it was not confident enough to "
               "change a memory on its own.")

    rows = parked_rows(conn)
    if not rows:
        st.success("Nothing waiting. The agent applied everything it was sure about.")
        return

    for row in rows:
        payload = load_parked_payload(graph, row["thread_id"], conn, clock)
        if payload is None:
            # The checkpoint is authoritative about whether a decision is
            # pending; the index can lag if it was settled from the CLI.
            st.info(f"`{row['topic']}` was already decided in another window.")
            if st.button("Clear it", key=f"clear-{row['thread_id']}"):
                store.settle_repair(conn, row["thread_id"], "applied", decided_by="agent",
                                    decided_at=clock.now())
                st.rerun()
            continue
        _card(conn, graph, clock, row, payload, resume_thread)


def _card(conn, graph, clock, row, payload: dict, resume) -> None:
    """Render one repair, and the four things a reviewer can do about it."""
    from src import clock as clockmod

    before = payload.get("before") or {}
    after = payload.get("after") or {}
    label = f"{payload['topic']}" + (f" / {payload['scope']}" if payload.get("scope") else "")

    with st.container(border=True):
        head = st.columns([3, 1])
        head[0].subheader(label)
        head[0].caption(f"{payload['op']} · found by {payload['detector']} · "
                        f"confidence {payload['confidence']:.2f}")
        head[1].progress(min(float(payload["confidence"]), 1.0))

        panes = st.columns(2)
        panes[0].markdown("**On file**")
        if before:
            panes[0].markdown(f"### {before.get('value', '-')}")
            panes[0].caption(before.get("text", ""))
            if before.get("valid_from"):
                panes[0].caption(f"true since {clockmod.to_iso(int(before['valid_from']))} · "
                                 f"{before.get('volatility', '')}")
        else:
            panes[0].caption("nothing on file")
        panes[1].markdown("**Proposed**")
        panes[1].markdown(f"### {after.get('value', '-')}")
        panes[1].caption(after.get("text", ""))

        st.caption(f"why: {payload.get('reason', '')}")
        if payload.get("evidence"):
            with st.expander("evidence"):
                st.code(payload["evidence"], language="diff")

        thread = row["thread_id"]
        actions = st.columns([1, 1, 2])
        if actions[0].button("Approve", key=f"ok-{thread}", type="primary"):
            resume(graph, conn, thread, "approve", clock=clock)
            st.rerun()
        if actions[1].button("Reject", key=f"no-{thread}"):
            resume(graph, conn, thread, "reject", clock=clock)
            st.rerun()

        with st.expander("Change the value, or keep both"):
            edited = st.text_input("Correct value", value=after.get("value", ""),
                                   key=f"edit-{thread}")
            if st.button("Apply my value", key=f"apply-{thread}"):
                resume(graph, conn, thread, {"decision": "edit", "value": edited}, clock=clock)
                st.rerun()
            st.caption("Both true, in different situations? Give each one a context.")
            pair = st.columns(2)
            old_scope = pair[0].text_input("context for the stored one",
                                           value=before.get("scope", ""), key=f"s0-{thread}")
            new_scope = pair[1].text_input("context for the new one",
                                           value=after.get("scope", ""), key=f"s1-{thread}")
            if st.button("Keep both", key=f"both-{thread}"):
                resume(graph, conn, thread,
                       {"decision": "keep_both", "scopes": [old_scope, new_scope]}, clock=clock)
                st.rerun()


def timeline_page() -> None:
    """Every version of one memory, and what this system would have said when."""
    from src import clock as clockmod, store

    conn, _, clock = get_runtime()
    sidebar(conn, clock)
    st.title("What was true, and when")

    users = [r[0] for r in conn.execute("SELECT DISTINCT user_id FROM memories").fetchall()]
    if not users:
        st.info("The store is empty. Run `python -m src.main seed` first.")
        return
    picks = st.columns(3)
    user = picks[0].selectbox("user", users)
    topics = [r[0] for r in conn.execute(
        "SELECT DISTINCT topic FROM memories WHERE user_id = ? ORDER BY topic", (user,)).fetchall()]
    topic = picks[1].selectbox("topic", topics)
    scopes = [r[0] for r in conn.execute(
        "SELECT DISTINCT scope FROM memories WHERE user_id = ? AND topic = ? ORDER BY scope",
        (user, topic)).fetchall()]
    scope = picks[2].selectbox("context", scopes, format_func=lambda s: s or "(none)")

    rows = store.history(conn, user, topic, scope)
    asked = st.date_input("as of", value=clockmod.to_iso(clock.now()))
    at = clockmod.to_epoch(str(asked))
    current = store.as_of(conn, user, topic, scope, at)
    st.markdown(f"On **{asked}** this system would have said: "
                f"**{current['value'] if current else 'nothing on file'}**")

    st.dataframe([{
        "id": row["id"], "value": row["value"],
        "true from": clockmod.to_iso(row["valid_from"]),
        "true until": "—" if row["valid_to"] >= config.OPEN_END else clockmod.to_iso(row["valid_to"]),
        "recorded": clockmod.to_iso(row["recorded_at"]),
        "status": row["status"],
        "answers your date": bool(current and row["id"] == current["id"]),
    } for row in rows], width="stretch")
    st.caption("Nothing here was deleted. A superseded row keeps the window it was "
               "true for, which is what makes the date picker above answerable.")


def shelf_page() -> None:
    """Facts the sweep flagged as needing a human's confirmation."""
    from src import clock as clockmod, repair, store
    from src.policy import POLICIES, RepairPlan

    conn, _, clock = get_runtime()
    sidebar(conn, clock)
    st.title("Aged out")
    st.caption("Nobody contradicted these. They simply outlived their shelf life, "
               "so the agent stopped presenting them as current fact.")

    rows = conn.execute(
        "SELECT * FROM memories WHERE status = 'needs_verification' ORDER BY last_verified_at"
    ).fetchall()
    if not rows:
        st.success("Nothing has gone stale.")
        return

    policy = POLICIES[config.WRITE_POLICY]
    for row in rows:
        age = clockmod.days_between(clock.now(), int(row["last_verified_at"]))
        half_life = config.HALF_LIFE_DAYS.get(row["volatility"], 180)
        with st.container(border=True):
            columns = st.columns([3, 1, 1])
            columns[0].markdown(f"**{row['text']}**")
            columns[0].caption(f"{row['topic']} · {row['volatility']} · last confirmed "
                               f"{age:.0f} days ago, half-life {half_life}")
            plan = RepairPlan(op="confirm", detector="human", user_id=row["user_id"],
                              topic=row["topic"], scope=row["scope"], old_id=int(row["id"]),
                              before=store.row_to_dict(row), confidence=1.0,
                              reason="a human confirmed it still holds")
            if columns[1].button("Still true", key=f"y-{row['id']}"):
                repair.apply(conn, plan, clock=clock, policy=policy, decided_by="human")
                st.rerun()
            if columns[2].button("Retire", key=f"n-{row['id']}"):
                plan.op, plan.extra = "expire", {"to_status": "expired", "close_window": True}
                plan.reason = "a human said it is no longer true"
                repair.apply(conn, plan, clock=clock, policy=policy, decided_by="human")
                st.rerun()


# Guarded so that importing a page function does not run the whole app. Streamlit
# executes this file as __main__, so `streamlit run` still gets its navigation.
if __name__ == "__main__":
    st.navigation([
        st.Page(review_page, title="Review", url_path="review", default=True),
        st.Page(shelf_page, title="Aged out", url_path="aged"),
        st.Page(timeline_page, title="Timeline", url_path="timeline"),
    ]).run()
