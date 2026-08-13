"""The morning inbox: every draft the agent parked overnight, waiting on you.

Inputs:  memory.db (the drafts) and nightscout.db (the index)
Outputs: resumed graph runs, and entries appended to output/reading-list.md

    PYTHONPATH=. uv run streamlit run src/inbox.py

This process never ran the graph forward. It reads drafts straight out of the
checkpoints the night process left behind, which is the whole point of the design:
the agent worked hours ago, in a process that has already exited.
"""

import streamlit as st

from src import config, store
from src.graph import build_graph, load_parked_payload, open_checkpointer, resume

st.set_page_config(page_title="night-scout inbox", page_icon="🌙", layout="wide")


@st.cache_resource(show_spinner=False)
def get_runtime():
    """Open both databases once per Streamlit process, not once per rerun.

    This is why open_checkpointer() builds its own connection instead of using
    SqliteSaver.from_conn_string(): that is a context manager, and it would close
    itself the moment the first script run finished.
    """
    graph = build_graph(open_checkpointer())
    conn = store.connect(config.DB_PATH, config.SQLITE_TIMEOUT)
    return graph, conn


def settle(graph, conn, thread_id: str, decision: str, feedback: str = "") -> None:
    """Resume the thread, update the index, and redraw."""
    with st.spinner(f"{decision}…"):
        status = resume(graph, thread_id, decision, feedback)
    store.settle_draft(conn, thread_id, status)
    st.rerun()


def render_card(graph, conn, row) -> None:
    """One parked draft, with the three things a human can do about it."""
    thread_id = row["thread_id"]
    payload = load_parked_payload(graph, thread_id)

    with st.container(border=True):
        if payload is None:
            # The checkpoint says this thread is finished but the index still says
            # parked — the two databases disagreed for a moment.
            st.warning(f"{row['title']} — already decided elsewhere.")
            if st.button("Clear it", key=f"clear-{thread_id}"):
                store.settle_draft(conn, thread_id, "done")
                st.rerun()
            return

        draft = payload["draft"]
        st.markdown(f"### {draft['headline']}")
        st.caption(
            f"{payload['source']} · score {payload['score']}/10 · "
            f"revision {payload['revision']} · {payload.get('reason', '')}"
        )
        st.write(draft["why_it_matters"])
        for point in draft.get("key_points", []):
            st.markdown(f"- {point}")
        tags = " ".join(f"`{tag}`" for tag in draft.get("tags", []))
        st.markdown(f"{tags}  \n[{payload['title']}]({payload['url']})")

        approve, reject, _ = st.columns([1, 1, 3])
        if approve.button("Approve", key=f"ok-{thread_id}", width="stretch",
                          type="primary"):
            settle(graph, conn, thread_id, "approve")
        if reject.button("Reject", key=f"no-{thread_id}", width="stretch"):
            settle(graph, conn, thread_id, "reject")

        remaining = config.MAX_REVISIONS - payload["revision"]
        with st.expander(f"Ask for a rewrite ({remaining} left)"):
            if remaining < 1:
                st.caption("This draft has used its revisions. Approve it or reject it.")
            else:
                note = st.text_area(
                    "What should change?",
                    key=f"fb-{thread_id}",
                    placeholder="shorter, and lead with what actually changed",
                )
                if st.button("Send back", key=f"edit-{thread_id}"):
                    if note.strip():
                        settle(graph, conn, thread_id, "edit", note.strip())
                    else:
                        st.warning("Say what to change first.")


def render_sidebar(conn) -> None:
    """Counts, the carry-over queue, and the last few wakes."""
    with st.sidebar:
        st.header("🌙 night-scout")
        counts = store.status_counts(conn)
        left, right = st.columns(2)
        left.metric("Parked", counts.get("parked", 0))
        right.metric("Committed", counts.get("committed", 0))

        st.caption(
            f"{store.queue_depth(conn, config.SCORE_THRESHOLD)} item(s) scored "
            f"{config.SCORE_THRESHOLD}+ still queued for a draft"
        )
        st.divider()

        st.subheader("Last wakes")
        for wake in store.recent_wakes(conn, limit=8):
            st.caption(
                f"{wake['sim_time'][-5:]} · polled {wake['polled']} · "
                f"fresh {wake['fresh']} · parked {wake['parked']}"
            )

        st.divider()
        if st.button("Refresh", width="stretch"):
            # Drops the cached connections so a night that finished while this tab
            # was open becomes visible.
            st.cache_resource.clear()
            st.rerun()


def main() -> None:
    """Draw the inbox."""
    graph, conn = get_runtime()
    render_sidebar(conn)

    st.title("Good morning")
    parked = store.parked_drafts(conn)

    if not parked:
        st.success("Nothing waiting. The agent either found nothing or you cleared it.")
        st.caption("Run a night:  PYTHONPATH=. uv run python -m src.night run --demo")
        return

    st.caption(f"{len(parked)} draft(s) parked overnight, best score first.")
    for row in parked:
        render_card(graph, conn, row)


main()
