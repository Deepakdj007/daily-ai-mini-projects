# app.py
import time
import streamlit as st
from graph import marketmind_graph

st.set_page_config(page_title="MarketMind", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .bullish  { color: #00e676; font-weight: bold; }
    .bearish  { color: #ff5252; font-weight: bold; }
    .neutral  { color: #ffeb3b; font-weight: bold; }
    .buy-box  { color: #00e676; font-size: 2.2rem; font-weight: bold; }
    .sell-box { color: #ff5252; font-size: 2.2rem; font-weight: bold; }
    .hold-box { color: #ffeb3b; font-size: 2.2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 MarketMind")
st.caption("Parallel Multi-Agent Stock & Crypto Analyst  |  LangGraph + Groq Llama-3.3-70B")
st.divider()

# Input
col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.text_input(
        "Enter ticker symbol",
        placeholder="e.g. AAPL  BTC  ETH  TSLA  NVDA",
        help="Stocks: AAPL, TSLA, NVDA  |  Crypto: BTC, ETH, SOL, DOGE, ADA",
    )
with col2:
    st.write(""); st.write("")
    run_btn = st.button("⚡ Analyze", type="primary", use_container_width=True)

SIGNAL_EMOJI = {"BULLISH": "🟢", "BEARISH": "🔴", "NEUTRAL": "🟡"}
VERDICT_EMOJI = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
AGENT_LABELS = {
    "price": "📈 Price & Technicals",
    "sentiment": "📰 News Sentiment",
    "onchain": "🔗 Market Structure",
    "macro": "🌍 Macro Context",
    "risk": "⚠️ Risk Analysis",
}
AGENT_ORDER = ["price", "sentiment", "onchain", "macro", "risk"]


def confidence_bar(c: float) -> str:
    return "█" * int(c * 10) + "░" * (10 - int(c * 10))


if run_btn and ticker.strip():
    status = st.empty()
    results = st.empty()

    with status.container():
        st.info(f"⚡ Launching 5 parallel agents for **{ticker.strip().upper()}**...")
        st.progress(0)

    start = time.time()

    try:
        output = marketmind_graph.invoke({
            "ticker": ticker.strip(),
            "asset_type": "stock",        # validate_input will auto-detect
            "agent_signals": [],
            "final_verdict": None,
            "final_confidence": None,
            "final_reasoning": None,
        })

        elapsed = round(time.time() - start, 1)
        status.empty()

        with results.container():
            verdict = output.get("final_verdict", "HOLD")
            confidence = output.get("final_confidence", 0.5)
            reasoning = output.get("final_reasoning", "")
            asset_type = output.get("asset_type", "stock")

            v1, v2, v3 = st.columns([1, 1, 2])
            with v1:
                st.markdown("### Verdict")
                css = verdict.lower() + "-box"
                st.markdown(
                    f'<p class="{css}">{VERDICT_EMOJI.get(verdict, "🟡")} {verdict}</p>',
                    unsafe_allow_html=True,
                )
            with v2:
                st.markdown("### Confidence")
                st.markdown(f"### {confidence:.0%}")
                st.progress(confidence)
            with v3:
                st.markdown("### Reasoning")
                st.markdown(f"*{reasoning}*")

            st.caption(
                f"⏱ {elapsed}s  |  Asset: {asset_type.upper()}  |  "
                "Model: llama-3.3-70b-versatile (Groq)"
            )
            st.divider()

            st.markdown("### 🤖 Agent Breakdown")
            signals_dict = {s["agent"]: s for s in output.get("agent_signals", [])}
            cols = st.columns(5)

            for i, key in enumerate(AGENT_ORDER):
                s = signals_dict.get(key)
                if not s:
                    continue
                with cols[i]:
                    sig_css = s["signal"].lower()
                    st.markdown(f"**{AGENT_LABELS[key]}**")
                    st.markdown(
                        f'<span class="{sig_css}">'
                        f'{SIGNAL_EMOJI[s["signal"]]} {s["signal"]}</span>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Confidence: {s['confidence']:.0%}")
                    st.text(confidence_bar(s["confidence"]))
                    st.caption(f"*{s['summary']}*")

            with st.expander("📊 View Raw Agent Data"):
                for key in AGENT_ORDER:
                    s = signals_dict.get(key)
                    if s and s.get("raw_data"):
                        st.markdown(f"**{AGENT_LABELS[key]}**")
                        st.json(s["raw_data"])

    except Exception as e:
        status.empty()
        st.error(f"Analysis failed: {str(e)}")
        st.exception(e)

elif run_btn:
    st.warning("Please enter a ticker symbol.")

st.divider()
st.caption("⚠️ For educational purposes only. Not financial advice. Always DYOR.")