# agents/synthesis_agent.py
from agents import llm, safe_parse_json
from state import MarketMindState

AGENT_WEIGHTS = {
    "price":     0.25,   # technical analysis
    "sentiment": 0.20,   # news sentiment
    "onchain":   0.20,   # market structure
    "macro":     0.20,   # macro conditions
    "risk":      0.15,   # risk-adjusted view
}

SIGNAL_SCORES = {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}


def synthesis_agent(state: MarketMindState) -> dict:
    ticker = state["ticker"]
    signals = state["agent_signals"]

    # Build readable summary + compute weighted score
    summaries = []
    weighted_score = 0.0
    total_weight = 0.0

    for s in signals:
        w = AGENT_WEIGHTS.get(s["agent"], 0.2)
        weighted_score += SIGNAL_SCORES.get(s["signal"], 0) * w * s["confidence"]
        total_weight += w
        summaries.append(
            f"- {s['agent'].upper()} Agent: {s['signal']} "
            f"(confidence: {s['confidence']:.0%}) — {s['summary']}"
        )

    normalized = weighted_score / total_weight if total_weight > 0 else 0
    preliminary = (
        "BULLISH" if normalized > 0.15
        else "BEARISH" if normalized < -0.15
        else "NEUTRAL"
    )

    prompt = f"""You are a senior portfolio analyst making a final decision on {ticker}.

Five specialist AI agents have completed their independent analysis:

{chr(10).join(summaries)}

Weighted signal score: {normalized:.3f}  (-1.0 = fully bearish, +1.0 = fully bullish)
Preliminary direction: {preliminary}

Synthesize all signals. Rules:
- Strong agreement across agents = high confidence
- Mixed signals = lower confidence, lean toward HOLD
- Risk signals override bullish technicals when VIX is elevated

Return ONLY valid JSON with exactly these fields, no other text:
{{"verdict": "BUY" or "HOLD" or "SELL", "confidence": 0.0 to 1.0, "reasoning": "2-3 sentences on key factors"}}"""

    response = llm.invoke(prompt)
    result = safe_parse_json(response.content)

    return {
        "final_verdict": result["verdict"],
        "final_confidence": float(result["confidence"]),
        "final_reasoning": result["reasoning"],
    }