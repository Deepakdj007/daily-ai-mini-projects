# agents/macro_agent.py
import os
import json
import requests
from agents import llm, safe_parse_json
from state import MarketMindState, AgentSignal


def fetch_fred(series_id: str, api_key: str, limit: int = 5) -> list:
    """Fetch recent values from a FRED economic data series."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id, "api_key": api_key,
        "file_type": "json", "sort_order": "desc", "limit": limit,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    # FRED uses "." as placeholder for missing/unreleased data — filter those out
    return [
        {"date": o["date"], "value": round(float(o["value"]), 4)}
        for o in resp.json().get("observations", [])
        if o["value"] != "."
    ]


def fetch_fear_greed() -> dict:
    """Fetch the Crypto Fear & Greed Index. Free, no key required."""
    try:
        resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        resp.raise_for_status()
        entry = resp.json()["data"][0]
        return {"value": int(entry["value"]), "label": entry["value_classification"]}
    except Exception:
        return {"value": 50, "label": "Neutral"}


def macro_agent(state: MarketMindState) -> dict:
    ticker = state["ticker"]
    asset_type = state["asset_type"]
    fred_key = os.getenv("FRED_API_KEY")
    raw_data = {}

    try:
        # DTWEXBGS = US Dollar Index (broad, trade-weighted)
        try:
            dxy = fetch_fred("DTWEXBGS", fred_key)
            if len(dxy) >= 2:
                raw_data["dxy_current"] = dxy[0]["value"]
                raw_data["dxy_5day_change"] = round(dxy[0]["value"] - dxy[-1]["value"], 4)
        except Exception:
            raw_data["dxy"] = "unavailable"

        # FEDFUNDS = Federal Funds Rate
        try:
            fed = fetch_fred("FEDFUNDS", fred_key, limit=2)
            if fed:
                raw_data["fed_funds_rate"] = fed[0]["value"]
        except Exception:
            raw_data["fed_funds_rate"] = "unavailable"

        # T10Y2Y = 10Y minus 2Y yield spread (negative = inverted = recession risk)
        try:
            yc = fetch_fred("T10Y2Y", fred_key, limit=2)
            if yc:
                raw_data["yield_curve_spread"] = yc[0]["value"]
                raw_data["yield_curve_inverted"] = yc[0]["value"] < 0
        except Exception:
            raw_data["yield_curve_spread"] = "unavailable"

        # Fear & Greed only meaningful for crypto
        if asset_type == "crypto":
            fg = fetch_fear_greed()
            raw_data["fear_greed_value"] = fg["value"]
            raw_data["fear_greed_label"] = fg["label"]

        prompt = f"""You are a macro economist assessing conditions for {ticker} ({asset_type}).

Data:
{json.dumps(raw_data, indent=2)}

Key rules:
- DXY rising = bearish for risk assets (stocks and crypto)
- Fed rate above 4.5% = restrictive policy, bearish pressure on equities
- Inverted yield curve = recession risk, bearish
- Fear & Greed below 25 = extreme fear (contrarian buy signal for brave)
- Fear & Greed above 75 = extreme greed (contrarian sell signal)

Return ONLY valid JSON with exactly these fields, no other text:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "one sentence under 20 words"}}"""

        result_raw = llm.invoke(prompt)
        result = safe_parse_json(result_raw.content)

        return {"agent_signals": [AgentSignal(
            agent="macro",
            signal=result["signal"],
            confidence=float(result["confidence"]),
            summary=result["summary"],
            raw_data=raw_data
        )]}

    except Exception as e:
        return {"agent_signals": [AgentSignal(
            agent="macro", signal="NEUTRAL", confidence=0.0,
            summary=f"Macro agent error: {str(e)[:60]}", raw_data={}
        )]}