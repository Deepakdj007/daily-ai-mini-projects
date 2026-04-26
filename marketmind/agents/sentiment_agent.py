# agents/sentiment_agent.py
import os
from datetime import datetime, timedelta
from newsapi import NewsApiClient
from agents import llm, safe_parse_json
from state import MarketMindState, AgentSignal

CRYPTO_NAMES = {
    "BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana",
    "BNB": "Binance Coin", "DOGE": "Dogecoin", "ADA": "Cardano",
    "AVAX": "Avalanche", "DOT": "Polkadot", "MATIC": "Polygon",
    "LINK": "Chainlink", "XRP": "Ripple", "LTC": "Litecoin",
}


def sentiment_agent(state: MarketMindState) -> dict:
    ticker = state["ticker"]
    asset_type = state["asset_type"]

    try:
        newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))

        # Full name gets better results than ticker symbol
        query = CRYPTO_NAMES.get(ticker, ticker) if asset_type == "crypto" else ticker

        # 3-day window — free tier has a ~24h delay on some regions
        from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        response = newsapi.get_everything(
            q=query, from_param=from_date,
            language="en", sort_by="relevancy", page_size=10,
        )

        headlines = [
            a["title"] for a in response.get("articles", [])
            if a.get("title") and a["title"] != "[Removed]"
        ][:10]

        if not headlines:
            return {"agent_signals": [AgentSignal(
                agent="sentiment", signal="NEUTRAL", confidence=0.3,
                summary="No recent news found", raw_data={"headline_count": 0}
            )]}

        headlines_text = "\n".join(f"- {h}" for h in headlines)

        prompt = f"""You are a financial sentiment analyst. Rate the market sentiment from these recent headlines about {ticker}:

{headlines_text}

Return ONLY valid JSON with exactly these fields, no other text:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "one sentence under 20 words", "positive_count": number, "negative_count": number}}"""

        result_raw = llm.invoke(prompt)
        result = safe_parse_json(result_raw.content)

        return {"agent_signals": [AgentSignal(
            agent="sentiment",
            signal=result["signal"],
            confidence=float(result["confidence"]),
            summary=result["summary"],
            raw_data={
                "headline_count": len(headlines),
                "positive_count": result.get("positive_count", 0),
                "negative_count": result.get("negative_count", 0),
                "sample_headline": headlines[0],
            }
        )]}

    except Exception as e:
        return {"agent_signals": [AgentSignal(
            agent="sentiment", signal="NEUTRAL", confidence=0.0,
            summary=f"Sentiment agent error: {str(e)[:60]}", raw_data={}
        )]}