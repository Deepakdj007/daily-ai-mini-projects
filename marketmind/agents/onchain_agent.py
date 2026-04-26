# agents/onchain_agent.py
import json
import yfinance as yf
from pycoingecko import CoinGeckoAPI
from agents import llm, safe_parse_json
from state import MarketMindState, AgentSignal

COINGECKO_IDS = {
    "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
    "BNB": "binancecoin", "DOGE": "dogecoin", "ADA": "cardano",
    "AVAX": "avalanche-2", "DOT": "polkadot", "MATIC": "matic-network",
    "LINK": "chainlink", "XRP": "ripple", "LTC": "litecoin",
}


def onchain_agent(state: MarketMindState) -> dict:
    ticker = state["ticker"]
    asset_type = state["asset_type"]

    try:
        if asset_type == "crypto":
            coin_id = COINGECKO_IDS.get(ticker.upper())
            if not coin_id:
                return {"agent_signals": [AgentSignal(
                    agent="onchain", signal="NEUTRAL", confidence=0.3,
                    summary=f"{ticker} not in supported crypto list", raw_data={}
                )]}

            cg = CoinGeckoAPI()
            data = cg.get_coin_by_id(
                coin_id, localization=False, tickers=False,
                market_data=True, community_data=False, developer_data=False,
            )
            market = data.get("market_data", {})
            market_cap = market.get("market_cap", {}).get("usd", 0)
            total_vol = market.get("total_volume", {}).get("usd", 0)

            raw_data = {
                "market_cap_rank": data.get("market_cap_rank", "N/A"),
                "price_change_24h_pct": round(market.get("price_change_percentage_24h", 0), 2),
                "price_change_7d_pct": round(market.get("price_change_percentage_7d", 0), 2),
                "volume_to_market_cap_ratio": round(total_vol / market_cap, 4) if market_cap > 0 else 0,
                "ath_change_pct": round(market.get("ath_change_percentage", {}).get("usd", 0), 2),
            }

        else:
            stock = yf.Ticker(ticker)
            info = stock.info
            avg_vol = max(info.get("averageVolume", 1), 1)
            raw_data = {
                "volume_ratio": round(info.get("volume", 0) / avg_vol, 3),
                "market_cap": info.get("marketCap", 0),
                "short_ratio": info.get("shortRatio", 0),
                "beta": info.get("beta", 1.0),
            }

        prompt = f"""You are a market microstructure analyst. Analyze this data for {ticker} ({asset_type}):

{json.dumps(raw_data, indent=2)}

Focus on: volume patterns, market activity, structural signals.

Return ONLY valid JSON with exactly these fields, no other text:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "one sentence under 20 words"}}"""

        result_raw = llm.invoke(prompt)
        result = safe_parse_json(result_raw.content)

        return {"agent_signals": [AgentSignal(
            agent="onchain",
            signal=result["signal"],
            confidence=float(result["confidence"]),
            summary=result["summary"],
            raw_data=raw_data
        )]}

    except Exception as e:
        return {"agent_signals": [AgentSignal(
            agent="onchain", signal="NEUTRAL", confidence=0.0,
            summary=f"On-chain agent error: {str(e)[:60]}", raw_data={}
        )]}