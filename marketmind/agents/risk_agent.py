# agents/risk_agent.py
import json
import numpy as np
import pandas as pd
import yfinance as yf
from agents import llm, safe_parse_json
from state import MarketMindState, AgentSignal


def annualized_vol(returns: np.ndarray) -> float:
    return round(float(np.std(returns)) * np.sqrt(252) * 100, 2)


def max_drawdown(prices: pd.Series) -> float:
    dd = (prices - prices.cummax()) / prices.cummax()
    return round(float(dd.min()) * 100, 2)


def sharpe(returns: np.ndarray) -> float:
    mean_r = float(np.mean(returns)) * 252
    std_r = float(np.std(returns)) * np.sqrt(252)
    return round(mean_r / std_r, 3) if std_r > 0 else 0.0


def risk_agent(state: MarketMindState) -> dict:
    ticker = state["ticker"]
    yfin_ticker = f"{ticker}-USD" if state["asset_type"] == "crypto" else ticker

    try:
        data = yf.download(yfin_ticker, period="180d", interval="1d", progress=False)

        if data.empty:
            return {"agent_signals": [AgentSignal(
                agent="risk", signal="NEUTRAL", confidence=0.0,
                summary="No data available for risk analysis", raw_data={}
            )]}

        close = data["Close"].squeeze()
        all_returns = close.pct_change().dropna().values

        raw_data = {
            "volatility_30d_pct": annualized_vol(close.pct_change().dropna().tail(30).values),
            "volatility_90d_pct": annualized_vol(close.pct_change().dropna().tail(90).values),
            "max_drawdown_180d_pct": max_drawdown(close),
            "sharpe_ratio_180d": sharpe(all_returns),
        }

        # VIX = CBOE Volatility Index — market-wide fear proxy
        try:
            vix_data = yf.download("^VIX", period="5d", interval="1d", progress=False)
            if not vix_data.empty:
                vix = round(float(vix_data["Close"].squeeze().iloc[-1]), 2)
                raw_data["vix"] = vix
                raw_data["vix_level"] = (
                    "low fear" if vix < 20
                    else "elevated fear" if vix < 30
                    else "high fear"
                )
        except Exception:
            raw_data["vix"] = "unavailable"

        prompt = f"""You are a risk analyst evaluating {ticker}:

{json.dumps(raw_data, indent=2)}

Key rules:
- Volatility above 50% annualized = high risk for stocks (crypto threshold: 80%)
- Max drawdown worse than -30% = history of severe crashes
- Sharpe ratio above 1.0 = good risk-adjusted returns
- VIX above 30 = elevated crash risk in the market

Is this a favorable risk environment to enter a new position?

Return ONLY valid JSON with exactly these fields, no other text:
{{"signal": "BULLISH" or "BEARISH" or "NEUTRAL", "confidence": 0.0 to 1.0, "summary": "one sentence under 20 words"}}"""

        result_raw = llm.invoke(prompt)
        result = safe_parse_json(result_raw.content)

        return {"agent_signals": [AgentSignal(
            agent="risk",
            signal=result["signal"],
            confidence=float(result["confidence"]),
            summary=result["summary"],
            raw_data=raw_data
        )]}

    except Exception as e:
        return {"agent_signals": [AgentSignal(
            agent="risk", signal="NEUTRAL", confidence=0.0,
            summary=f"Risk agent error: {str(e)[:60]}", raw_data={}
        )]}