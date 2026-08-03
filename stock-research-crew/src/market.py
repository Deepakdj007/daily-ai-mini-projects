"""The four compact text blocks — one per analyst.

Every function here returns a short labelled string, never a DataFrame and never
an exception. That matters for two reasons: raw yfinance output is enormous
(Ticker.info alone is 166 keys, and the statements are full DataFrames) and four
agents read from this module at the same time.

Inputs:  a ticker symbol, e.g. "RELIANCE.NS".
Outputs: price_block / fundamentals_block / news_block / street_block strings.

Run it on its own to see exactly what the agents will see, and what it costs,
before spending a single token:
    PYTHONPATH=. uv run python -m src.market RELIANCE.NS
"""

from typing import Any

from src import fmt, yahoo
from src.config import HISTORY_PERIOD, NEWS_COUNT

UNAVAILABLE = "DATA UNAVAILABLE"


def price_block(symbol: str) -> str:
    """Trend, moving averages, 52-week position, volatility and drawdown."""
    try:
        history = yahoo.ticker(symbol).history(period=HISTORY_PERIOD)
    except Exception as exc:
        return f"{UNAVAILABLE}: price history for {symbol} ({type(exc).__name__})."
    if history.empty:
        return f"{UNAVAILABLE}: no price history for {symbol}."

    close = history["Close"].dropna()
    unit = yahoo.currency(symbol)
    last = close.iloc[-1]
    peak = close.max()

    def change(sessions: int) -> str:
        """Percentage move over a lookback, or n/a if the history is too short."""
        if len(close) <= sessions:
            return fmt.NA
        return fmt.growth(last, close.iloc[-sessions - 1])

    def dma(window: int) -> str:
        """Simple moving average, or n/a if there are not enough sessions."""
        if len(close) < window:
            return fmt.NA
        return fmt.price(close.rolling(window).mean().iloc[-1], unit)

    returns = close.pct_change().dropna()
    volatility = float(returns.std()) * (252**0.5) * 100 if len(returns) > 1 else None
    avg_volume = history["Volume"].tail(20).mean() if "Volume" in history else None

    return "\n".join(
        [
            f"PRICE AND TECHNICALS — {symbol} (period: {HISTORY_PERIOD}, {len(close)} sessions)",
            fmt.line("last close", fmt.price(last, unit)),
            fmt.line("change 1 week", change(5)),
            fmt.line("change 1 month", change(21)),
            fmt.line("change 6 months", change(126)),
            fmt.line("change over period", fmt.growth(last, close.iloc[0])),
            fmt.line("50-day MA", dma(50)),
            fmt.line("200-day MA", dma(200)),
            fmt.line(
                "period high / low",
                f"{fmt.price(peak, unit)} / {fmt.price(close.min(), unit)}",
            ),
            fmt.line("off period high", fmt.growth(last, peak)),
            fmt.line("annualised vol", f"{fmt.ratio(volatility)}%" if volatility else fmt.NA),
            fmt.line("avg volume (20d)", f"{avg_volume:,.0f}" if avg_volume else fmt.NA),
        ]
    )


def fundamentals_block(symbol: str) -> str:
    """Revenue and profit trend, margins, leverage, cash flow and valuation."""
    handle = yahoo.ticker(symbol)
    fields = yahoo.info(symbol)
    unit = yahoo.currency(symbol)
    try:
        income, balance, cash = handle.income_stmt, handle.balance_sheet, handle.cashflow
    except Exception as exc:
        return f"{UNAVAILABLE}: financial statements for {symbol} ({type(exc).__name__})."

    years = fmt.fiscal_years(income)
    if not years:
        return f"{UNAVAILABLE}: no filed statements for {symbol}."

    revenue, revenue_prev = fmt.row(income, "Total Revenue"), fmt.row(income, "Total Revenue", 1)
    profit, profit_prev = fmt.row(income, "Net Income"), fmt.row(income, "Net Income", 1)
    equity = fmt.row(balance, "Stockholders Equity")

    # Yahoo leaves returnOnEquity empty for plenty of Indian listings, but both
    # inputs are sitting right there in the statements — so derive it.
    roe = fields.get("returnOnEquity")
    if fmt.is_missing(roe) and not fmt.is_missing(profit) and not fmt.is_missing(equity):
        roe = float(profit) / float(equity)

    return "\n".join(
        [
            f"FUNDAMENTALS — {symbol} (fiscal years: {', '.join(years)})",
            fmt.line("revenue latest", f"{fmt.money(revenue, unit)}  (yoy {fmt.growth(revenue, revenue_prev)})"),
            fmt.line("net income latest", f"{fmt.money(profit, unit)}  (yoy {fmt.growth(profit, profit_prev)})"),
            fmt.line("EBITDA", fmt.money(fmt.row(income, "EBITDA"), unit)),
            fmt.line("operating income", fmt.money(fmt.row(income, "Operating Income"), unit)),
            fmt.line("diluted EPS", fmt.ratio(fmt.row(income, "Diluted EPS"))),
            fmt.line("operating cash flow", fmt.money(fmt.row(cash, "Operating Cash Flow"), unit)),
            fmt.line("free cash flow", fmt.money(fmt.row(cash, "Free Cash Flow"), unit)),
            fmt.line("capex", fmt.money(fmt.row(cash, "Capital Expenditure"), unit)),
            fmt.line("total debt", fmt.money(fmt.row(balance, "Total Debt"), unit)),
            fmt.line("net debt", fmt.money(fmt.row(balance, "Net Debt"), unit)),
            fmt.line("shareholder equity", fmt.money(equity, unit)),
            fmt.line("profit margin", fmt.pct(fields.get("profitMargins"))),
            fmt.line("return on equity", fmt.pct(roe)),
            fmt.line("debt / equity", fmt.ratio(fields.get("debtToEquity"))),
            fmt.line("trailing P/E", fmt.ratio(fields.get("trailingPE"))),
            fmt.line("forward P/E", fmt.ratio(fields.get("forwardPE"))),
            fmt.line("price / book", fmt.ratio(fields.get("priceToBook"))),
            fmt.line("dividend yield", fmt.pct(fields.get("dividendYield"), already_percent=True)),
            fmt.line("beta", fmt.ratio(fields.get("beta"))),
        ]
    )


def news_block(symbol: str) -> str:
    """Recent headlines with one-line summaries.

    Current yfinance nests every field under a "content" key, so the older
    item["title"] access pattern raises KeyError.
    """
    try:
        items = yahoo.ticker(symbol).get_news(count=NEWS_COUNT) or []
    except Exception as exc:
        return f"{UNAVAILABLE}: news for {symbol} ({type(exc).__name__})."
    if not items:
        return f"{UNAVAILABLE}: no recent headlines for {symbol}."

    lines = [f"RECENT NEWS — {symbol} ({len(items)} items, newest first)"]
    for item in items:
        content = item.get("content") or {}
        title = content.get("title") or "(untitled)"
        date = str(content.get("pubDate") or "")[:10]
        summary = (content.get("summary") or content.get("description") or "").strip()
        summary = " ".join(summary.split())[:240]
        lines.append(f"  [{date}] {title}")
        if summary:
            lines.append(f"      {summary}")
    return "\n".join(lines)


def street_block(symbol: str) -> str:
    """Analyst consensus, price targets and the next earnings date."""
    handle = yahoo.ticker(symbol)
    fields = yahoo.info(symbol)
    unit = yahoo.currency(symbol)
    lines = [f"ANALYST VIEW — {symbol}"]

    def spread(entry: Any) -> str:
        """One row of the buy/hold/sell distribution as a flat string."""
        return (
            f"strong buy {entry.get('strongBuy', 0)}, buy {entry.get('buy', 0)}, "
            f"hold {entry.get('hold', 0)}, sell {entry.get('sell', 0)}, "
            f"strong sell {entry.get('strongSell', 0)}"
        )

    try:
        recommendations = handle.recommendations
        if recommendations is not None and not recommendations.empty:
            lines.append(fmt.line("ratings now", spread(recommendations.iloc[0])))
            if len(recommendations) > 2:
                lines.append(fmt.line("ratings 2mo ago", spread(recommendations.iloc[2])))
    except Exception:
        lines.append(fmt.line("ratings", fmt.NA))

    lines.append(fmt.line("consensus label", fields.get("recommendationKey") or fmt.NA))
    lines.append(fmt.line("analysts covering", fields.get("numberOfAnalystOpinions") or fmt.NA))

    try:
        targets = handle.analyst_price_targets or {}
        if targets:
            mean_target = targets.get("mean")
            lines.append(
                fmt.line(
                    "price target",
                    f"mean {fmt.price(mean_target, unit)}, "
                    f"low {fmt.price(targets.get('low'), unit)}, "
                    f"high {fmt.price(targets.get('high'), unit)}",
                )
            )
            lines.append(fmt.line("implied upside", fmt.growth(mean_target, targets.get("current"))))
    except Exception:
        lines.append(fmt.line("price target", fmt.NA))

    try:
        # calendar carries the next earnings date without needing lxml, which
        # get_earnings_dates() requires and which is not a dependency here.
        calendar = handle.calendar or {}
        if calendar.get("Earnings Date"):
            lines.append(fmt.line("next earnings", calendar["Earnings Date"][0]))
        if calendar.get("Earnings Average") is not None:
            lines.append(fmt.line("EPS estimate", fmt.ratio(calendar["Earnings Average"])))
        if calendar.get("Revenue Average") is not None:
            lines.append(fmt.line("revenue estimate", fmt.money(calendar["Revenue Average"], unit)))
    except Exception:
        lines.append(fmt.line("next earnings", fmt.NA))

    return "\n".join(lines)


def _demo() -> None:
    """Print all four blocks for one ticker. Costs nothing — no model involved."""
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    symbol = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"

    ok, detail = yahoo.validate(symbol)
    print(f"validate({symbol}) -> {ok}: {detail}\n")
    if not ok:
        return
    for block in (price_block, fundamentals_block, news_block, street_block):
        text = block(symbol)
        print(text)
        print(f"  [{len(text)} chars, ~{len(text) // 4} tokens]\n")


if __name__ == "__main__":
    _demo()
