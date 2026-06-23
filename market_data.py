import datetime
import math
import os
from multiprocessing.util import info
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from streamlit.errors import StreamlitSecretNotFoundError


# Prefer environment variable (local dev). Only try st.secrets if env var absent;
# guard against missing secrets.toml to avoid StreamlitSecretNotFoundError.
env_api = os.getenv("ALPHA_VANTAGE_KEY")
try:
    secret_api = st.secrets.get("ALPHA_VANTAGE_KEY")
except StreamlitSecretNotFoundError:
    secret_api = None
API_KEY = env_api or secret_api or "REPLACE_ME"


def _safe_info_get(info: Dict[str, Any], keys: list) -> Optional[float]:
    """
    Return the first present numeric value from keys in info, or None.
    Do NOT include raw price keys when requesting ratio fields like P/E.
    """
    for k in keys or []:
        if not k:
            continue
        v = info.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None


def get_market_data(ticker: str) -> Dict[str, Any]:
    """
    Fetch 5 years of price history for `ticker` and compute metrics:
      - daily_returns (list)
      - annualized_volatility (float)
      - sharpe_ratio (float) using risk-free rate = 4.5% (annual)
      - max_drawdown (float, negative number)
      - current_pe, current_pb, debt_to_equity (floats or None)

    Returns a dictionary with results or with an 'error' key on failure.
    """
    result: Dict[str, Any] = {"ticker": ticker.upper()}
    try:
        t = yf.Ticker(ticker)
        # Fetch 5 years of daily data
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=365 * 5 + 30)  # a bit more to ensure ~5 years
        df = t.history(start=start.date(), end=end.date(), interval="1d", auto_adjust=False)

        if df is None or df.empty or "Close" not in df.columns:
            return {"ticker": ticker.upper(), "error": "No price history found for ticker."}

        # Ensure sorted by date ascending
        df = df.sort_index()

        # Daily returns
        daily_returns = df["Close"].pct_change().dropna()
        if daily_returns.empty:
            return {"ticker": ticker.upper(), "error": "Insufficient price data to compute returns."}

        # Annualized metrics
        trading_days = 252
        mean_daily = float(daily_returns.mean())
        ann_return = mean_daily * trading_days
        ann_vol = float(daily_returns.std(ddof=0) * math.sqrt(trading_days))

        # Sharpe Ratio (annual) using risk-free rate of 4.5% (0.045)
        rf = 0.045
        sharpe = None
        if ann_vol and not math.isclose(ann_vol, 0.0):
            sharpe = float((ann_return - rf) / ann_vol)

        # Maximum Drawdown
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative / rolling_max) - 1
        max_drawdown = float(drawdown.min())

        # Fundamental metrics from yfinance info (robust to missing keys)
        info = {}
        try:
            info = t.get_info() if hasattr(t, "get_info") else t.info  # support different yfinance versions
        except Exception:
            # as fallback, try .info attribute
            try:
                info = t.info
            except Exception:
                info = {}

        current_pe = _safe_info_get(info, ["trailingPE", "peRatio", "trailingPERatio"])
        current_pb = _safe_info_get(info, ["priceToBook", "priceToBookTrailing12Months", "pbRatio"])
        debt_to_equity = _safe_info_get(info, ["debtToEquity", "totalDebt/totalStockholdersEquity", "debtToEquityRatio"])

        # Build result dict
        result.update(
            {
                "daily_returns": [float(x) for x in daily_returns.tolist()],
                "annualized_return": float(ann_return),
                "annualized_volatility": float(ann_vol),
                "sharpe_ratio": (float(sharpe) if sharpe is not None else None),
                "max_drawdown": float(max_drawdown),
                "current_pe": current_pe,
                "current_pb": current_pb,
                "debt_to_equity": debt_to_equity,
                "price_last": float(df["Close"].iloc[-1]),
                "history_start": str(df.index[0].date()),
                "history_end": str(df.index[-1].date()),
            }
        )

        return result

    except Exception as exc:
        return {"ticker": ticker.upper(), "error": f"Exception while fetching data: {exc}"}


def _format_float(x: Optional[float]) -> str:
    return "N/A" if x is None else f"{x:.4f}"


def _print_summary(data: Dict[str, Any]) -> None:
    if "error" in data:
        print(f"Error for {data.get('ticker', '')}: {data['error']}")
        return

    print(f"Ticker: {data.get('ticker')}")
    print(f"Price (last): {data.get('price_last')}")
    print(f"History: {data.get('history_start')} -> {data.get('history_end')}")
    print()
    print("Metrics:")
    print(f"  Annualized Return: {data.get('annualized_return'):.4f}")
    print(f"  Annualized Volatility: {_format_float(data.get('annualized_volatility'))}")
    sr = data.get("sharpe_ratio")
    print(f"  Sharpe Ratio (rf=4.5%): {_format_float(sr)}")
    print(f"  Maximum Drawdown: {data.get('max_drawdown'):.4f}")
    print()
    print("Fundamentals (from yfinance):")
    print(f"  P/E: {_format_float(data.get('current_pe'))}")
    print(f"  P/B: {_format_float(data.get('current_pb'))}")
    print(f"  Debt/Equity: {_format_float(data.get('debt_to_equity'))}")
    print()
    print(f"Daily returns sample (first 5): {data.get('daily_returns')[:5]}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch 5y market data and compute metrics for a ticker.")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker symbol (e.g. AAPL)")
    args = parser.parse_args()

    out = get_market_data(args.ticker)
    _print_summary(out)