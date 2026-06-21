# ...existing code...
import os
import json
import re
from typing import Dict, Any

from agents.base import get_deepseek_client  # shared LLM client

SYSTEM = (
    "You are a risk-focused investor. Focus on volatility, drawdowns, tail risk, leverage, liquidity, "
    "and downside scenario analysis. Produce JSON with recommendation (BUY/HOLD/SELL), confidence (0-100), reason."
)


def analyze(ticker: str, md: Dict[str, Any]) -> Dict[str, Any]:
    # use shared client (reads env or Streamlit secrets)
    client = get_deepseek_client()

    prompt = (
        f"Ticker: {ticker}\n"
        f"Price: {md.get('price_last')}  Annualized Vol: {md.get('annualized_volatility')}\n"
        f"Max Drawdown: {md.get('max_drawdown')}  Debt/Equity: {md.get('debt_to_equity')}\n\n"
        "Assess risk factors and return a short JSON with recommendation, confidence, and reason."
    )
    try:
        raw = client.chat(SYSTEM, prompt)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(m.group(0)) if m else json.loads(raw)
    except Exception as e:
        payload = {"recommendation": "HOLD", "confidence": 50, "reason": f"LLM error or missing data: {e}"}

    payload["recommendation"] = str(payload.get("recommendation", "HOLD")).upper()
    try:
        payload["confidence"] = int(float(payload.get("confidence", 50)))
    except Exception:
        payload["confidence"] = 50
    payload.setdefault("reason", "")

    return {"agent": "risk_agent", "ticker": ticker.upper(), "result": payload}
# ...existing code...