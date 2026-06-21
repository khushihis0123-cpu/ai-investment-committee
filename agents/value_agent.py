import os
import json
import re
from typing import Dict, Any

from agents.base import get_deepseek_client  # shared LLM client

SYSTEM = (
    "You are Warren Buffett-like value investor. Be conservative, focus on P/E, P/B, debt, free cash flow, "
    "return on capital. Produce a short JSON object with keys: recommendation (BUY/HOLD/SELL), confidence (0-100), reason (string)."
)


def analyze(ticker: str, md: Dict[str, Any]) -> Dict[str, Any]:
    """
    md: market_data dict from market_data.get_market_data(...)
    """
    # use the shared client factory (reads env or Streamlit secrets)
    client = get_deepseek_client()

    prompt = (
        f"Ticker: {ticker}\n\nFundamentals:\n"
        f"P/E: {md.get('current_pe')}\nP/B: {md.get('current_pb')}\nDebt/Equity: {md.get('debt_to_equity')}\n"
        f"Price: {md.get('price_last')}\nAnnualized Volatility: {md.get('annualized_volatility')}\n"
        f"Max Drawdown: {md.get('max_drawdown')}\nFree cash flow info may be missing; infer from available info.\n\n"
        "As a value investor, produce the JSON described in system role."
    )
    try:
        raw = client.chat(SYSTEM, prompt)
        # try to extract json substring
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            payload = json.loads(m.group(0))
        else:
            payload = json.loads(raw)
    except Exception as e:
        # fallback heuristic
        payload = {"recommendation": "HOLD", "confidence": 50, "reason": f"LLM error or missing data: {e}"}
    # sanitize
    payload["recommendation"] = str(payload.get("recommendation", "HOLD")).upper()
    try:
        payload["confidence"] = int(float(payload.get("confidence", 50)))
    except Exception:
        payload["confidence"] = 50
    payload.setdefault("reason", "")
    return {"agent": "value_agent", "ticker": ticker.upper(), "result": payload}