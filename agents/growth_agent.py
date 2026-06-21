# ...existing code...
import os
import json
import re
from typing import Dict, Any

from agents.base import get_deepseek_client  # shared LLM client

SYSTEM = (
    "You are a growth investor. Focus on revenue growth, earnings growth, margins, industry tailwinds, "
    "and scalability. Produce JSON with recommendation (BUY/HOLD/SELL), confidence (0-100), reason."
)


def analyze(ticker: str, md: Dict[str, Any], extra: Dict[str, Any] = None) -> Dict[str, Any]:
    # use shared client that reads env or Streamlit secrets
    client = get_deepseek_client()

    prompt = (
        f"Ticker: {ticker}\n"
        f"Available metrics (may be incomplete):\n"
        f"Annualized Return: {md.get('annualized_return')}\n"
        f"Annualized Volatility: {md.get('annualized_volatility')}\n"
        f"Max Drawdown: {md.get('max_drawdown')}\n"
        f"P/E: {md.get('current_pe')} P/B: {md.get('current_pb')}\n\n"
        "Using the above and general industry growth signals, give a short JSON as specified."
    )
    try:
        raw = client.chat(SYSTEM, prompt)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(m.group(0)) if m else json.loads(raw)
    except Exception as e:
        payload = {"recommendation": "HOLD", "confidence": 50, "reason": f"LLM error: {e}"}

    # sanitize
    payload["recommendation"] = str(payload.get("recommendation", "HOLD")).upper()
    try:
        payload["confidence"] = int(float(payload.get("confidence", 50)))
    except Exception:
        payload["confidence"] = 50
    payload.setdefault("reason", "")

    return {"agent": "growth_agent", "ticker": ticker.upper(), "result": payload}
# ...existing code...