# ...existing code...
import os
import json
import re
from typing import Dict, Any

from agents.base import get_deepseek_client  # shared LLM client

SYSTEM = (
    "You are a quantitative analyst. Focus on Monte Carlo outputs, probabilities, percentiles, "
    "and mean/variance signals. Produce JSON with recommendation (BUY/HOLD/SELL), confidence (0-100), reason."
)


def analyze(ticker: str, md: Dict[str, Any], mc: Dict[str, Any]) -> Dict[str, Any]:
    # use shared client (reads env or Streamlit secrets)
    client = get_deepseek_client()

    prompt = (
        f"Ticker: {ticker}\n"
        f"Current price: {md.get('price_last')}\n"
        f"P(>10%): {mc.get('probability_gain_more_than_10')}  P(<-20%): {mc.get('probability_loss_more_than_20')}\n"
        f"5th/50th/95th percentiles: {mc.get('percentile_5')}/{mc.get('percentile_50')}/{mc.get('percentile_95')}\n\n"
        "Using the Monte Carlo stats and fundamentals, produce the short JSON described in the system role."
    )
    try:
        raw = client.chat(SYSTEM, prompt)
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(m.group(0)) if m else json.loads(raw)
    except Exception as e:
        payload = {"recommendation": "HOLD", "confidence": 50, "reason": f"LLM/quant error: {e}"}

    payload["recommendation"] = str(payload.get("recommendation", "HOLD")).upper()
    try:
        payload["confidence"] = int(float(payload.get("confidence", 50)))
    except Exception:
        payload["confidence"] = 50
    payload.setdefault("reason", "")

    return {"agent": "quant_agent", "ticker": ticker.upper(), "result": payload}
# ...existing code...