# ...existing code...
import os
import json
import re
from typing import Dict, Any
import requests
# ...existing code...

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = "deepseek-chat"


class DeepSeekClient:
    def __init__(self, api_key: str = DEEPSEEK_API_KEY, base_url: str = DEEPSEEK_BASE, model: str = MODEL):
        self.api_key = api_key
        self.base = base_url.rstrip("/")
        self.model = model

    def chat(self, system: str, user_prompt: str, max_tokens: int = 800) -> str:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set in environment")
        url = f"{self.base}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # best-effort extraction
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data)


SYSTEM = (
    "You are Warren Buffett-like value investor. Be conservative, focus on P/E, P/B, debt, free cash flow, "
    "return on capital. Produce a short JSON object with keys: recommendation (BUY/HOLD/SELL), confidence (0-100), reason (string)."
)


def analyze(ticker: str, md: Dict[str, Any]) -> Dict[str, Any]:
    """
    md: market_data dict from market_data.get_market_data(...)
    """
    client = DeepSeekClient()
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
        import re

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            payload = json.loads(m.group(0))
        else:
            payload = json.loads(raw)
    except Exception as e:
        # fallback heuristic
        payload = {"recommendation": "HOLD", "confidence": 50, "reason": f"LLM error or missing data: {e}"}
    # sanitize
    payload["recommendation"] = payload.get("recommendation", "HOLD").upper()
    try:
        payload["confidence"] = int(float(payload.get("confidence", 50)))
    except Exception:
        payload["confidence"] = 50
    payload.setdefault("reason", "")
    return {"agent": "value_agent", "ticker": ticker.upper(), "result": payload}