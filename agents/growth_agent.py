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
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        url = f"{self.base}/v1/chat/completions"
        payload = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user_prompt}], "max_tokens": max_tokens}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(data)


SYSTEM = (
    "You are a growth investor. Focus on revenue growth, earnings growth, margins, industry tailwinds, "
    "and scalability. Produce JSON with recommendation (BUY/HOLD/SELL), confidence (0-100), reason."
)


def analyze(ticker: str, md: Dict[str, Any], extra: Dict[str, Any] = None) -> Dict[str, Any]:
    client = DeepSeekClient()
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
        import re

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(m.group(0)) if m else json.loads(raw)
    except Exception as e:
        payload = {"recommendation": "HOLD", "confidence": 50, "reason": f"LLM error: {e}"}
    payload["recommendation"] = payload.get("recommendation", "HOLD").upper()
    try:
        payload["confidence"] = int(float(payload.get("confidence", 50)))
    except Exception:
        payload["confidence"] = 50
    payload.setdefault("reason", "")
    return {"agent": "growth_agent", "ticker": ticker.upper(), "result": payload}