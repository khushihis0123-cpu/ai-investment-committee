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
    "You are a quantitative analyst. Interpret Monte Carlo probabilities and numerical metrics. Provide a JSON with "
    "recommendation (BUY/HOLD/SELL), confidence (0-100), reason."
)


def analyze(ticker: str, md: Dict[str, Any], mc: Dict[str, Any]) -> Dict[str, Any]:
    client = DeepSeekClient()
    prompt = (
        f"Ticker: {ticker}\nMarket metrics:\n"
        f"Annualized Return: {md.get('annualized_return')} Vol: {md.get('annualized_volatility')} MaxDD: {md.get('max_drawdown')}\n"
        f"Monte Carlo -> P(>+10%): {mc.get('probability_gain_more_than_10')} P(<-20%): {mc.get('probability_loss_more_than_20')}\n"
        f"Percentiles: 5th {mc.get('percentile_5')}, 50th {mc.get('percentile_50')}, 95th {mc.get('percentile_95')}\n\n"
        "Provide a concise JSON as specified."
    )
    try:
        raw = client.chat(SYSTEM, prompt)
        import re

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(m.group(0)) if m else json.loads(raw)
    except Exception as e:
        payload = {"recommendation": "HOLD", "confidence": 55, "reason": f"LLM/quant error: {e}"}
    payload["recommendation"] = payload.get("recommendation", "HOLD").upper()
    try:
        payload["confidence"] = int(float(payload.get("confidence", 55)))
    except Exception:
        payload["confidence"] = 55
    payload.setdefault("reason", "")
    return {"agent": "quant_agent", "ticker": ticker.upper(), "result": payload}