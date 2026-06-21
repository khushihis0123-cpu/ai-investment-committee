import os
from typing import Optional
import requests

# read env first, then Streamlit secrets if available
try:
    import streamlit as st  # type: ignore
    _secrets = st.secrets
except Exception:
    _secrets = {}

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or _secrets.get("DEEPSEEK_API_KEY")
DEEPSEEK_BASE = os.getenv("DEEPSEEK_BASE_URL") or _secrets.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = "deepseek-chat"


class DeepSeekClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: str = MODEL):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.base = (base_url or DEEPSEEK_BASE).rstrip("/")
        self.model = model

    def chat(self, system: str, user_prompt: str, max_tokens: int = 800) -> str:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set in environment or Streamlit secrets")
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
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code == 402:
                raise RuntimeError("DeepSeek API error: Insufficient balance (HTTP 402).")
            resp.raise_for_status()
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
        except requests.HTTPError as e:
            text = getattr(e.response, "text", "")[:800]
            raise RuntimeError(f"DeepSeek HTTP error {getattr(e.response,'status_code',None)}: {text}")
        except Exception as e:
            raise RuntimeError(f"DeepSeek request failed: {e}")


def get_deepseek_client() -> DeepSeekClient:
    return DeepSeekClient()