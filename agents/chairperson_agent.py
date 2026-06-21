# ...existing code...
import os
import re
import json
from typing import Dict, Any, List

from agents.base import get_deepseek_client  # shared LLM client


def aggregate(agent_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    agent_outputs: list of {"agent": name, "ticker": T, "result": {"recommendation","confidence","reason"}}
    Returns final dict with recommendation, confidence (0-100), summary (2-3 sentences) and individual votes.

    Attempts to synthesize a professional chair summary via the LLM. On failure (or when MOCK_CHAIR is set)
    falls back to deterministic aggregation.
    """

    # Development mock: set MOCK_CHAIR=1 to return a canned committee decision
    if os.getenv("MOCK_CHAIR"):
        recomm = os.getenv("MOCK_CHAIR_RECOMM", "HOLD").upper()
        conf = int(os.getenv("MOCK_CHAIR_CONF", "60"))
        if recomm == "BUY":
            votes = {
                "BUY": [{"agent": "value_agent", "confidence": 80, "reason": "Mock: favorable fundamentals"}],
                "HOLD": [],
                "SELL": [],
            }
        elif recomm == "SELL":
            votes = {
                "BUY": [],
                "HOLD": [],
                "SELL": [{"agent": "risk_agent", "confidence": 75, "reason": "Mock: elevated risk"}],
            }
        else:
            votes = {
                "BUY": [],
                "HOLD": [{"agent": "growth_agent", "confidence": 60, "reason": "Mock: neutral view"}],
                "SELL": [],
            }
        summary = f"Mocked chairperson response: {recomm} (development mode)"
        result = {"final_recommendation": recomm, "final_confidence": conf, "summary": summary, "votes": votes}
        result.setdefault("risk_report", {"expected_drawdown_pct": None, "tail_risk_comment": "", "volatility": None})
        result.setdefault("suggested_portfolio_allocation", {"cash_pct": None, "equity_pct": None, "target_weight_pct": {}})
        return result

    # Build a compact representation to send to the LLM
    lines = []
    for out in agent_outputs:
        r = out.get("result", {})
        rec = (r.get("recommendation") or "HOLD").upper()
        conf = int(r.get("confidence") or 50)
        reason = (r.get("reason") or "").replace("\n", " ").strip()
        lines.append(f"- {out.get('agent')}: {rec} ({conf}%) — {reason}")

    votes_block = "\n".join(lines)
    chair_prompt = (
        "You are the investment committee chair. Review the analyst opinions and produce a JSON object ONLY.\n\n"
        "Instructions:\n"
        "1) Identify major agreements (short list).\n"
        "2) Identify major disagreements (short list).\n"
        "3) State the strongest bullish argument (one sentence).\n"
        "4) State the strongest bearish argument (one sentence).\n"
        "5) Issue a final recommendation (BUY, HOLD, or SELL) and a numeric confidence 0-100.\n\n"
        "Respond with a JSON object exactly like:\n"
        '{"final_recommendation":"BUY|HOLD|SELL","final_confidence":int,'
        '"summary":"2-3 sentence summary","agreements":[],"disagreements":[],"bullish":"",'
        '"bearish":"","votes":{}}\n\n'
        f"Analyst votes:\n{votes_block}\n\n"
        "Be concise and professional. Output only valid JSON."
    )

    try:
        client = get_deepseek_client()
        raw = client.chat("You are a succinct investment committee chair.", chair_prompt, max_tokens=500)

        # Extract JSON from LLM response robustly
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = None
        if m:
            payload = json.loads(m.group(0))
        else:
            payload = json.loads(raw)

        # Normalize and sanitize fields
        payload["final_recommendation"] = str(payload.get("final_recommendation", "HOLD")).upper()
        try:
            payload["final_confidence"] = int(float(payload.get("final_confidence", 50)))
        except Exception:
            payload["final_confidence"] = 50
        payload.setdefault("summary", "")
        payload.setdefault("votes", {})
        payload.setdefault("agreements", [])
        payload.setdefault("disagreements", [])
        payload.setdefault("bullish", "")
        payload.setdefault("bearish", "")
        result = {
            "final_recommendation": payload["final_recommendation"],
            "final_confidence": payload["final_confidence"],
            "summary": payload["summary"],
            "votes": payload["votes"],
            "agreements": payload["agreements"],
            "disagreements": payload["disagreements"],
            "bullish": payload["bullish"],
            "bearish": payload["bearish"],
        }
        result.setdefault("risk_report", {"expected_drawdown_pct": None, "tail_risk_comment": "", "volatility": None})
        result.setdefault("suggested_portfolio_allocation", {"cash_pct": None, "equity_pct": None, "target_weight_pct": {}})
        return result

    except Exception:
        # Fallback deterministic aggregation (existing heuristic)
        votes = {"BUY": [], "HOLD": [], "SELL": []}
        for out in agent_outputs:
            r = out.get("result", {})
            rec = (r.get("recommendation") or "HOLD").upper()
            conf = int(r.get("confidence") or 50)
            reason = r.get("reason", "")
            votes.setdefault(rec, []).append({"agent": out.get("agent"), "confidence": conf, "reason": reason})

        # weighted score: BUY=+1 HOLD=0 SELL=-1 weighted by confidence/100
        score = 0.0
        total_weight = 0.0
        for label, sign in [("BUY", 1), ("HOLD", 0), ("SELL", -1)]:
            for v in votes.get(label, []):
                w = v["confidence"] / 100.0
                score += sign * w
                total_weight += w

        if total_weight == 0:
            overall_rec = "HOLD"
        else:
            avg = score / total_weight
            if avg > 0.15:
                overall_rec = "BUY"
            elif avg < -0.15:
                overall_rec = "SELL"
            else:
                overall_rec = "HOLD"

        all_confs = []
        for lst in votes.values():
            for v in lst:
                all_confs.append(v["confidence"])
        overall_conf = int(sum(all_confs) / len(all_confs)) if all_confs else 50

        reasons = []
        for label in ["BUY", "HOLD", "SELL"]:
            items = votes.get(label, [])
            if items:
                reasons.append(f"{len(items)} agent(s) recommended {label} (sample reason: {items[0]['reason'][:120]})")
        summary = " ".join(reasons)[:600] or "No clear signals; insufficient data."

        result = {
            "final_recommendation": overall_rec,
            "final_confidence": overall_conf,
            "summary": summary,
            "votes": votes,
        }
        result.setdefault("risk_report", {"expected_drawdown_pct": None, "tail_risk_comment": "", "volatility": None})
        result.setdefault("suggested_portfolio_allocation", {"cash_pct": None, "equity_pct": None, "target_weight_pct": {}})
        return result
# ...existing code...