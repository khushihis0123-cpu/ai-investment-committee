# ...existing code...
import os
from typing import Dict, Any, List


def aggregate(agent_outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    agent_outputs: list of {"agent": name, "ticker": T, "result": {"recommendation","confidence","reason"}}
    Returns final dict with recommendation, confidence (0-100), summary (2-3 sentences) and individual votes.
    """

    # Development mock: set MOCK_CHAIR=1 to return a canned committee decision
    # Optionally set MOCK_CHAIR_RECOMM (BUY/HOLD/SELL) and MOCK_CHAIR_CONF (0-100).
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
        return {"final_recommendation": recomm, "final_confidence": conf, "summary": summary, "votes": votes}

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

    # overall confidence: mean of confidences weighted toward agreement
    all_confs = []
    for lst in votes.values():
        for v in lst:
            all_confs.append(v["confidence"])
    overall_conf = int(sum(all_confs) / len(all_confs)) if all_confs else 50

    # build short summary 2-3 sentences
    reasons = []
    for label in ["BUY", "HOLD", "SELL"]:
        items = votes.get(label, [])
        if items:
            reasons.append(f"{len(items)} agent(s) recommended {label} (sample reason: {items[0]['reason'][:120]})")
    summary = " ".join(reasons)[:600] or "No clear signals; insufficient data."

    return {
        "final_recommendation": overall_rec,
        "final_confidence": overall_conf,
        "summary": summary,
        "votes": votes,
    }
# ...existing code...