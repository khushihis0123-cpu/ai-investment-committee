import os
from typing import Dict, Any

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import traceback

# Quick import guard: display any import/runtime error on the Streamlit page
try:
    # ...existing imports that may fail...
    import market_data
    import agents
    # (keep additional imports as in your file)
except Exception as e:
    st.title("App failed to start")
    st.error("Import/runtime error — see traceback below")
    st.code(traceback.format_exc())
    st.stop()

# helper: select sim paths and compute metrics (must be defined before use)
def _select_sim_paths(mc: Dict[str, Any]):
    # try common keys that monte_carlo.run_monte_carlo might return
    for k in ("sim_paths", "paths", "simulated_paths", "sim_array", "simulations", "sim_matrix"):
        sp = mc.get(k)
        if sp is not None:
            return np.asarray(sp)
    return None


def compute_expected_drawdown(sim_arr: np.ndarray) -> float | None:
    """
    Mean of per-path maximum peak-to-trough drawdowns.
    Returns decimal fraction (e.g. 0.33 for 33% loss).
    """
    try:
        arr = np.asarray(sim_arr, dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        running_max = np.maximum.accumulate(arr, axis=1)
        drawdowns = (running_max - arr) / running_max
        drawdowns = np.nan_to_num(drawdowns, nan=0.0, posinf=0.0, neginf=0.0)
        max_dd_per_path = np.max(drawdowns, axis=1)
        return float(np.nanmean(max_dd_per_path))
    except Exception:
        return None


def compute_annualized_volatility(sim_arr: np.ndarray, trading_days: int = 252) -> float | None:
    """
    Compute annualized volatility from simulated daily log returns.
    Returns decimal fraction (e.g. 0.27 for 27%).
    """
    try:
        arr = np.asarray(sim_arr, dtype=float)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            return None
        logr = np.diff(np.log(arr), axis=1)
        if logr.size == 0:
            return None
        per_path_vol = np.std(logr, axis=1, ddof=1) * np.sqrt(trading_days)
        return float(np.nanmean(per_path_vol))
    except Exception:
        return None


import market_data
import monte_carlo
from agents import value_agent, growth_agent, risk_agent, quant_agent, chairperson_agent  # type: ignore

st.set_page_config(page_title="AI Investment Committee", layout="wide", initial_sidebar_state="collapsed")

# --- Custom dark theme via CSS ---
st.markdown(
    """
    <style>
    :root {
        --bg: #0b0f14;
        --card: #0f1720;
        --muted: #9aa7b2;
        --accent: #7dd3fc;
        --buy: #16a34a;
        --sell: #dc2626;
        --hold: #f59e0b;
    }
    html, body, .streamlit-container {
        background-color: var(--bg);
        color: #e6eef8;
    }
    .stApp, .reportview-container {
        background-color: var(--bg);
    }
    .block-container {
        padding-top: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    .card {
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
        border: 1px solid rgba(255,255,255,0.03);
        padding: 14px;
        border-radius: 8px;
    }
    .agent-badge {
        display:inline-block;
        padding:8px 12px;
        border-radius:999px;
        color: white;
        font-weight:600;
        font-size:14px;
    }
    .badge-buy { background: var(--buy); }
    .badge-hold { background: var(--hold); color: #111; }
    .badge-sell { background: var(--sell); }
    .final-box {
        padding: 18px;
        border-radius: 10px;
        color: #061014;
        font-weight:700;
        font-size:20px;
    }
    .small-muted { color: var(--muted); font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 style='margin:0 0 6px 0'>AI Investment Committee</h1>", unsafe_allow_html=True)
st.markdown("<div class='small-muted'>Multi-agent stock analysis (Value, Growth, Risk, Quant)</div>", unsafe_allow_html=True)
st.write("")

col_input, col_blank = st.columns([3, 1])
with col_input:
    mode = st.selectbox("Mode", ["Single ticker", "Compare tickers"])
    ticker = st.text_input("Ticker A (e.g. AAPL)", value="AAPL", max_chars=10).strip().upper()
    second_ticker = ""
    if mode == "Compare tickers":
        second_ticker = st.text_input("Ticker B (e.g. MSFT)", value="MSFT", max_chars=10).strip().upper()
    sims = st.number_input("Monte Carlo simulations", min_value=100, max_value=20000, value=1000, step=100)
    analyze = st.button("Analyze", type="primary")

if not ticker:
    st.info("Please enter a ticker to analyze.")
    st.stop()

# Run analysis when button clicked
if analyze:
    with st.spinner("Running committee — fetching market data and running agents..."):
        md = market_data.get_market_data(ticker)
        if md.get("error"):
            st.error(f"Market data error: {md.get('error')}")
            st.stop()

        mc = monte_carlo.run_monte_carlo(ticker, num_sims=int(sims))
        if mc.get("error"):
            st.error(f"Monte Carlo error: {mc.get('error')}")
            st.stop()

        # Run agents defensively
        agent_outputs = []
        try:
            agent_outputs.append(value_agent.analyze(ticker, md))
        except Exception as e:
            agent_outputs.append({"agent": "value_agent", "ticker": ticker, "result": {"recommendation": "HOLD", "confidence": 40, "reason": f"error: {e}"}})

        try:
            agent_outputs.append(growth_agent.analyze(ticker, md))
        except Exception as e:
            agent_outputs.append({"agent": "growth_agent", "ticker": ticker, "result": {"recommendation": "HOLD", "confidence": 45, "reason": f"error: {e}"}})

        try:
            agent_outputs.append(risk_agent.analyze(ticker, md))
        except Exception as e:
            agent_outputs.append({"agent": "risk_agent", "ticker": ticker, "result": {"recommendation": "HOLD", "confidence": 60, "reason": f"error: {e}"}})

        try:
            agent_outputs.append(quant_agent.analyze(ticker, md, mc))
        except Exception as e:
            agent_outputs.append({"agent": "quant_agent", "ticker": ticker, "result": {"recommendation": "HOLD", "confidence": 50, "reason": f"error: {e}"}})

        final = chairperson_agent.aggregate(agent_outputs)

        # --- If compare mode selected, run full pipeline for second ticker ---
        final_b = None
        md_b = mc_b = None
        agent_outputs_b = []
        rr_b = {}
        alloc_b = {}
        sim_arr_b = None
        if mode == "Compare tickers" and second_ticker:
            md_b = market_data.get_market_data(second_ticker)
            if md_b.get("error"):
                st.error(f"Market data error for {second_ticker}: {md_b.get('error')}")
                md_b = None
            else:
                mc_b = monte_carlo.run_monte_carlo(second_ticker, num_sims=int(sims))
                if mc_b.get("error"):
                    st.error(f"Monte Carlo error for {second_ticker}: {mc_b.get('error')}")
                    mc_b = None

            if md_b:
                # run agents defensively for ticker B
                try:
                    agent_outputs_b.append(value_agent.analyze(second_ticker, md_b))
                except Exception as e:
                    agent_outputs_b.append({"agent": "value_agent", "ticker": second_ticker, "result": {"recommendation": "HOLD", "confidence": 40, "reason": f"error: {e}"}})
                try:
                    agent_outputs_b.append(growth_agent.analyze(second_ticker, md_b))
                except Exception as e:
                    agent_outputs_b.append({"agent": "growth_agent", "ticker": second_ticker, "result": {"recommendation": "HOLD", "confidence": 45, "reason": f"error: {e}"}})
                try:
                    agent_outputs_b.append(risk_agent.analyze(second_ticker, md_b))
                except Exception as e:
                    agent_outputs_b.append({"agent": "risk_agent", "ticker": second_ticker, "result": {"recommendation": "HOLD", "confidence": 60, "reason": f"error: {e}"}})
                try:
                    agent_outputs_b.append(quant_agent.analyze(second_ticker, md_b, mc_b))
                except Exception as e:
                    agent_outputs_b.append({"agent": "quant_agent", "ticker": second_ticker, "result": {"recommendation": "HOLD", "confidence": 50, "reason": f"error: {e}"}})

                final_b = chairperson_agent.aggregate(agent_outputs_b)

                # defensive defaults & populate risk/allocation for B (reuse same heuristics)
                rr_b = final_b.get("risk_report", {}) or {}
                alloc_b = final_b.get("suggested_portfolio_allocation", {}) or {}
                sim_arr_b = _select_sim_paths(mc_b) if mc_b else None
                if sim_arr_b is not None:
                    try:
                        mc_dd_b = compute_expected_drawdown(sim_arr_b)
                        mc_vol_b = compute_annualized_volatility(sim_arr_b)
                        if mc_dd_b is not None:
                            rr_b.setdefault("expected_drawdown_pct", mc_dd_b)
                        if mc_vol_b is not None:
                            rr_b.setdefault("volatility", mc_vol_b)
                    except Exception:
                        pass

                rr_b.setdefault("volatility", md_b.get("annualized_volatility") or (mc_b.get("mc_params") or {}).get("annualized_volatility") if md_b else None)
                rr_b.setdefault("expected_drawdown_pct", (md_b.get("max_drawdown") if md_b else None) or (mc_b.get("max_drawdown") if mc_b else None))
                prob_loss_b = (mc_b or {}).get("probability_loss_more_than_20")
                if not rr_b.get("tail_risk_comment"):
                    if prob_loss_b is None:
                        rr_b["tail_risk_comment"] = ""
                    elif prob_loss_b > 0.30:
                        rr_b["tail_risk_comment"] = "Elevated tail risk (high probability of >20% loss)"
                    elif prob_loss_b > 0.10:
                        rr_b["tail_risk_comment"] = "Moderate tail risk"
                    else:
                        rr_b["tail_risk_comment"] = "Tail risk appears limited"

                if alloc_b.get("equity_pct") is None:
                    rec_b = (final_b.get("final_recommendation") or "HOLD").upper()
                    conf_b = int(final_b.get("final_confidence", 50) or 50)
                    if rec_b == "BUY":
                        equity_b = min(90, 30 + int(conf_b * 0.6))
                    elif rec_b == "SELL":
                        equity_b = max(10, 30 - int(conf_b * 0.6))
                    else:
                        equity_b = 50
                    alloc_b["equity_pct"] = equity_b
                    alloc_b["cash_pct"] = 100 - equity_b
                    alloc_b.setdefault("target_weight_pct", {})

                final_b["risk_report"] = rr_b
                final_b["suggested_portfolio_allocation"] = alloc_b

        # --- Populate defensive risk_report & suggested_portfolio_allocation if empty ---
        rr = final.get("risk_report", {}) or {}
        alloc = final.get("suggested_portfolio_allocation", {}) or {}

        # try to compute nicer risk metrics from Monte Carlo simulated paths
        sim_arr = _select_sim_paths(mc)
        if sim_arr is not None:
            try:
                mc_dd = compute_expected_drawdown(sim_arr)
                mc_vol = compute_annualized_volatility(sim_arr)
                # prefer Monte Carlo derived values if not provided by chair/market data
                if mc_dd is not None:
                    # keep decimal fraction (0.33 -> 33%)
                    rr.setdefault("expected_drawdown_pct", mc_dd)
                if mc_vol is not None:
                    rr.setdefault("volatility", mc_vol)
            except Exception:
                pass

        # basic risk fields from market data / monte carlo
        rr.setdefault("volatility", md.get("annualized_volatility") or (mc.get("mc_params") or {}).get("annualized_volatility"))
        rr.setdefault("expected_drawdown_pct", md.get("max_drawdown") or mc.get("max_drawdown"))
        prob_loss = mc.get("probability_loss_more_than_20")
        if not rr.get("tail_risk_comment"):
            if prob_loss is None:
                rr["tail_risk_comment"] = ""
            elif prob_loss > 0.30:
                rr["tail_risk_comment"] = "Elevated tail risk (high probability of >20% loss)"
            elif prob_loss > 0.10:
                rr["tail_risk_comment"] = "Moderate tail risk"
            else:
                rr["tail_risk_comment"] = "Tail risk appears limited"

        # simple suggested allocation heuristic (adjust to your strategy)
        if alloc.get("equity_pct") is None:
            rec = (final.get("final_recommendation") or "HOLD").upper()
            conf = int(final.get("final_confidence", 50) or 50)
            if rec == "BUY":
                equity = min(90, 30 + int(conf * 0.6))
            elif rec == "SELL":
                equity = max(10, 30 - int(conf * 0.6))
            else:
                equity = 50
            alloc["equity_pct"] = equity
            alloc["cash_pct"] = 100 - equity
            alloc.setdefault("target_weight_pct", {})

        final["risk_report"] = rr
        final["suggested_portfolio_allocation"] = alloc

        # debug: show chair payload when summary missing (enable via DEV_SHOW_CHAIR=1)
        if os.getenv("DEV_SHOW_CHAIR") or not (
            final.get("summary")
            or final.get("votes")
            or final.get("agreements")
            or final.get("bullish")
            or final.get("bearish")
        ):
            with st.expander("Chair output (debug)", expanded=False):
                st.json(final)

    # --- Top summary metrics ---
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    top_cols = st.columns([2, 1, 1, 1, 1, 1])

    price_last = md.get("price_last")
    pe = md.get("current_pe")
    pb = md.get("current_pb")
    de = md.get("debt_to_equity")
    vol = md.get("annualized_volatility")
    maxdd = md.get("max_drawdown")

    # Price & simple KPI metrics
    with top_cols[0]:
        st.markdown("<div style='font-size:13px;color:var(--muted)'>Current Price</div>", unsafe_allow_html=True)
        price_display = f"{price_last:.2f}" if isinstance(price_last, (int, float)) else (str(price_last) if price_last is not None else "N/A")
        st.markdown(f"<h2 style='margin:6px 0'>{price_display}</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>History: {md.get('history_start')} → {md.get('history_end')}</div>", unsafe_allow_html=True)

    with top_cols[1]:
        st.metric(label="P/E", value=(f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A"))

    with top_cols[2]:
        st.metric(label="P/B", value=(f"{pb:.2f}" if isinstance(pb, (int, float)) else "N/A"))

    with top_cols[3]:
        st.metric(label="Debt/Equity", value=(f"{de:.2f}" if isinstance(de, (int, float)) else "N/A"))

    with top_cols[4]:
        st.metric(label="Ann. Vol", value=(f"{vol:.2%}" if isinstance(vol, (int, float)) else "N/A"))

    # compute Sharpe ratio defensively
    ann_return_raw = None
    for k in ("annualized_return", "annualized_return_pct", "annual_return", "cagr"):
        if k in md and md.get(k) is not None:
            ann_return_raw = md.get(k)
            break

    ann_return = None
    if ann_return_raw is not None:
        try:
            ar = float(ann_return_raw)
            # accept either decimal (0.08) or percent (8.0) -> normalize to decimal
            if abs(ar) > 1:
                ar = ar / 100.0
            ann_return = ar
        except Exception:
            ann_return = None

    # fallback: try Monte Carlo params if still missing
    if ann_return is None:
        ann_return = (mc.get("mc_params") or {}).get("annualized_return")
        try:
            if ann_return is not None:
                ar = float(ann_return)
                if abs(ar) > 1:
                    ar = ar / 100.0
                ann_return = ar
        except Exception:
            ann_return = None

    # final vol fallback from computed rr if needed
    if not isinstance(vol, (int, float)):
        vol = final.get("risk_report", {}).get("volatility") or vol

    sharpe = None
    try:
        if isinstance(ann_return, (int, float)) and isinstance(vol, (int, float)) and vol != 0:
            Rf = 0.045
            sharpe = (float(ann_return) - Rf) / float(vol)
    except Exception:
        sharpe = None

    with top_cols[5]:
        st.metric(label="Sharpe", value=(f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else "N/A"))

    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    # --- Agent vote cards ---
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-bottom:6px'>Agent Votes</h3>", unsafe_allow_html=True)
    votes_cols = st.columns(4)
    badge_map = {"BUY": "badge-buy", "HOLD": "badge-hold", "SELL": "badge-sell"}

    for i, ag in enumerate(agent_outputs):
        agent_name = ag.get("agent", f"agent_{i}")
        res = ag.get("result", {})
        rec = (res.get("recommendation") or "HOLD").upper()
        conf = res.get("confidence", 50)
        # simple confidence interval for display (±10% clamp)
        try:
            conf_int = int(max(3, min(20, int(conf * 0.1))))
        except Exception:
            conf_int = 10
        reason = res.get("reason", "")
        badge_cls = badge_map.get(rec, "badge-hold")
        with votes_cols[i]:
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:10px'>"
                f"<div style='flex:1'><strong>{agent_name.replace('_',' ').title()}</strong></div>"
                f"<div class='agent-badge {badge_cls}'>{rec} • {int(conf)}% (±{conf_int}%)</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Reason (click to expand)"):
                st.write(reason or "No reason provided.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.write("")

    # --- Monte Carlo chart and probabilities ---
    left, right = st.columns([2, 1])
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-bottom:6px'>Monte Carlo Simulations (1 year)</h3>", unsafe_allow_html=True)

        plot_path = mc.get("plot_path")
        if plot_path and os.path.exists(plot_path):
            # updated Streamlit API: width='stretch' replaces use_container_width=True
            st.image(plot_path, width='stretch')

            # show what parameters drove the simulation (drift & annualized vol)
            mc_params = mc.get("mc_params") or {}
            method = mc_params.get("method")
            drift = mc_params.get("drift")
            ann_vol = mc_params.get("annualized_volatility")
            days_used = mc_params.get("days_used", mc.get("days", 252))
            if method or (drift is not None and ann_vol is not None):
                drift_txt = (f"{drift:.2%}" if isinstance(drift, (int, float)) else (str(drift) if drift is not None else "N/A"))
                vol_txt = (f"{ann_vol:.2%}" if isinstance(ann_vol, (int, float)) else (str(ann_vol) if ann_vol is not None else "N/A"))
                method_txt = f" using {method}" if method else ""
                st.markdown(
                    f"<div class='small-muted' style='margin-top:8px'>Using {days_used}-day historical log returns{method_txt} for drift ({drift_txt}) and annualized volatility ({vol_txt}).</div>",
                    unsafe_allow_html=True,
                )
            # Histogram of simulated 1-year returns (percent)
            sim_arr = _select_sim_paths(mc)
            if sim_arr is not None:
                try:
                    sim = np.asarray(sim_arr, dtype=float)
                    if sim.ndim == 1:
                        # single path -> treat as terminal/initial
                        returns_pct = np.array([ (sim[-1] / sim[0] - 1) * 100 ])
                    else:
                        init = sim[:, 0]
                        term = sim[:, -1]
                        with np.errstate(divide='ignore', invalid='ignore'):
                            returns_pct = (term / init - 1) * 100
                        returns_pct = returns_pct[np.isfinite(returns_pct)]

                    if returns_pct.size > 0:
                        p5, p50, p95 = np.percentile(returns_pct, [5, 50, 95])
                        fig, ax = plt.subplots(figsize=(6, 3))
                        n, bins, patches = ax.hist(returns_pct, bins=50, edgecolor='black')
                        # color bars by sign of bin center
                        for patch in patches:
                            center = patch.get_x() + patch.get_width() / 2.0
                            patch.set_facecolor("#16a34a" if center >= 0 else "#dc2626")
                        ax.axvline(p5, color="gray", linestyle="--", linewidth=1, label=f"5th: {p5:.2f}%")
                        ax.axvline(p50, color="blue", linestyle="--", linewidth=1, label=f"50th: {p50:.2f}%")
                        ax.axvline(p95, color="gray", linestyle="--", linewidth=1, label=f"95th: {p95:.2f}%")
                        ax.set_xlabel("1-Year Return (%)")
                        ax.set_title("Distribution of 1-Year Returns (simulated)")
                        ax.grid(alpha=0.2)
                        ax.legend(loc="upper right", fontsize="small")
                        st.pyplot(fig)
                        st.write(f"5th: {p5:.2f}% — 50th: {p50:.2f}% — 95th: {p95:.2f}%")
                except Exception:
                    pass
        else:
            st.info("Monte Carlo plot not available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='margin-bottom:6px'>Probabilities & Percentiles</h4>", unsafe_allow_html=True)
        prob_gain = mc.get("probability_gain_more_than_10")
        prob_loss = mc.get("probability_loss_more_than_20")
        p5 = mc.get("percentile_5")
        p50 = mc.get("percentile_50")
        p95 = mc.get("percentile_95")
        st.metric("P(> +10%)", f"{prob_gain:.2%}" if prob_gain is not None else "N/A")
        st.metric("P(< -20%)", f"{prob_loss:.2%}" if prob_loss is not None else "N/A")
        st.write("")
        st.write(f"5th percentile: {p5:.2f}" if p5 is not None else "5th percentile: N/A")
        st.write(f"50th percentile: {p50:.2f}" if p50 is not None else "50th percentile: N/A")
        st.write(f"95th percentile: {p95:.2f}" if p95 is not None else "95th percentile: N/A")
        st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # --- Chairperson final recommendation ---
    final_rec = final.get("final_recommendation", "HOLD")
    final_conf = final.get("final_confidence", 50)

    # Prefer explicit summary, then structured votes, then other LLM fields
    summary = (final.get("summary") or "").strip()
    votes = final.get("votes") or {}
    agreements = final.get("agreements") or []
    disagreements = final.get("disagreements") or []
    bullish = (final.get("bullish") or "").strip()
    bearish = (final.get("bearish") or "").strip()

    def _has_structured_votes(v):
        if not isinstance(v, dict):
            return False
        if any(k in v for k in ("BUY", "HOLD", "SELL")):
            return True
        return any(isinstance(val, list) for val in v.values())

    if summary:
        # make sure the summary wraps and is readable
        summary_html = f"<div style='color:var(--muted);white-space:normal;word-break:break-word'>{summary}</div>"
    elif _has_structured_votes(votes):
        lines = []
        for label in ["BUY", "HOLD", "SELL"]:
            for v in votes.get(label, []):
                agent = v.get("agent", "unknown")
                conf = int(v.get("confidence", 0) or 0)
                reason = (v.get("reason", "") or "").replace("\n", " ").strip()
                reason = reason.replace("<", "&lt;").replace(">", "&gt;")
                lines.append(f"<li><strong>{agent}</strong>: {label} • {conf}% — {reason}</li>")
        summary_html = "<ul style='margin:6px 0 0 0;padding-left:18px;color:var(--muted);'>" + "".join(lines) + "</ul>"
    else:
        parts = []
        if agreements:
            parts.append("<strong>Agreements:</strong> " + ", ".join(agreements))
        if disagreements:
            parts.append("<strong>Disagreements:</strong> " + ", ".join(disagreements))
        if bullish:
            parts.append("<strong>Bullish:</strong> " + bullish)
        if bearish:
            parts.append("<strong>Bearish:</strong> " + bearish)
        if parts:
            summary_html = "<div style='color:var(--muted);white-space:normal;word-break:break-word'>" + "<br>".join(parts) + "</div>"
        else:
            summary_html = "<div style='color:var(--muted)'>No summary available.</div>"

    color = {"BUY": "#16a34a", "HOLD": "#f59e0b", "SELL": "#ef4444"}.get(final_rec, "#f59e0b")

    st.markdown(
        f"<div class='card' style='padding:16px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center;gap:18px'>"
        f"<div style='flex:1'>"
        f"<div style='font-size:13px;color:var(--muted)'>Committee Recommendation</div>"
        f"<div class='final-box' style='background:{color};margin-top:8px'>{final_rec} • {int(final_conf)}%</div>"
        f"</div>"
        f"<div style='width:55%'>"
        f"<div style='margin-left:18px;color:#cfe8ff'><strong>Summary</strong></div>"
        f"<div style='margin-left:18px;margin-top:6px'>{summary_html}</div>"
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Render risk report and suggested allocation in a polished card
    rr = final.get("risk_report", {}) or {}
    alloc = final.get("suggested_portfolio_allocation", {}) or {}
    with st.container():
        st.markdown("<div class='card' style='margin-top:8px;padding:12px'>", unsafe_allow_html=True)
        st.markdown("<div style='display:flex;gap:20px;align-items:flex-start'>", unsafe_allow_html=True)
        # left: risk metrics
        st.markdown("<div style='flex:1'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px;color:var(--muted)'>Risk Report</div>", unsafe_allow_html=True)
        exp_dd = rr.get("expected_drawdown_pct")
        vol = rr.get("volatility")
        tail = rr.get("tail_risk_comment", "")
        # format numeric decimals as percentages for display
        if isinstance(exp_dd, (int, float)):
            try:
                st.write(f"- Expected drawdown: {exp_dd:.2%}")
            except Exception:
                st.write(f"- Expected drawdown: {exp_dd}")
        else:
            st.write(f"- Expected drawdown: {exp_dd if exp_dd is not None else 'N/A'}")

        if isinstance(vol, (int, float)):
            st.write(f"- Volatility (ann.): {vol:.2%}")
        else:
            st.write(f"- Volatility (ann.): {vol if vol is not None else 'N/A'}")
        st.write(f"- Tail risk: {tail or 'N/A'}")
        st.markdown("</div>", unsafe_allow_html=True)
        # right: suggested allocation
        st.markdown("<div style='width:320px'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:13px;color:var(--muted)'>Suggested Allocation</div>", unsafe_allow_html=True)
        eq = alloc.get("equity_pct")
        cash = alloc.get("cash_pct")
        targets = alloc.get("target_weight_pct", {})
        if eq is not None:
            st.metric("Equity %", f"{eq}%")
        else:
            st.write("- Equity %: N/A")
        if cash is not None:
            st.write(f"- Cash %: {cash}%")
        if targets:
            st.write("- Targets:")
            for k, v in targets.items():
                st.write(f"  - {k}: {v}%")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Export PDF report (simple): uses fpdf if installed
    try:
        from fpdf import FPDF

        def _make_pdf_bytes(final, md, mc, agent_outputs):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            final_rec = final.get("final_recommendation", "HOLD")
            final_conf = final.get("final_confidence", 50)
            pdf.cell(0, 8, f"AI Investment Committee — Recommendation: {final_rec} ({final_conf}%)", ln=True)
            pdf.ln(4)
            pdf.multi_cell(0, 6, f"Summary: {final.get('summary','')}")
            pdf.ln(4)
            pdf.cell(0, 6, "Agent votes:", ln=True)
            for a in agent_outputs:
                name = a.get("agent", "agent")
                res = a.get("result", {})
                rec = (res.get("recommendation") or "HOLD").upper()
                conf = res.get("confidence", 50)
                pdf.multi_cell(0, 6, f"- {name}: {rec} ({conf}%) — {res.get('reason','')}")
            pdf.ln(4)
            rr = final.get("risk_report", {}) or {}
            pdf.cell(0, 6, f"Risk report: drawdown={rr.get('expected_drawdown_pct')} vol={rr.get('volatility')}", ln=True)
            return pdf.output(dest="S").encode("latin-1")

        pdf_bytes = None
        if st.button("Export PDF report"):
            pdf_bytes = _make_pdf_bytes(final, md, mc, agent_outputs)
            st.download_button("Download PDF", data=pdf_bytes, file_name=f"{ticker}_report.pdf", mime="application/pdf")
    except Exception:
        # fpdf not installed or error — show a small hint
        st.markdown("<div style='margin-top:8px;color:var(--muted)'>Install `fpdf` to enable PDF export: python3 -m pip install fpdf</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='small-muted' style='margin-top:12px'>Enter a ticker and click Analyze to run the AI Investment Committee.</div>", unsafe_allow_html=True)

    # If comparing, render side-by-side metrics, votes, and overlaid MC paths + preference
    if mode == "Compare tickers" and second_ticker and final_b:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h3 style='margin-top:10px'>Comparison: {} vs {}</h3>".format(ticker, second_ticker), unsafe_allow_html=True)
        # Side-by-side metrics
        mcols = st.columns(2)
        def _render_metrics(md_obj, mc_obj, final_obj, col, label):
            with col:
                st.markdown(f"<div class='card'><strong>{label}</strong></div>", unsafe_allow_html=True)
                price = md_obj.get("price_last")
                pe = md_obj.get("current_pe")
                pb = md_obj.get("current_pb")
                de = md_obj.get("debt_to_equity")
                volx = md_obj.get("annualized_volatility") or final_obj.get("risk_report", {}).get("volatility")
                st.write(f"Price: {price if price is not None else 'N/A'}")
                st.write(f"P/E: {(f'{pe:.2f}' if isinstance(pe,(int,float)) else 'N/A')}")
                st.write(f"P/B: {(f'{pb:.2f}' if isinstance(pb,(int,float)) else 'N/A')}")
                st.write(f"Debt/Equity: {(f'{de:.2f}' if isinstance(de,(int,float)) else 'N/A')}")
                st.write(f"Ann. Vol: {(f'{volx:.2%}' if isinstance(volx,(int,float)) else 'N/A')}")
        _render_metrics(md, mc, final, mcols[0], ticker)
        _render_metrics(md_b, mc_b, final_b, mcols[1], second_ticker)
        # Side-by-side agent votes
        vcols = st.columns(2)
        def _render_votes(outputs, col):
            with col:
                st.markdown("<div class='card'><h4 style='margin:0'>Agent Votes</h4></div>", unsafe_allow_html=True)
                for a in outputs:
                    name = a.get("agent","unknown").replace("_"," ").title()
                    res = a.get("result",{})
                    rec = (res.get("recommendation") or "HOLD").upper()
                    conf = res.get("confidence",50)
                    reason = res.get("reason","")
                    cls = badge_map.get(rec,"badge-hold")
                    st.markdown(f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                                f"<div><strong>{name}</strong></div>"
                                f"<div class='agent-badge {cls}'>{rec} • {int(conf)}%</div>"
                                f"</div>", unsafe_allow_html=True)
                    with st.expander("Reason (click to expand)"):
                        st.write(reason or "No reason provided.")
        _render_votes(agent_outputs, vcols[0])
        _render_votes(agent_outputs_b, vcols[1])
        # Overlaid Monte Carlo paths (sampled)
        sim_a = _select_sim_paths(mc)
        sim_b = _select_sim_paths(mc_b) if mc_b else None
        if sim_a is not None and sim_b is not None:
            try:
                simA = np.asarray(sim_a, dtype=float)
                simB = np.asarray(sim_b, dtype=float)
                fig, ax = plt.subplots(figsize=(10,4))
                # plot sample of paths (up to 100) for clarity
                nplot = min(100, simA.shape[0])
                for i in range(min(nplot, simA.shape[0])):
                    ax.plot(simA[i,:], color="#3b82f6", alpha=0.12)
                for i in range(min(nplot, simB.shape[0])):
                    ax.plot(simB[i,:], color="#fb923c", alpha=0.12)
                # median lines
                medA = np.median(simA, axis=0)
                medB = np.median(simB, axis=0)
                ax.plot(medA, color="#1e40af", linewidth=2.0, label=ticker + " median")
                ax.plot(medB, color="#c2410c", linewidth=2.0, label=second_ticker + " median")
                ax.set_title(f"Monte Carlo paths: {ticker} (blue) vs {second_ticker} (orange)")
                ax.set_xlabel("Trading days")
                ax.set_ylabel("Price")
                ax.legend()
                st.pyplot(fig)
            except Exception:
                pass
        # Simple preference heuristic & summary
        def _score_final(f):
            if not f:
                return 0
            m = {"BUY": 1, "HOLD": 0, "SELL": -1}
            r = (f.get("final_recommendation") or "HOLD").upper()
            c = int(f.get("final_confidence", 50) or 50)
            return m.get(r,0) * c
        sa = _score_final(final)
        sb = _score_final(final_b)
        if sa > sb:
            pref = ticker
            why = final.get("summary","")
        elif sb > sa:
            pref = second_ticker
            why = final_b.get("summary","")
        else:
            pref = "No clear preference (tie)"
            why = f"{ticker}: {final.get('summary','')} || {second_ticker}: {final_b.get('summary','')}"
        st.markdown(
            f"""<div class='card' style='padding:12px;margin-top:12px'>
            <strong>Committee Preference:</strong> {pref}
            <div style='color:var(--muted);margin-top:8px'>{why}</div>
            </div>""",
            unsafe_allow_html=True,
        )