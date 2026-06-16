import os
from typing import Dict, Any

import streamlit as st

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
    ticker = st.text_input("Enter ticker (e.g. AAPL)", value="AAPL", max_chars=10).strip().upper()
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

    # --- Top summary metrics ---
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    top_cols = st.columns([2, 1, 1, 1, 1])

    price_last = md.get("price_last")
    pe = md.get("current_pe")
    pb = md.get("current_pb")
    de = md.get("debt_to_equity")
    vol = md.get("annualized_volatility")
    maxdd = md.get("max_drawdown")

    # Price & simple KPI metrics
    with top_cols[0]:
        st.markdown("<div style='font-size:13px;color:var(--muted)'>Current Price</div>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='margin:6px 0'>{price_last if price_last is not None else 'N/A'}</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='small-muted'>History: {md.get('history_start')} → {md.get('history_end')}</div>", unsafe_allow_html=True)

    with top_cols[1]:
        st.metric(label="P/E", value=(f"{pe:.2f}" if isinstance(pe, (int, float)) else "N/A"))

    with top_cols[2]:
        st.metric(label="P/B", value=(f"{pb:.2f}" if isinstance(pb, (int, float)) else "N/A"))

    with top_cols[3]:
        st.metric(label="Debt/Equity", value=(f"{de:.2f}" if isinstance(de, (int, float)) else "N/A"))

    with top_cols[4]:
        st.metric(label="Ann. Vol", value=(f"{vol:.2%}" if isinstance(vol, (int, float)) else "N/A"))

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
        reason = res.get("reason", "")
        badge_cls = badge_map.get(rec, "badge-hold")
        with votes_cols[i]:
            st.markdown(f"<div style='display:flex;align-items:center;gap:10px'>"
                        f"<div style='flex:1'><strong>{agent_name.replace('_',' ').title()}</strong></div>"
                        f"<div class='agent-badge {badge_cls}'>{rec} • {int(conf)}%</div>"
                        f"</div>", unsafe_allow_html=True)
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
            st.image(plot_path, use_column_width=True)
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
    summary = final.get("summary", "")
    color = {"BUY": "#16a34a", "HOLD": "#f59e0b", "SELL": "#ef4444"}.get(final_rec, "#f59e0b")
    st.markdown(
        f"<div class='card' style='padding:16px'>"
        f"<div style='display:flex;justify-content:space-between;align-items:center'>"
        f"<div style='flex:1'>"
        f"<div style='font-size:13px;color:var(--muted)'>Committee Recommendation</div>"
        f"<div class='final-box' style='background:{color};margin-top:8px'>{final_rec} • {int(final_conf)}%</div>"
        f"</div>"
        f"<div style='width:55%'>"
        f"<div style='margin-left:18px;color:#cfe8ff'><strong>Summary</strong></div>"
        f"<div style='margin-left:18px;color:var(--muted);margin-top:6px'>{summary}</div>"
        f"</div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("<div class='small-muted'>Results are for informational purposes only and do not constitute financial advice.</div>", unsafe_allow_html=True)
else:
    st.markdown("<div class='small-muted' style='margin-top:12px'>Enter a ticker and click Analyze to run the AI Investment Committee.</div>", unsafe_allow_html=True)