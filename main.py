import argparse
import json
import sys
from agents import value_agent, growth_agent, risk_agent, quant_agent, chairperson_agent  # type: ignore
import market_data
import monte_carlo


def run_committee(ticker: str, sims: int = 1000):
    ticker = ticker.upper()
    md = market_data.get_market_data(ticker)
    if md.get("error"):
        print(f"Market data error: {md.get('error')}")
        return {"error": md.get("error")}

    mc = monte_carlo.run_monte_carlo(ticker, num_sims=sims)
    if mc.get("error"):
        print(f"Monte Carlo error: {mc.get('error')}")
        return {"error": mc.get("error")}

    outputs = []
    try:
        outputs.append(value_agent.analyze(ticker, md))
    except Exception as e:
        outputs.append({"agent": "value_agent", "ticker": ticker, "result": {"recommendation": "HOLD", "confidence": 40, "reason": f"error: {e}"}})

    try:
        outputs.append(growth_agent.analyze(ticker, md))
    except Exception as e:
        outputs.append({"agent": "growth_agent", "ticker": ticker, "result": {"recommendation": "HOLD", "confidence": 45, "reason": f"error: {e}"}})

    try:
        outputs.append(risk_agent.analyze(ticker, md))
    except Exception as e:
        outputs.append({"agent": "risk_agent", "ticker": ticker, "result": {"recommendation": "HOLD", "confidence": 60, "reason": f"error: {e}"}})

    try:
        outputs.append(quant_agent.analyze(ticker, md, mc))
    except Exception as e:
        outputs.append({"agent": "quant_agent", "ticker": ticker, "result": {"recommendation": "HOLD", "confidence": 50, "reason": f"error: {e}"}})

    final = chairperson_agent.aggregate(outputs)
    # Print nicely
    print("\n--- Individual agent votes ---")
    for o in outputs:
        r = o["result"]
        print(f"{o['agent']}: {r['recommendation']} (conf {r['confidence']}) - {r.get('reason','')[:140]}")
    print("\n--- Committee Decision ---")
    print(f"Final recommendation: {final['final_recommendation']} (confidence {final['final_confidence']})")
    print(f"Summary: {final['summary']}")
    print(f"Saved Monte Carlo plot: {mc.get('plot_path')}")
    return {"market_data": md, "monte_carlo": mc, "agents": outputs, "final": final}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-agent investment committee for a ticker.")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker symbol (e.g. AAPL)")
    parser.add_argument("--sims", type=int, default=1000, help="Monte Carlo simulations (default 1000)")
    args = parser.parse_args()
    run_committee(args.ticker, sims=args.sims)