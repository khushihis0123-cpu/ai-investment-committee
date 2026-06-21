import argparse
import json
import sys
from agents import value_agent, growth_agent, risk_agent, quant_agent, chairperson_agent  # type: ignore
import market_data
import monte_carlo
from concurrent.futures import ThreadPoolExecutor, as_completed
import inspect
from typing import List, Dict, Any


def _call_agent(agent_mod, ticker: str, md: Dict[str, Any], mc: Dict[str, Any], context: List[Dict[str, Any]] = None):
    """Call agent.analyze with best-effort arguments (md, mc, context support)."""
    sig = inspect.signature(agent_mod.analyze)
    params = list(sig.parameters.keys())
    try:
        if len(params) == 3:
            # common: (ticker, md, mc) or (ticker, md, extra)
            return agent_mod.analyze(ticker, md, mc)
        elif len(params) == 2:
            return agent_mod.analyze(ticker, md)
        else:
            # try passing context if supported by name
            if "context" in params or "prev" in params:
                return agent_mod.analyze(ticker, md, context)
            return agent_mod.analyze(ticker, md)
    except TypeError:
        # last resort: call with only ticker
        return agent_mod.analyze(ticker)


def run_committee(ticker: str, sims: int = 1000, debate_rounds: int = 1):
    ticker = ticker.upper()
    md = market_data.get_market_data(ticker)
    if md.get("error"):
        print(f"Market data error: {md.get('error')}")
        return {"error": md.get("error")}
    mc = monte_carlo.run_monte_carlo(ticker, num_sims=sims)
    if mc.get("error"):
        print(f"Monte Carlo error: {mc.get('error')}")
        return {"error": mc.get("error")}

    agents = [value_agent, growth_agent, risk_agent, quant_agent]
    outputs: List[Dict[str, Any]] = []

    # parallel initial run
    with ThreadPoolExecutor(max_workers=len(agents)) as ex:
        futures = {ex.submit(_call_agent, a, ticker, md, mc, None): a for a in agents}
        for fut in as_completed(futures):
            try:
                outputs.append(fut.result())
            except Exception as e:
                outputs.append({
                    "agent": futures[fut].__name__,
                    "ticker": ticker,
                    "result": {"recommendation": "HOLD", "confidence": 50, "reason": f"agent error: {e}"}
                })

    # simple debate rounds: let agents see previous outputs (debate_rounds includes initial run)
    for _ in range(max(0, debate_rounds - 1)):
        with ThreadPoolExecutor(max_workers=len(agents)) as ex:
            futures = {ex.submit(_call_agent, a, ticker, md, mc, outputs): a for a in agents}
            new_outputs = []
            for fut in as_completed(futures):
                try:
                    new_outputs.append(fut.result())
                except Exception as e:
                    new_outputs.append({
                        "agent": futures[fut].__name__,
                        "ticker": ticker,
                        "result": {"recommendation": "HOLD", "confidence": 50, "reason": f"agent error: {e}"}
                    })
            outputs = new_outputs

    final = chairperson_agent.aggregate(outputs)

    # Print nicely (CLI)
    print("\n--- Individual agent votes ---")
    for o in outputs:
        r = o.get("result", {})
        print(f"{o.get('agent')}: {r.get('recommendation')} (conf {r.get('confidence')}) - {r.get('reason','')[:140]}")
    print("\n--- Committee Decision ---")
    print(f"Final recommendation: {final.get('final_recommendation')} (confidence {final.get('final_confidence')})")
    print(f"Summary: {final.get('summary')}")
    print(f"Saved Monte Carlo plot: {mc.get('plot_path')}")
    return {"market_data": md, "monte_carlo": mc, "agent_outputs": outputs, "final": final}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-agent investment committee for a ticker.")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker symbol (e.g. AAPL)")
    parser.add_argument("--sims", type=int, default=1000, help="Monte Carlo simulations (default 1000)")
    parser.add_argument("--debate-rounds", type=int, default=1, help="Number of agent debate rounds (default 1)")
    args = parser.parse_args()

    run_committee(args.ticker, sims=args.sims, debate_rounds=args.debate_rounds)