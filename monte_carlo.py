import datetime
import os
from typing import Any, Dict, Optional

import math
import numpy as np
import yfinance as yf

# Use non-interactive backend so script works on headless machines
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_monte_carlo(ticker: str, num_sims: int = 1000, days: int = 252, outfile: Optional[str] = None) -> Dict[str, Any]:
    """
    Run Monte Carlo simulations for `ticker` over `days` trading days using `num_sims` paths.

    Returns a dictionary with:
      - ticker
      - simulations
      - days
      - probability_gain_more_than_10 (float)
      - probability_loss_more_than_20 (float)
      - percentile_5, percentile_50, percentile_95 (floats)
      - ending_prices (list of floats, length num_sims)
      - plot_path (path to saved plot) or error key on failure
      - mc_params (metadata about drift/volatility used)
    """
    ticker = ticker.upper()
    result: Dict[str, Any] = {"ticker": ticker, "simulations": num_sims, "days": days}
    try:
        t = yf.Ticker(ticker)
        # Fetch recent price history - use 5 years if available to estimate returns
        end = datetime.datetime.now().date()
        start = end - datetime.timedelta(days=365 * 5 + 30)
        df = t.history(start=start, end=end, interval="1d", auto_adjust=False)

        if df is None or df.empty or "Close" not in df.columns:
            return {"ticker": ticker, "error": "No price history found for ticker."}

        df = df.sort_index()
        prices = df["Close"].dropna()
        if prices.empty:
            return {"ticker": ticker, "error": "Insufficient price data to compute returns."}

        S0 = float(prices.iloc[-1])

        # Use log returns for GBM parameters
        log_returns = np.log(prices / prices.shift(1)).dropna()
        if log_returns.empty:
            return {"ticker": ticker, "error": "Insufficient return data."}

        mu = float(log_returns.mean())           # daily mean log return
        sigma = float(log_returns.std(ddof=0))   # daily volatility (log-return std)

        # If sigma is zero (constant price), handle specially
        if math.isclose(sigma, 0.0):
            # All future prices equal S0
            ending_prices = np.full(num_sims, S0, dtype=float)
            cum_log_full = None
        else:
            # Precompute constants
            dt = 1.0  # one trading day
            drift = (mu - 0.5 * sigma ** 2) * dt
            # Generate random components: shape (days, num_sims)
            rand = np.random.normal(loc=0.0, scale=1.0, size=(days, num_sims))
            increments = drift + sigma * np.sqrt(dt) * rand
            # Cumulative log returns for each simulation
            cum_log = np.cumsum(increments, axis=0)
            # Ending prices: S0 * exp(cum_log[-1])
            ending_prices = S0 * np.exp(cum_log[-1, :])
            # full cumulative with initial 0 for plotting
            cum_log_full = np.vstack([np.zeros((1, num_sims)), np.cumsum(increments, axis=0)])

        # Metrics
        gain_threshold = 1.10 * S0  # >10% gain
        loss_threshold = 0.80 * S0  # >20% loss (price drops below 80%)
        prob_gain = float(np.mean(ending_prices > gain_threshold))
        prob_loss = float(np.mean(ending_prices < loss_threshold))
        p5, p50, p95 = [float(x) for x in np.percentile(ending_prices, [5, 50, 95])]

        # Plot simulations (plot a subset if too many for readability)
        plt.figure(figsize=(10, 6))
        plot_sims = min(num_sims, 200)
        if math.isclose(sigma, 0.0):
            t_axis = np.arange(0, days + 1)
            for i in range(plot_sims):
                plt.plot(t_axis, np.full_like(t_axis, S0), color="gray", alpha=0.5, linewidth=0.8)
        else:
            t_axis = np.arange(0, days + 1)
            for i in range(plot_sims):
                path = S0 * np.exp(cum_log_full[:, i])
                plt.plot(t_axis, path, lw=0.8, alpha=0.5, color="gray")

        # Plot percentiles over time (optional, compute across simulations)
        if cum_log_full is not None:
            cum_exp = np.exp(cum_log_full) * S0  # shape (days+1, num_sims)
            p10_ts = np.percentile(cum_exp, 10, axis=1)
            p50_ts = np.percentile(cum_exp, 50, axis=1)
            p90_ts = np.percentile(cum_exp, 90, axis=1)
            plt.plot(t_axis, p50_ts, color="blue", lw=2.0, label="Median path")
            plt.plot(t_axis, p10_ts, color="red", lw=1.2, linestyle="--", label="10th pct")
            plt.plot(t_axis, p90_ts, color="green", lw=1.2, linestyle="--", label="90th pct")
            plt.legend(loc="best")

        plt.title(f"Monte Carlo Simulations for {ticker} — {num_sims} paths, {days} days")
        plt.xlabel("Trading days")
        plt.ylabel("Price")
        plt.grid(alpha=0.3)

        # Save plot into relative outputs/ (ephemeral runtime in Cloud)
        if outfile is None:
            out_dir = os.path.join(os.getcwd(), "outputs")
            os.makedirs(out_dir, exist_ok=True)
            outfile = os.path.join(out_dir, f"monte_carlo_{ticker}.png")
        plt.tight_layout()
        plt.savefig(outfile, dpi=150)
        plt.close()

        # Build result with mc_params metadata
        # Build sim_paths matrix (num_sims x days+1) for downstream risk calculations
        if cum_log_full is not None:
            sim_paths_matrix = (S0 * np.exp(cum_log_full)).T  # shape: (num_sims, days+1)
        else:
            sim_paths_matrix = np.full((num_sims, days + 1), S0, dtype=float)

        result.update(
            {
                "probability_gain_more_than_10": prob_gain,
                "probability_loss_more_than_20": prob_loss,
                "percentile_5": p5,
                "percentile_50": p50,
                "percentile_95": p95,
                "ending_prices": [float(x) for x in ending_prices.tolist()],
                "sim_paths": sim_paths_matrix.tolist(),  # full paths for risk metric computation
                "plot_path": os.path.abspath(outfile),
                "history_start": str(prices.index[0].date()),
                "history_end": str(prices.index[-1].date()),
                "price_last": S0,
                "estimated_daily_mu": mu,
                "estimated_daily_sigma": sigma,
                "mc_params": {
                    "method": "historical log returns (GBM)",
                    "days_used": days,
                    "drift": float(mu * days) if mu is not None else None,
                    "annualized_volatility": float(sigma * (days ** 0.5)) if sigma is not None else None,
                    "num_sims": num_sims,
                },
            }
        )
        return result

    except Exception as exc:
        return {"ticker": ticker, "error": f"Exception while running Monte Carlo: {exc}"}


def _print_summary(res: Dict[str, Any]) -> None:
    if "error" in res:
        print(f"Error for {res.get('ticker', '')}: {res['error']}")
        return
    print(f"Ticker: {res['ticker']}")
    print(f"Price (last): {res.get('price_last')}")
    print(f"History: {res.get('history_start')} -> {res.get('history_end')}")
    print()
    print(f"Simulations: {res.get('simulations')}, Days: {res.get('days')}")
    print()
    print("Probabilities:")
    print(f"  P(> +10%): {res.get('probability_gain_more_than_10'):.4f}")
    print(f"  P(< -20%): {res.get('probability_loss_more_than_20'):.4f}")
    print()
    print("Percentiles (ending prices):")
    print(f"  5th: {res.get('percentile_5'):.4f}")
    print(f"  50th: {res.get('percentile_50'):.4f}")
    print(f"  95th: {res.get('percentile_95'):.4f}")
    print()
    print(f"Plot saved to: {res.get('plot_path')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for a ticker.")
    parser.add_argument("ticker", nargs="?", default="AAPL", help="Ticker symbol (e.g. AAPL)")
    parser.add_argument("--sims", type=int, default=1000, help="Number of simulation paths (default 1000)")
    parser.add_argument("--days", type=int, default=252, help="Number of trading days to simulate (default 252)")
    parser.add_argument("--outfile", type=str, default=None, help="Path to save plot (default: monte_carlo_<TICKER>.png)")
    args = parser.parse_args()

    out = run_monte_carlo(args.ticker, num_sims=args.sims, days=args.days, outfile=args.outfile)
    _print_summary(out)