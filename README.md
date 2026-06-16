# AI Investment Committee

AI Investment Committee is a production-oriented multi-agent stock analysis system that simulates a small investment committee. Five AI agents — Value, Growth, Risk, Quant, and a Chairperson aggregator — each reason from a different investing philosophy. They debate a given ticker using market data and Monte Carlo simulations, vote BUY / HOLD / SELL with confidence scores, and produce a short, explainable committee recommendation.

Architecture
------------
Simple pipeline showing data flow and decision aggregation:

ticker (AAPL)
   |
   v
+-----------------------------+
|  Market Data & Monte Carlo  |
|  (yfinance price + fundamentals,
|   Monte Carlo 1-year sims)  |
+-----------------------------+
   |
   v
+-------------------------------+   +-------------------------------+   +-------------------------------+   +-------------------------------+
| Value Agent (Buffett-like)    |   | Growth Agent                  |   | Risk Agent                    |   | Quant Agent                   |
| - P/E, P/B, D/E, FCF focus    |   | - Revenue/earnings growth     |   | - Volatility, debt, trends    |   | - Interprets MC & metrics     |
+-------------------------------+   +-------------------------------+   +-------------------------------+   +-------------------------------+
   \___________   ___________/__________________________   _______________________   ____________________/
               \ /                                \     / /
                v                                  v   v v
               +------------------------------------------------+
               | Chairperson Agent (aggregates votes & reasons) |
               +------------------------------------------------+
                                |
                                v
                   Final recommendation (BUY / HOLD / SELL + confidence)

Key Features
------------
- Multi‑agent debate system with four specialized analyst agents + a chairperson aggregator.
- Monte Carlo simulation (default 1,000 paths) projecting 252 trading days (1 year).
- Quantitative risk metrics: daily returns, annualized volatility, Sharpe ratio, maximum drawdown.
- Live data via Yahoo Finance (yfinance) for price history and fundamentals.
- Interactive Streamlit dashboard with polished dark theme, agent badges, and simulation visualization.

Tech Stack
----------
- Python 3.9+
- yfinance (market data)
- numpy, pandas (data processing)
- matplotlib (plots)
- streamlit (interactive dashboard)
- requests (LLM API calls)
- DeepSeek LLM (model: deepseek-chat) — API integration via environment variable DEEPSEEK_API_KEY

Quickstart — Run Locally
------------------------
1. Clone the repo and change directory:
   ```
   git clone <repo-url>
   cd ai-investment-committee
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   If you don't have a requirements.txt, install the essentials:
   ```
   pip install streamlit yfinance numpy pandas matplotlib requests
   ```

3. Set environment variables (required for agent LLM calls):
   ```
   export DEEPSEEK_API_KEY="your_deepseek_api_key"
   export DEEPSEEK_BASE_URL="https://api.deepseek.com"  # optional if using default
   ```

4. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

   Or run the CLI committee for a single ticker:
   ```
   python main.py AAPL --sims 1000
   ```

Example Output
--------------
After running the system you will get structured results like:

- Individual agent votes (example):
  - value_agent: BUY (conf 72) — "Attractive P/E, strong FCF..."
  - growth_agent: HOLD (conf 45) — "Revenue growth slowing vs peers..."
  - risk_agent: SELL (conf 60) — "High debt and recent volatility..."
  - quant_agent: HOLD (conf 55) — "MC: 22% chance > +10%, 18% chance <-20%..."

- Chairperson summary (example):
  ```
  final_recommendation: HOLD
  final_confidence: 58
  summary: "2 agents lean toward caution (risk & quant) while value finds pockets of strength. Committee favors holding given near-term risks."
  ```

Notes & Configuration
---------------------
- The default Monte Carlo uses 1,000 simulations; adjust via the Streamlit UI or CLI `--sims` flag.
- LLM outputs are parsed heuristically; ensure DEEPSEEK_API_KEY is set for agent reasoning. If the API key is absent, agents fall back to conservative default votes.
- Plot images are saved locally (mono-carlo_<TICKER>.png) and displayed in the dashboard.

Disclaimer
----------
This project is an experimental research tool and demonstration of multi-agent reasoning. It is NOT financial, investment, tax, or legal advice. Always consult a licensed professional before making investment decisions.

Contributing
------------
Contributions, bug reports, and feature requests are welcome. Please open an issue or a pull request with a clear description and rationale.

License
-------
Include your chosen open-source license here (e.g., MIT).