import requests

def get_market_data(ticker: str) -> dict:
    """
    Fetch market data for the given ticker symbol.
    Returns a dictionary containing market data.
    """
    # Example API endpoint (replace with a real one)
    api_url = f"https://api.example.com/marketdata/{ticker}"
    
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        
        # Process and return relevant market data
        return {
            "price_last": data.get("last_price"),
            "current_pe": data.get("pe_ratio"),
            "current_pb": data.get("pb_ratio"),
            "debt_to_equity": data.get("debt_equity_ratio"),
            "annualized_volatility": data.get("annual_volatility"),
            "max_drawdown": data.get("max_drawdown"),
            "history_start": data.get("history_start"),
            "history_end": data.get("history_end"),
            "annualized_return": data.get("annualized_return"),
            "annualized_return_pct": data.get("annualized_return_pct"),
        }
    except requests.RequestException as e:
        return {"error": str(e)}