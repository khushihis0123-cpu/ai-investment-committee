import math
from market_data import get_market_data

def test_get_market_data_smoke():
    res = get_market_data("AAPL")
    assert isinstance(res, dict)
    assert "ticker" in res
    # price_last should be present and finite (or test will reveal data issues)
    price = res.get("price_last")
    assert price is not None
    assert isinstance(price, (int, float))
    assert not math.isnan(float(price))