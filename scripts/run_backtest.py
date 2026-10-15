import sys; sys.path.insert(0, ".")
from price_forecast import loader, backtest
prices = loader.load_prices("data")
print(prices.groupby("item")["date"].agg(["min", "max", "count"]))
res = backtest.run(prices)
res.to_csv("data/backtest_results.csv", index=False)
print(backtest.report(res))
