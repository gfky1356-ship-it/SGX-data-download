"""
Integration test for RSI filter step.

Simulates the scanner's internal state (stock_data dict + matched list)
using synthetic price series, so no network access is required.

The RSI filter is added as a new step between scanner output and export:
  Step 3 → matched list → RSI filter (Step 3b) → TV watchlist export
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ── Wilder's RSI (same function that will go into sp500_scanner.py) ─────────
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def apply_rsi_index_filter(matched, stock_data, spy_close,
                            rsi_period=14, threshold=3.0):
    """
    Remove entries from `matched` whose latest RSI exceeds
    spy_close latest RSI by more than `threshold` points.

    Returns (rsi_passed, rsi_excluded) lists.
    """
    # Calculate index RSI once
    spy_rsi_series = calc_rsi(spy_close, rsi_period)
    index_rsi = float(spy_rsi_series.iloc[-1])
    print(f"    SPY/Index RSI-{rsi_period}: {index_rsi:.2f}")

    rsi_passed   = []
    rsi_excluded = []
    rsi_details  = []

    for r in matched:
        sym = r["symbol"]
        df  = stock_data.get(sym)
        if df is None or len(df) < rsi_period + 5:
            # Can't compute RSI — keep in list (don't exclude on missing data)
            rsi_passed.append(r)
            rsi_details.append({**r, "stock_rsi": None, "index_rsi": index_rsi, "rsi_diff": None, "rsi_excluded": False})
            continue

        stock_rsi_series = calc_rsi(df["close"], rsi_period)
        stock_rsi = float(stock_rsi_series.iloc[-1])
        diff = stock_rsi - index_rsi
        excluded = diff > threshold

        rsi_details.append({**r, "stock_rsi": round(stock_rsi, 2),
                             "index_rsi": round(index_rsi, 2),
                             "rsi_diff": round(diff, 2),
                             "rsi_excluded": excluded})
        if excluded:
            rsi_excluded.append(r)
        else:
            rsi_passed.append(r)

    return rsi_passed, rsi_excluded, rsi_details, index_rsi


def export_tv_watchlist(stocks, exchange="NYSE"):
    """Generate TradingView watchlist lines."""
    lines = []
    for r in stocks:
        sym = r["symbol"]
        # SG tickers end with .SI → reformat for SGX
        if sym.endswith(".SI"):
            lines.append(f"SGX:{sym.replace('.SI', '')}")
        else:
            lines.append(f"{exchange}:{sym}")
    return lines


# ══ Synthetic test data ═══════════════════════════════════════════════════════
np.random.seed(99)
N = 80  # bars

def make_prices(daily_drift, n=N):
    dates = pd.date_range(end=datetime.today(), periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01 + daily_drift))
    return pd.Series(prices, index=dates)

# SPY: mild uptrend → RSI ~55-65
spy_close = make_prices(0.0005)

# Individual stocks with varied RSIs
stock_configs = {
    # overbought: strong uptrend, RSI likely > SPY + 3
    "ROCKET": make_prices(0.018),
    "MOON":   make_prices(0.015),
    # normal: similar to SPY
    "STABLE": make_prices(0.0006),
    "MILD":   make_prices(0.0003),
    # underperformer: downtrend, RSI below SPY
    "WEAK":   make_prices(-0.003),
}

# Build stock_data dict (mimics scanner's internal dict with 'close' column)
stock_data = {}
for sym, prices in stock_configs.items():
    df = pd.DataFrame({"close": prices, "open": prices * 0.999,
                        "high": prices * 1.005, "low": prices * 0.995,
                        "volume": np.full(N, 500_000)})
    stock_data[sym] = df

# Simulate `matched` — all stocks passed scanner (in real scanner these
# are the confirmed signal stocks from Step 3)
matched = [
    {"symbol": "ROCKET", "name": "Rocket Corp",  "sector": "Tech",    "last_close": float(stock_data["ROCKET"]["close"].iloc[-1])},
    {"symbol": "MOON",   "name": "Moon Inc",     "sector": "Biotech", "last_close": float(stock_data["MOON"]["close"].iloc[-1])},
    {"symbol": "STABLE", "name": "Stable Ltd",   "sector": "Finance", "last_close": float(stock_data["STABLE"]["close"].iloc[-1])},
    {"symbol": "MILD",   "name": "Mild Corp",    "sector": "Energy",  "last_close": float(stock_data["MILD"]["close"].iloc[-1])},
    {"symbol": "WEAK",   "name": "Weak Holdings","sector": "Retail",  "last_close": float(stock_data["WEAK"]["close"].iloc[-1])},
]

# ══ Run the RSI filter step ════════════════════════════════════════════════════
print("=" * 60)
print("  INTEGRATION TEST — RSI Filter Step")
print(f"  Threshold: exclude if stock RSI > index RSI + 3.0")
print("=" * 60)

print(f"\n[3b] RSI comparison filter ({len(matched)} matched stocks → filter vs SPY)...")
rsi_passed, rsi_excluded, rsi_details, index_rsi = apply_rsi_index_filter(
    matched, stock_data, spy_close, rsi_period=14, threshold=3.0
)

# Print detail table
print(f"\n    {'Symbol':<10} {'RSI':>8}  {'Diff':>8}  {'Result'}")
print(f"    {'─'*40}")
for d in rsi_details:
    rsi_str  = f"{d['stock_rsi']:.2f}" if d['stock_rsi'] is not None else "N/A"
    diff_str = f"{d['rsi_diff']:+.2f}" if d['rsi_diff'] is not None else "—"
    result   = "EXCLUDED" if d["rsi_excluded"] else "PASS"
    print(f"    {d['symbol']:<10} {rsi_str:>8}  {diff_str:>8}  {result}")

print(f"\n    Before: {len(matched)}  →  After RSI filter: {len(rsi_passed)}  ({len(rsi_excluded)} removed)")
if rsi_excluded:
    print(f"    Excluded: {', '.join(r['symbol'] for r in rsi_excluded)}")

# ══ TradingView watchlist export ══════════════════════════════════════════════
print(f"\n[4] Exporting RSI-filtered TradingView watchlist...")
tv_lines = export_tv_watchlist(rsi_passed, exchange="NYSE")
print(f"\n    --- TradingView Watchlist ---")
for line in tv_lines:
    print(f"    {line}")

output_path = "/tmp/sp500_rsi_filtered_watchlist.txt"
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(tv_lines) + "\n")
print(f"\n    Saved → {output_path}  ({len(tv_lines)} tickers)")

# ══ Assertions ════════════════════════════════════════════════════════════════
print("\n[ASSERT] Verifying expected behavior...")
spy_rsi_val = float(calc_rsi(spy_close, 14).iloc[-1])
rocket_rsi  = float(calc_rsi(stock_data["ROCKET"]["close"], 14).iloc[-1])
moon_rsi    = float(calc_rsi(stock_data["MOON"]["close"], 14).iloc[-1])
stable_rsi  = float(calc_rsi(stock_data["STABLE"]["close"], 14).iloc[-1])

assert rocket_rsi > spy_rsi_val + 3, f"ROCKET should be overbought: {rocket_rsi:.2f} vs {spy_rsi_val:.2f}"
assert moon_rsi   > spy_rsi_val + 3, f"MOON should be overbought:   {moon_rsi:.2f} vs {spy_rsi_val:.2f}"
assert "ROCKET" not in [r["symbol"] for r in rsi_passed], "ROCKET should be excluded"
assert "MOON"   not in [r["symbol"] for r in rsi_passed], "MOON should be excluded"
assert "STABLE" in [r["symbol"] for r in rsi_passed],     "STABLE should pass"
print("    All assertions PASSED")

print("\n" + "=" * 60)
print("  INTEGRATION TEST COMPLETE — logic is correct")
print("  Safe to integrate into sp500_scanner.py")
print("=" * 60)
