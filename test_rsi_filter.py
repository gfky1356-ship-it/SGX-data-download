"""
Standalone test for 14-RSI comparison filter.

Logic:
  - US stocks  → compare 14-RSI vs SPY 14-RSI
  - SG stocks  → compare 14-RSI vs ^STI 14-RSI
  - If stock RSI > index RSI + 3, EXCLUDE from watchlist

This script tests a small sample so we can validate before wiring
into sp500_scanner.py.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── Wilder's RSI ─────────────────────────────────────────────
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's smoothed RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # First average: simple mean over first `period` rows
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def fetch_latest_rsi(ticker: str, period: int = 14, lookback_days: int = 100) -> float | None:
    """Download recent history and return the latest RSI value."""
    end   = datetime.today()
    start = end - timedelta(days=lookback_days)
    try:
        raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
        if raw.empty or len(raw) < period + 1:
            return None
        close = raw["Close"].squeeze()
        rsi   = calc_rsi(close, period)
        return float(rsi.iloc[-1])
    except Exception as e:
        print(f"    ERROR fetching {ticker}: {e}")
        return None


# ── Small test universe ──────────────────────────────────────
# A mix of US stocks — some that might be overbought vs SPY, some not.
TEST_US_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "TSLA", "GOOGL", "JPM", "XOM", "PFE",
]
# A few Singapore-style tickers (real SGX tickers for smoke-test)
TEST_SG_TICKERS = ["D05.SI", "O39.SI", "U11.SI"]  # DBS, OCBC, UOB

US_INDEX  = "SPY"
SG_INDEX  = "^STI"
RSI_PERIOD   = 14
RSI_THRESHOLD = 3.0   # exclude if stock RSI > index RSI + this


def run_test():
    print("=" * 60)
    print("  14-RSI COMPARISON FILTER — smoke test")
    print(f"  Threshold : stock RSI > index RSI + {RSI_THRESHOLD}")
    print("=" * 60)

    # Fetch index RSIs
    print(f"\nFetching index RSIs ...")
    spy_rsi = fetch_latest_rsi(US_INDEX, RSI_PERIOD)
    sti_rsi = fetch_latest_rsi(SG_INDEX, RSI_PERIOD)
    print(f"  SPY  RSI-{RSI_PERIOD}: {spy_rsi:.2f}" if spy_rsi else "  SPY RSI: FAILED")
    print(f"  ^STI RSI-{RSI_PERIOD}: {sti_rsi:.2f}" if sti_rsi else "  ^STI RSI: FAILED")

    # ── US stocks ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  US stocks (vs SPY RSI {spy_rsi:.2f})" if spy_rsi else "  US stocks (SPY RSI unavailable)")
    print(f"{'─'*60}")
    print(f"  {'Ticker':<8} {'RSI-14':>8}  {'vs SPY':>8}  {'Result'}")
    us_passed = []
    for t in TEST_US_TICKERS:
        rsi = fetch_latest_rsi(t, RSI_PERIOD)
        if rsi is None:
            print(f"  {t:<8} {'N/A':>8}  {'—':>8}  SKIP (no data)")
            continue
        if spy_rsi is not None and rsi > spy_rsi + RSI_THRESHOLD:
            verdict = "EXCLUDED (RSI too high)"
        else:
            verdict = "PASS"
            us_passed.append(t)
        diff = f"+{rsi - spy_rsi:.2f}" if spy_rsi else "—"
        print(f"  {t:<8} {rsi:>8.2f}  {diff:>8}  {verdict}")

    # ── SG stocks ─────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  SG stocks (vs ^STI RSI {sti_rsi:.2f})" if sti_rsi else "  SG stocks (^STI RSI unavailable)")
    print(f"{'─'*60}")
    print(f"  {'Ticker':<10} {'RSI-14':>8}  {'vs STI':>8}  {'Result'}")
    sg_passed = []
    for t in TEST_SG_TICKERS:
        rsi = fetch_latest_rsi(t, RSI_PERIOD)
        if rsi is None:
            print(f"  {t:<10} {'N/A':>8}  {'—':>8}  SKIP (no data)")
            continue
        if sti_rsi is not None and rsi > sti_rsi + RSI_THRESHOLD:
            verdict = "EXCLUDED (RSI too high)"
        else:
            verdict = "PASS"
            sg_passed.append(t)
        diff = f"+{rsi - sti_rsi:.2f}" if sti_rsi else "—"
        print(f"  {t:<10} {rsi:>8.2f}  {diff:>8}  {verdict}")

    # ── TradingView watchlist format ──────────────────────────
    print(f"\n{'─'*60}")
    print("  TradingView watchlist output (RSI-filtered)")
    print(f"{'─'*60}")

    watchlist_lines = []
    for t in us_passed:
        watchlist_lines.append(f"NASDAQ:{t}")   # simplified; real scanner uses NYSE/NASDAQ
    for t in sg_passed:
        watchlist_lines.append(f"SGX:{t.replace('.SI', '')}")

    for line in watchlist_lines:
        print(f"  {line}")

    output_path = "/tmp/test_rsi_watchlist.txt"
    with open(output_path, "w") as f:
        f.write("\n".join(watchlist_lines) + "\n")
    print(f"\n  Saved → {output_path}")
    print(f"  Passed: {len(us_passed)} US + {len(sg_passed)} SG = {len(watchlist_lines)} total")
    print("=" * 60)
    print("  TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_test()
