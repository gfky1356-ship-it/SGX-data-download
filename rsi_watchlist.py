# ============================================================
# RSI WATCHLIST FILTER — standalone module
#
# Reads an existing scan results CSV, downloads price data
# for confirmed tickers only, applies 14-RSI vs index filter,
# and exports a TradingView watchlist.
#
# Usage in Colab:
#   !python rsi_watchlist.py
#   # or to point at a specific CSV:
#   !python rsi_watchlist.py --csv /content/sgx_scan_results.csv
#
# Requires: yfinance, pandas, numpy
# Expects CSV columns: symbol, signal_confirmed, [market]
# ============================================================

import argparse
import os
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────
RSI_PERIOD    = 14
RSI_THRESHOLD = 3.0     # exclude if stock RSI > index RSI + this

# Default CSV paths to try (in order) if --csv not given
DEFAULT_CSV_PATHS = [
    "/content/sp500_scan_results.csv",
    "/content/us_scan_results.csv",
    "/content/sgx_scan_results.csv",
]

LOOKBACK_DAYS = 100     # enough for stable RSI-14 (~70 trading bars)


# ── RSI helper ───────────────────────────────────────────────
def calc_rsi(series, period=14):
    """Wilder's smoothed RSI."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def detect_market(symbols):
    """Guess market from ticker format. Returns 'SG', 'US', or 'MIXED'."""
    sg = sum(1 for s in symbols if str(s).endswith(".SI") or str(s).isdigit())
    return "SG" if sg > len(symbols) / 2 else ("MIXED" if sg > 0 else "US")


def download_index(ticker, label):
    """Download index close with validation. Returns (Series, label) or (None, label)."""
    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    print(f"    Downloading {label} ({ticker})...", end=" ", flush=True)
    try:
        raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
        if raw.empty or len(raw) < RSI_PERIOD + 5:
            print("FAILED (empty or too few bars)")
            return None, label
        close = raw["Close"].squeeze()
        print(f"ok  |  {len(close)} bars  |  last bar: {close.index[-1].date()}  close: {float(close.iloc[-1]):.2f}")
        return close, label
    except Exception as e:
        print(f"ERROR: {e}")
        return None, label


def download_stocks(tickers):
    """Download recent price history for a list of tickers. Returns dict of close Series."""
    end   = datetime.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    print(f"    Downloading {len(tickers)} tickers...", end=" ", flush=True)
    try:
        raw = yf.download(tickers, start=start.strftime("%Y-%m-%d"),
                          end=end.strftime("%Y-%m-%d"), auto_adjust=True, progress=False,
                          threads=True)
        result = {}
        if len(tickers) == 1:
            if not raw.empty:
                result[tickers[0]] = raw["Close"].squeeze()
        else:
            for t in tickers:
                try:
                    s = raw["Close"][t].dropna()
                    if len(s) >= RSI_PERIOD + 5:
                        result[t] = s
                except Exception:
                    pass
        print(f"ok  ({len(result)}/{len(tickers)} with data)")
        return result
    except Exception as e:
        print(f"ERROR: {e}")
        return {}


def run(csv_path, output_path=None):
    print("=" * 60)
    print("  RSI WATCHLIST FILTER")
    print(f"  RSI-{RSI_PERIOD}  |  threshold: index RSI + {RSI_THRESHOLD}")
    print("=" * 60)

    # ── Load CSV ─────────────────────────────────────────────
    print(f"\n[1] Loading scan results from: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"    ERROR: file not found — {csv_path}")
        return

    df_results = pd.read_csv(csv_path)
    confirmed  = df_results[df_results["signal_confirmed"] == True]
    if confirmed.empty:
        print("    No confirmed signals in this CSV.")
        return
    print(f"    Total rows: {len(df_results)}  |  Confirmed signals: {len(confirmed)}")

    symbols = confirmed["symbol"].tolist()
    market  = detect_market(symbols)

    # For Yahoo Finance, SGX tickers need .SI suffix if not already present
    def to_yf(sym):
        s = str(sym)
        if market == "SG" and not s.endswith(".SI"):
            return s + ".SI"
        return s

    yf_symbols = [to_yf(s) for s in symbols]

    # ── Download index(es) ────────────────────────────────────
    print(f"\n[2] Downloading RSI reference index(es)...")

    index_close, index_label = None, None

    if market in ("US", "MIXED"):
        spy_close, spy_label = download_index("SPY", "SPY")
        if spy_close is None:
            spy_close, spy_label = download_index("^GSPC", "^GSPC (fallback)")
        if market == "US":
            index_close, index_label = spy_close, spy_label

    if market in ("SG", "MIXED"):
        sti_close, sti_label = download_index("^STI", "^STI")

    if market == "MIXED":
        # Use SPY for US-format tickers, STI for .SI — handled per-stock below
        pass
    elif market == "US":
        index_close, index_label = spy_close, spy_label
    elif market == "SG":
        index_close, index_label = sti_close, sti_label

    if market != "MIXED" and index_close is None:
        print("    ERROR: index download failed. Cannot apply RSI filter.")
        return

    # ── Download confirmed stocks ─────────────────────────────
    print(f"\n[3] Downloading price data for {len(yf_symbols)} confirmed stocks...")
    stock_closes = download_stocks(yf_symbols)

    # ── Apply RSI filter ──────────────────────────────────────
    print(f"\n[4] Applying RSI-{RSI_PERIOD} filter...")

    def get_index_rsi(sym):
        """Return (index_rsi, label) for a given symbol."""
        if market == "MIXED":
            yf_sym = to_yf(sym)
            if yf_sym.endswith(".SI"):
                if sti_close is not None:
                    return float(calc_rsi(sti_close, RSI_PERIOD).iloc[-1]), "^STI"
                return None, "^STI (unavailable)"
            else:
                if spy_close is not None:
                    return float(calc_rsi(spy_close, RSI_PERIOD).iloc[-1]), "SPY"
                return None, "SPY (unavailable)"
        return float(calc_rsi(index_close, RSI_PERIOD).iloc[-1]), index_label

    passed_rows, excluded_rows = [], []
    print(f"  {'Symbol':<10} {'Stock RSI':>10}  {'Index RSI':>10}  {'Diff':>8}  Result")
    print(f"  {'─'*55}")

    for _, row in confirmed.iterrows():
        sym    = str(row["symbol"])
        yf_sym = to_yf(sym)
        close  = stock_closes.get(yf_sym)

        idx_rsi_val, idx_label = get_index_rsi(sym)

        if close is None or len(close) < RSI_PERIOD + 5:
            print(f"  {sym:<10} {'N/A':>10}  {str(idx_rsi_val or '—'):>10}  {'—':>8}  KEEP (no price data)")
            passed_rows.append(row)
            continue

        if idx_rsi_val is None:
            print(f"  {sym:<10} {'—':>10}  {'N/A':>10}  {'—':>8}  KEEP (index unavailable)")
            passed_rows.append(row)
            continue

        stock_rsi = float(calc_rsi(close, RSI_PERIOD).iloc[-1])
        diff      = stock_rsi - idx_rsi_val

        if diff > RSI_THRESHOLD:
            result = "EXCLUDED"
            excluded_rows.append({**row.to_dict(), "stock_rsi": round(stock_rsi, 2),
                                   "index_rsi": round(idx_rsi_val, 2), "rsi_diff": round(diff, 2)})
        else:
            result = "PASS"
            passed_rows.append(row)

        print(f"  {sym:<10} {stock_rsi:>10.2f}  {idx_rsi_val:>10.2f}  {diff:>+8.2f}  {result}")

    print(f"\n  Before: {len(confirmed)}  →  After RSI filter: {len(passed_rows)}  "
          f"({len(excluded_rows)} removed)")

    # ── Export TradingView watchlist ──────────────────────────
    if output_path is None:
        base = os.path.dirname(csv_path)
        output_path = os.path.join(base, "rsi_filtered_watchlist.txt")

    lines = []
    for row in passed_rows:
        sym = str(row["symbol"] if isinstance(row, dict) else row["symbol"])
        if market == "SG" or sym.endswith(".SI"):
            lines.append(f"SGX:{sym.replace('.SI', '')}")
        else:
            lines.append(f"NYSE:{sym}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[5] Watchlist saved → {output_path}  ({len(lines)} tickers)")
    print("\n" + "=" * 60)
    print("  Done. Import into TradingView:")
    print("  Watchlist → ⋮ → Import watchlist → rsi_filtered_watchlist.txt")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSI watchlist filter on saved scan CSV")
    parser.add_argument("--csv",    default=None, help="Path to scan results CSV")
    parser.add_argument("--output", default=None, help="Output watchlist path")
    args = parser.parse_args()

    csv_path = args.csv
    if csv_path is None:
        for path in DEFAULT_CSV_PATHS:
            if os.path.exists(path):
                csv_path = path
                break
        if csv_path is None:
            print("No scan results CSV found. Run the scanner first, or pass --csv <path>")
            exit(1)

    run(csv_path, output_path=args.output)
