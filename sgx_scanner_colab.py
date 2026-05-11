# ============================================================
# SGX STOCK SCANNER — Colab Ready
#
# Sources:
#   Ticker list : api.sgx.com  (SGX official API)
#   Price data  : Yahoo Finance (yfinance, .SI suffix)
#   Benchmark   : ^STI (Straits Times Index)
#
# Scanners embedded (no file uploads needed):
#   1. SMA Gap Enlarge Buy
#   2. Breakout Buy
#   3. OBV Buy
#
# Signal confirmation logic:
#   TRUE if sma_gap_enlarge_buy  AND  obv sell_ban is ON
#   TRUE if breakout_buy         AND  obv sell_ban is ON
#   FALSE for all other combinations
#
# obv sell_ban is ON when OBV long SMA slope > 0
# (volume accumulation trend is still rising — don't sell)
#
# Outputs:
#   /content/sgx_scan_results.csv   full results for all stocks
#   /content/sgx_watchlist.txt      TradingView watchlist (confirmed only)
# ============================================================

# ── Cell 1: Install ─────────────────────────────────────────
# !pip install yfinance requests -q


# ── Cell 2: Run Scanner ──────────────────────────────────────
import csv, time, warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

OUTPUT_RESULTS   = "/content/sgx_scan_results.csv"
OUTPUT_WATCHLIST = "/content/sgx_watchlist.txt"
BATCH_SIZE       = 50
BENCHMARK        = "^STI"


# ════════════════════════════════════════════════════════════
# SCANNER 1 — SMA GAP ENLARGE BUY
# ════════════════════════════════════════════════════════════
def _sma(series, length):
    return series.rolling(length, min_periods=length).mean()


def add_sma_gap_rs_buy_sell_signals(
    df, index_close,
    b_sma20_len=20, b_sma50_len=50, b_slope_lookback=5,
    b_slope_smooth_len=1, a_rs_sma_length=30, a_delta_ma_length=9,
):
    out = df.copy()
    index_close = index_close.reindex(out.index).astype(float)

    out["b_sma20"]       = _sma(out["close"], b_sma20_len)
    out["b_sma50"]       = _sma(out["close"], b_sma50_len)
    out["b_raw_slope20"] = (out["b_sma20"] - out["b_sma20"].shift(b_slope_lookback)) / out["b_sma20"].shift(b_slope_lookback) * 100.0
    out["b_raw_slope50"] = (out["b_sma50"] - out["b_sma50"].shift(b_slope_lookback)) / out["b_sma50"].shift(b_slope_lookback) * 100.0
    out["b_slope20"]     = _sma(out["b_raw_slope20"], b_slope_smooth_len)
    out["b_slope50"]     = _sma(out["b_raw_slope50"], b_slope_smooth_len)
    out["b_bull"]        = out["b_slope20"] > out["b_slope50"]

    stock_sma        = _sma(out["close"], a_rs_sma_length)
    index_sma        = _sma(index_close, a_rs_sma_length)
    out["delta_ab"]  = (out["close"] / stock_sma) - (index_close / index_sma)
    out["delta_ma"]  = _sma(out["delta_ab"], a_delta_ma_length)
    out["slope_gap"] = out["b_slope20"] - out["b_slope50"]

    out["a_two_day_green_rising"] = (
        (out["delta_ab"] >= 0) & (out["delta_ab"].shift(1) >= 0) &
        (out["delta_ab"] > out["delta_ab"].shift(1))
    )
    out["green_candle"]       = out["close"] > out["open"]
    out["red_candle"]         = out["close"] < out["open"]
    out["slope_gap_reducing"] = (out["b_slope20"] > out["b_slope50"]) & (out["slope_gap"] < out["slope_gap"].shift(1))
    out["buy_condition"]      = out["b_bull"] & out["a_two_day_green_rising"] & out["green_candle"]
    out["sell_condition"]     = (out["delta_ab"] < out["delta_ma"]) & out["slope_gap_reducing"] & out["red_candle"]

    buy_signal  = np.zeros(len(out), dtype=bool)
    sell_signal = np.zeros(len(out), dtype=bool)
    in_position = buy_ban = sell_ban = False

    for i in range(len(out)):
        buy  = bool(out["buy_condition"].iat[i])  and not buy_ban  and not in_position
        sell = bool(out["sell_condition"].iat[i]) and not sell_ban and in_position
        if buy:
            buy_signal[i] = True;  in_position = True;  buy_ban = True;  sell_ban = False
        if sell:
            sell_signal[i] = True; in_position = False; sell_ban = True; buy_ban = False

    out["buy_signal"]  = buy_signal
    out["sell_signal"] = sell_signal
    return out


# ════════════════════════════════════════════════════════════
# SCANNER 2 — BREAKOUT BUY
# ════════════════════════════════════════════════════════════
def add_breakout_buy_signals(
    df, lookback_n1=30, lookback_n2=60, lookback_n3=90,
    sma200_slope_lookback=5, max_upside_pct=5.0,
):
    out = df.copy()
    out["sma20"]  = _sma(out["close"], 20)
    out["sma50"]  = _sma(out["close"], 50)
    out["sma200"] = _sma(out["close"], 200)
    out["sma200_slope_positive"] = out["sma200"] > out["sma200"].shift(sma200_slope_lookback)
    out["green_candle"]  = out["close"] > out["open"]
    out["sma20_above50"] = out["sma20"] > out["sma50"]
    out["b_above20sma"]  = out["close"] > out["sma20"]

    for label, lookback in [("n1", lookback_n1), ("n2", lookback_n2), ("n3", lookback_n3)]:
        signals = []
        for pos in range(len(out)):
            if pos < 1:
                signals.append(False)
                continue
            end_i = min(lookback, pos)
            a_price, a_offset = float("nan"), None
            for off in range(1, end_i + 1):
                h = out["high"].iat[pos - off]
                if pd.notna(h) and (pd.isna(a_price) or h > a_price):
                    a_price, a_offset = h, off
            pb = False
            if a_offset:
                for off in range(1, end_i + 1):
                    if off < a_offset:
                        if pd.notna(out["low"].iat[pos - off]) and pd.notna(out["sma20"].iat[pos - off]):
                            if out["low"].iat[pos - off] < out["sma20"].iat[pos - off]:
                                pb = True
            a_above  = (a_offset is not None and pd.notna(a_price) and
                        pd.notna(out["sma20"].iat[pos - a_offset]) and
                        a_price > out["sma20"].iat[pos - a_offset])
            b_cross  = pd.notna(a_price) and out["close"].iat[pos] > a_price and out["close"].iat[pos - 1] <= a_price
            b_within = pd.notna(a_price) and out["close"].iat[pos] <= a_price * (1 + max_upside_pct / 100)
            sig = (bool(out["sma200_slope_positive"].iat[pos]) and
                   bool(out["green_candle"].iat[pos])          and
                   bool(out["sma20_above50"].iat[pos])         and
                   a_above and bool(out["b_above20sma"].iat[pos]) and
                   pb and b_cross and b_within)
            signals.append(sig)
        out[f"{label}_signal"] = signals

    out["any_buy_signal"] = out["n1_signal"] | out["n2_signal"] | out["n3_signal"]
    return out


# ════════════════════════════════════════════════════════════
# SCANNER 3 — OBV (used for sell_ban state only)
# sell_ban is ON when OBV long SMA slope > 0
# (volume accumulation still rising — bullish confirmation)
# ════════════════════════════════════════════════════════════
def add_obv_signals(df, short_sma_length=20, long_sma_length=100, confirm_days=1.0):
    out = df.copy()
    close, volume = out["close"], out["volume"]
    dv = np.where(close > close.shift(1), volume,
                  np.where(close < close.shift(1), -volume, 0))
    out["obv"]           = pd.Series(dv, index=out.index).cumsum()
    out["obv_short_sma"] = _sma(out["obv"], short_sma_length)
    out["obv_long_sma"]  = _sma(out["obv"], long_sma_length)
    out["long_sma_slope"] = out["obv_long_sma"] - out["obv_long_sma"].shift(1)

    # sell_ban: True when long OBV SMA is still rising (bullish volume trend)
    out["sell_ban"] = out["long_sma_slope"] > 0
    # buy_ban:  True when long OBV SMA is falling (bearish volume trend)
    out["buy_ban"]  = out["long_sma_slope"] < 0

    out["obv_above_both"] = (out["obv"] > out["obv_short_sma"]) & (out["obv"] > out["obv_long_sma"])
    out["obv_below_both"] = (out["obv"] < out["obv_short_sma"]) & (out["obv"] < out["obv_long_sma"])

    buy_signal  = np.zeros(len(out), dtype=bool)
    sell_signal = np.zeros(len(out), dtype=bool)
    above_start = below_start = None
    buy_triggered = sell_triggered = False

    for i in range(len(out)):
        above = bool(out["obv_above_both"].iat[i]) if pd.notna(out["obv_above_both"].iat[i]) else False
        below = bool(out["obv_below_both"].iat[i]) if pd.notna(out["obv_below_both"].iat[i]) else False

        if above:
            if above_start is None: above_start = i
            below_start = None; sell_triggered = False
        else:
            above_start = None; buy_triggered = False

        if below:
            if below_start is None: below_start = i
            above_start = None; buy_triggered = False
        else:
            below_start = None; sell_triggered = False

        def elapsed(start):
            idx = out.index
            return ((idx[i] - idx[start]).total_seconds() / 86400.0
                    if isinstance(idx, pd.DatetimeIndex) else float(i - start))

        if above and above_start is not None and elapsed(above_start) > confirm_days:
            if not buy_triggered and not bool(out["buy_ban"].iat[i]):
                buy_signal[i] = True; buy_triggered = True

        if below and below_start is not None and elapsed(below_start) > confirm_days:
            if not sell_triggered and not bool(out["sell_ban"].iat[i]):
                sell_signal[i] = True; sell_triggered = True

    out["buy_signal"]  = buy_signal
    out["sell_signal"] = sell_signal
    return out


# ════════════════════════════════════════════════════════════
# STEP 1 — Fetch SGX stock list
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("  SGX STOCK SCANNER")
print("=" * 55)
print("\n[1/4] Fetching SGX stock list...")

resp = requests.get(
    "https://api.sgx.com/securities/v1.1/",
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=30,
)
resp.raise_for_status()
stocks = [s for s in resp.json()["data"]["prices"] if s.get("type") == "stocks"]
ticker_info = {
    f"{s['nc']}.SI": {"symbol": s["nc"], "name": s.get("n", ""), "market": s.get("m", "")}
    for s in stocks if s.get("nc")
}
print(f"    Found {len(ticker_info)} stocks")


# ════════════════════════════════════════════════════════════
# STEP 2 — Download 1-year price history + STI benchmark
# ════════════════════════════════════════════════════════════
end_date   = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
print(f"\n[2/4] Downloading data ({start_date} → {end_date})...")
print(f"    Fetching benchmark {BENCHMARK}...")

sti_raw   = yf.download(BENCHMARK, start=start_date, end=end_date, auto_adjust=True, progress=False)
sti_close = sti_raw["Close"].squeeze()

all_tickers = list(ticker_info.keys())
batches     = [all_tickers[i:i+BATCH_SIZE] for i in range(0, len(all_tickers), BATCH_SIZE)]
stock_data  = {}

for i, batch in enumerate(batches):
    print(f"    Batch {i+1}/{len(batches)} ({len(batch)} tickers)...", end=" ", flush=True)
    try:
        raw = yf.download(batch, start=start_date, end=end_date,
                          auto_adjust=True, progress=False, threads=True)
        if len(batch) == 1:
            if not raw.empty:
                stock_data[batch[0]] = raw.rename(columns=str.lower)
        else:
            for t in batch:
                try:
                    df = raw.xs(t, level=1, axis=1).dropna(how="all")
                    if not df.empty:
                        stock_data[t] = df.rename(columns=str.lower)
                except KeyError:
                    pass
        print(f"ok ({sum(1 for t in batch if t in stock_data)}/{len(batch)})")
    except Exception as e:
        print(f"ERROR: {e}")
    time.sleep(0.3)

print(f"    Total stocks with data: {len(stock_data)}")


# ════════════════════════════════════════════════════════════
# STEP 3 — Run scanners, check last bar
# ════════════════════════════════════════════════════════════
print(f"\n[3/4] Running scanners on {len(stock_data)} stocks...")

results  = []
min_bars = 110   # OBV long SMA needs 100 bars minimum

for idx, (yf_t, df) in enumerate(stock_data.items()):
    if len(df) < min_bars:
        continue
    info = ticker_info[yf_t]
    try:
        df_sma = add_sma_gap_rs_buy_sell_signals(df, sti_close)
        df_bo  = add_breakout_buy_signals(df)
        df_obv = add_obv_signals(df)

        sma_sig      = bool(df_sma["buy_signal"].iloc[-1])
        bo_sig       = bool(df_bo["any_buy_signal"].iloc[-1])
        obv_sell_ban = bool(df_obv["sell_ban"].iloc[-1])   # OBV long SMA slope > 0

        # Signal TRUE if (sma_gap AND obv sell_ban) OR (breakout AND obv sell_ban)
        confirmed = (sma_sig and obv_sell_ban) or (bo_sig and obv_sell_ban)

        results.append({
            "symbol":           info["symbol"],
            "name":             info["name"],
            "market":           info["market"],
            "signal_confirmed": confirmed,
            "sma_gap_buy":      sma_sig,
            "breakout_buy":     bo_sig,
            "obv_sell_ban":     obv_sell_ban,
            "last_close":       round(float(df["close"].iloc[-1]), 4),
            "last_date":        df.index[-1].strftime("%Y-%m-%d"),
            "bars":             len(df),
        })
    except Exception:
        pass

    if (idx + 1) % 50 == 0:
        matched = sum(1 for r in results if r["signal_confirmed"])
        print(f"    {idx+1}/{len(stock_data)} scanned | {matched} confirmed so far")

matched = [r for r in results if r["signal_confirmed"]]
print(f"\n    Scan complete: {len(results)} scanned, {len(matched)} confirmed signals")


# ════════════════════════════════════════════════════════════
# STEP 4 — Export results
# ════════════════════════════════════════════════════════════
print(f"\n[4/4] Exporting results...")

fieldnames = ["symbol", "name", "market", "signal_confirmed",
              "sma_gap_buy", "breakout_buy", "obv_sell_ban",
              "last_close", "last_date", "bars"]

with open(OUTPUT_RESULTS, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(sorted(results, key=lambda r: r["signal_confirmed"], reverse=True))
print(f"    Full results  → {OUTPUT_RESULTS}")

with open(OUTPUT_WATCHLIST, "w", encoding="utf-8") as f:
    for r in matched:
        f.write(f"SGX:{r['symbol']}\n")
print(f"    TV watchlist  → {OUTPUT_WATCHLIST}  ({len(matched)} tickers)")

print("\n" + "=" * 60)
print(f"  CONFIRMED SIGNALS ({len(matched)} stocks)")
print("=" * 60)
if matched:
    print(f"  {'Symbol':<8} {'Name':<30} {'Market':<12} {'SMA':^5} {'BO':^5} {'OBV-SB':^7}")
    print(f"  {'-'*8} {'-'*30} {'-'*12} {'-'*5} {'-'*5} {'-'*7}")
    for r in matched:
        print(f"  {r['symbol']:<8} {r['name'][:30]:<30} {r['market']:<12} "
              f"{'Y' if r['sma_gap_buy'] else 'N':^5} "
              f"{'Y' if r['breakout_buy'] else 'N':^5} "
              f"{'Y' if r['obv_sell_ban'] else 'N':^7}")
else:
    print("  No confirmed signals today.")
print("=" * 60)
print("\nDone! Import into TradingView:")
print("  Watchlist → ⋮ → Import watchlist → sgx_watchlist.txt")
