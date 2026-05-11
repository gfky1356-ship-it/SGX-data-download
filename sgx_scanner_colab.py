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
# Filters applied after data download:
#   volume > 100,000  AND  price > 0.20
#   Only stocks passing both filters proceed to scanning.
#
# FA data sources (in priority order):
#   1. Yahoo Finance (yfinance)  — primary, all fields
#   2. SGX API                   — fallback for P/E, P/B, yield, EPS, mkt cap
#
# Outputs:
#   /content/sgx_scan_results.csv        full results for filtered stocks
#   /content/sgx_watchlist.txt           TradingView watchlist (confirmed only)
#   /content/fa_reports/<TICKER>_FA_...  HTML FA report per confirmed stock
# ============================================================

# ── Cell 1: Install ─────────────────────────────────────────
# !pip install yfinance requests -q


# ── Cell 2: Run Scanner ──────────────────────────────────────
import csv, importlib.util, os, time, warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

OUTPUT_RESULTS   = "/content/sgx_scan_results.csv"
OUTPUT_WATCHLIST = "/content/sgx_watchlist.txt"
FA_REPORTS_DIR   = "/content/fa_reports"
BATCH_SIZE       = 50
BENCHMARK        = "^STI"

# Volume and price thresholds for pre-scan filter
MIN_VOLUME = 100_000
MIN_PRICE  = 0.20


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
# Extended fields (pe, pb, yld, eps, mc) are stored here and
# used as FA fallback in Step 5 when Yahoo Finance is empty.
# ════════════════════════════════════════════════════════════
print("=" * 55)
print("  SGX STOCK SCANNER")
print("=" * 55)
print("\n[1/5] Fetching SGX stock list...")

resp = requests.get(
    "https://api.sgx.com/securities/v1.1/",
    headers={"User-Agent": "Mozilla/5.0"},
    timeout=30,
)
resp.raise_for_status()
stocks = [s for s in resp.json()["data"]["prices"] if s.get("type") == "stocks"]
ticker_info = {
    f"{s['nc']}.SI": {
        "symbol": s["nc"],
        "name":   s.get("n", ""),
        "market": s.get("m", ""),
        # SGX API FA fields — fallback when Yahoo Finance unavailable
        "sgx_pe":  s.get("pe"),                      # trailing P/E
        "sgx_pb":  s.get("pb"),                      # P/B ratio
        "sgx_yld": s.get("yld") or s.get("dy"),      # dividend yield %
        "sgx_eps": s.get("eps") or s.get("es"),      # EPS (S$)
        "sgx_mc":  s.get("mc")  or s.get("mktcap"),  # market cap
    }
    for s in stocks if s.get("nc")
}
print(f"    Found {len(ticker_info)} stocks")


# ════════════════════════════════════════════════════════════
# STEP 2 — Download 1-year price history + STI benchmark
# ════════════════════════════════════════════════════════════
end_date   = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
print(f"\n[2/5] Downloading data ({start_date} → {end_date})...")
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
# STEP 2b — Filter: volume > 100,000 AND price > 0.20
# Only qualified stocks proceed to scanning.
# ════════════════════════════════════════════════════════════
print(f"\n[2b/5] Applying filters  (volume > {MIN_VOLUME:,}  AND  price > {MIN_PRICE})...")

filtered_data = {}
for t, df in stock_data.items():
    last_vol   = float(df["volume"].iloc[-1])
    last_close = float(df["close"].iloc[-1])
    if last_vol > MIN_VOLUME and last_close > MIN_PRICE:
        filtered_data[t] = df

print(f"    Before filter : {len(stock_data)} stocks")
print(f"    After  filter : {len(filtered_data)} stocks  "
      f"({len(stock_data) - len(filtered_data)} removed)")
stock_data = filtered_data


# ════════════════════════════════════════════════════════════
# STEP 3 — Run scanners on filtered tickers, check last bar
# ════════════════════════════════════════════════════════════
print(f"\n[3/5] Running scanners on {len(stock_data)} filtered stocks...")

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
        matched_so_far = sum(1 for r in results if r["signal_confirmed"])
        print(f"    {idx+1}/{len(stock_data)} scanned | {matched_so_far} confirmed so far")

matched = [r for r in results if r["signal_confirmed"]]
print(f"\n    Scan complete: {len(results)} scanned, {len(matched)} confirmed signals")


# ════════════════════════════════════════════════════════════
# STEP 4 — Export scan results
# ════════════════════════════════════════════════════════════
print(f"\n[4/5] Exporting scan results...")

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


# ════════════════════════════════════════════════════════════
# STEP 5 — FA Scan: run GlobalScreenerV59_CN on each
#          confirmed stock and export an HTML report.
#
# Requires: "Glaude Global_FA_Scan_CN_5_9 .py" in the same
#           directory as this script.
#
# FA data priority:
#   1. Yahoo Finance (yfinance) — all fields attempted first
#   2. SGX API                  — fallback for fields that
#      Yahoo Finance could not provide (P/E, P/B, yield, EPS)
# ════════════════════════════════════════════════════════════
print(f"\n[5/5] Running FA scan on {len(matched)} confirmed stocks...")

_fa_filename = "Glaude Global_FA_Scan_CN_5_9 .py"
_fa_path     = os.path.join(os.getcwd(), _fa_filename)

if not matched:
    print("    No confirmed signals — FA scan skipped.")
elif not os.path.exists(_fa_path):
    print(f"    WARNING: FA scan file not found at:\n      {_fa_path}")
    print(f"    Ensure '{_fa_filename}' is in the same directory.")
else:
    # Dynamically load GlobalScreenerV59_CN from the FA scan file
    _spec = importlib.util.spec_from_file_location("fa_scan", _fa_path)
    _fa_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_fa_mod)
    GlobalScreenerV59_CN = _fa_mod.GlobalScreenerV59_CN

    os.makedirs(FA_REPORTS_DIR, exist_ok=True)

    def _sgx_float(sgx_info, key):
        """Safely parse an SGX API value (may be string, '--', None) to float."""
        v = sgx_info.get(key)
        try:
            f = float(v)
            return f if f > 0 else None
        except (TypeError, ValueError):
            return None

    def _fetch_fa_data(yf_ticker, fallback_name, sgx_info):
        """
        Fetch FA fundamentals for one stock.

        Priority order per field:
          1. Yahoo Finance (yfinance .info dict)
          2. SGX API data already captured in Step 1
          3. [NO DATA] — handled gracefully by the screener
        """
        # ── 1. Yahoo Finance ────────────────────────────────
        try:
            yf_info = yf.Ticker(yf_ticker).info
            if not yf_info or yf_info.get("regularMarketPrice") is None:
                yf_info = {}
        except Exception:
            yf_info = {}

        # ── 2. SGX API fallback values ───────────────────────
        sgx_pe  = _sgx_float(sgx_info, "sgx_pe")
        sgx_pb  = _sgx_float(sgx_info, "sgx_pb")   # not used in screener yet, stored for reference
        sgx_eps = _sgx_float(sgx_info, "sgx_eps")  # S$ EPS — used to flag data availability

        # ── Helper converters ────────────────────────────────
        def _pct(v):
            return round(v * 100, 2) if v is not None else "[NO DATA]"

        def _num(v, dp=2):
            return round(v, dp) if v is not None else "[NO DATA]"

        def _pe(v):
            return round(v, 2) if v is not None else "—"

        # ── Field mapping (Yahoo first, SGX fallback) ────────
        # P/E: Yahoo trailingPE → SGX pe
        yf_pe  = yf_info.get("trailingPE")
        ttm_pe = _pe(yf_pe if yf_pe is not None else sgx_pe)

        # Forward P/E: Yahoo only
        fwd_pe = _pe(yf_info.get("forwardPE"))

        # Profitability & growth: Yahoo only
        roe        = _pct(yf_info.get("returnOnEquity"))
        margin     = _pct(yf_info.get("profitMargins"))
        rev_cagr   = _pct(yf_info.get("revenueGrowth"))
        eps_raw    = yf_info.get("earningsGrowth") or yf_info.get("earningsQuarterlyGrowth")
        eps_growth = _pct(eps_raw)
        de_raw     = yf_info.get("debtToEquity")
        debt_eq    = round(de_raw / 100, 2) if de_raw is not None else "[NO DATA]"
        p_fcf      = _num(yf_info.get("priceToFreeCashflows"))

        # Rule of 40 = revenue growth % + net margin %
        rule40 = "[NO DATA]"
        if isinstance(rev_cagr, float) and isinstance(margin, float):
            rule40 = round(rev_cagr + margin, 2)

        # Most recent quarter date
        mrq      = yf_info.get("mostRecentQuarter")
        rel_date = datetime.fromtimestamp(mrq).strftime("%Y-%m-%d") if mrq else "N/A"

        # Bank / REIT detection
        sector   = yf_info.get("sector", "")
        industry = yf_info.get("industry", "")
        is_bank  = any(k in (sector + industry).lower()
                       for k in ["bank", "financ", "reit", "trust", "insur"])

        # Company name: Yahoo → SGX name field → fallback
        company_name = (yf_info.get("longName") or yf_info.get("shortName")
                        or sgx_info.get("name") or fallback_name)

        # Log which source was used for P/E
        pe_source = "Yahoo" if yf_pe is not None else ("SGX" if sgx_pe is not None else "N/A")

        return {
            "company_name":   company_name,
            "releasing_date": rel_date,
            "is_bank":        is_bank,
            "eps_growth":     eps_growth,
            "rev_cagr":       rev_cagr,
            "roe":            roe,
            "roic":           "[NO DATA]",   # requires balance-sheet calculation
            "p_fcf":          p_fcf,
            "ttm_pe":         ttm_pe,
            "forward_pe":     fwd_pe,
            "margin":         margin,
            "debt_equity":    debt_eq,
            "rule_of_40":     rule40,
            "_pe_source":     pe_source,     # internal — for console log only
        }

    fa_reports  = []
    fa_failures = []

    for r in matched:
        yf_t     = f"{r['symbol']}.SI"
        sgx_info = ticker_info.get(yf_t, {})
        print(f"    FA → {r['symbol']:<8} {r['name'][:32]:<32}", end=" ", flush=True)
        try:
            fa_data     = _fetch_fa_data(yf_t, r["name"], sgx_info)
            pe_src      = fa_data.pop("_pe_source", "?")   # remove internal key before passing to screener
            screener    = GlobalScreenerV59_CN(r["symbol"], r["last_close"], fa_data)
            report_path = screener.save_report(output_dir=FA_REPORTS_DIR)
            fa_reports.append(report_path)
            _, verdict_label, _ = screener.get_verdict()
            print(f"✅  {verdict_label}  [PE src: {pe_src}]")
        except Exception as e:
            fa_failures.append(r["symbol"])
            print(f"❌  {e}")
        time.sleep(0.5)   # be polite to Yahoo Finance

    print(f"\n    FA reports saved  → {FA_REPORTS_DIR}/")
    print(f"    Reports generated : {len(fa_reports)}")
    if fa_failures:
        print(f"    Failed            : {', '.join(fa_failures)}")

print("\n" + "=" * 60)
print("  DONE")
print(f"  Scan results  : {OUTPUT_RESULTS}")
print(f"  TV watchlist  : {OUTPUT_WATCHLIST}")
print(f"  FA reports    : {FA_REPORTS_DIR}/")
print("=" * 60)
print("\nImport into TradingView:")
print("  Watchlist → ⋮ → Import watchlist → sgx_watchlist.txt")
