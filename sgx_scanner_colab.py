# ============================================================
# SGX STOCK SCANNER — Colab Ready  (v7.0, SGX only)
#
# ── Quick Start (3 Colab cells) ─────────────────────────────
#
#   Cell 1 — Mount Google Drive (run once, approve the popup):
#     from google.colab import drive
#     drive.mount('/content/drive')
#
#   Cell 2 — Download script:
#     !wget -qO sgx_scanner_colab.py https://raw.githubusercontent.com/gfky1356-ship-it/SGX-data-download/main/sgx_scanner_colab.py \
#       && pip install yfinance requests weasyprint pypdf anthropic -q
#
#   Cell 3 — Run:
#     !python sgx_scanner_colab.py
#
# ─────────────────────────────────────────────────────────────
#
# Market    : Singapore Exchange (SGX)
# Benchmark : ^STI  (TradingView: TVC:STI)
# Filters   : volume > 100,000  AND  price > S$0.20
#
# Scanners:
#   1. SMA Gap Enlarge Buy
#   2. Breakout Buy
#   3. OBV (sell_ban confirmation)
#
# Outputs (saved to Google Drive → MyDrive/StockScanner/):
#   sgx_scan_results.csv
#   sgx_watchlist.txt             ← import into TradingView
#   sgx_rsi_comparison.csv        ← RSI14 ratio vs TVC:STI
#   benchmark_data/STI.csv
#   fa_reports/<TICKER>_FA_v6.0_<date>.html
#   fa_reports/SGX_FA_REPORTS_<date>.pdf
# ============================================================

# !pip install yfinance requests weasyprint pypdf anthropic -q

import csv, os, time, warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Google Drive ─────────────────────────────────────────────
# Drive must be mounted in a separate Colab cell BEFORE running
# this script (drive.mount() cannot run inside a subprocess).
# If already mounted, we just detect the path automatically.
if os.path.isdir("/content/drive/MyDrive"):
    GDRIVE_BASE = "/content/drive/MyDrive/StockScanner"
    print(f"✅ Google Drive detected → {GDRIVE_BASE}")
else:
    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        GDRIVE_BASE = "/content/drive/MyDrive/StockScanner"
        print(f"✅ Google Drive mounted → {GDRIVE_BASE}")
    except Exception:
        GDRIVE_BASE = "/content"
        print("⚠️  Google Drive not mounted — saving to /content instead.\n"
              "    To save to Drive: run  from google.colab import drive; drive.mount('/content/drive')\n"
              "    in a separate cell first, then re-run this script.")

os.makedirs(GDRIVE_BASE, exist_ok=True)
BENCHMARK_DATA_DIR = os.path.join(GDRIVE_BASE, "benchmark_data")
os.makedirs(BENCHMARK_DATA_DIR, exist_ok=True)

# ── Output paths ─────────────────────────────────────────────
OUTPUT_RESULTS    = os.path.join(GDRIVE_BASE, "sgx_scan_results.csv")
OUTPUT_WATCHLIST  = os.path.join(GDRIVE_BASE, "sgx_watchlist.txt")
OUTPUT_RSI        = os.path.join(GDRIVE_BASE, "sgx_rsi_comparison.csv")
FA_REPORTS_DIR    = os.path.join(GDRIVE_BASE, "fa_reports")
COMBINED_FA_PDF   = os.path.join(FA_REPORTS_DIR, f"SGX_FA_REPORTS_{datetime.now().strftime('%Y%m%d')}.pdf")

# ── Settings ─────────────────────────────────────────────────
BENCHMARK   = "^STI"
MIN_VOLUME  = 100_000
MIN_PRICE   = 0.20
BATCH_SIZE  = 50
MIN_BARS    = 110


# ════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════
# FA ANALYSIS — powered by Claude AI
# Set ANTHROPIC_API_KEY before running:
#   Colab: add to Colab Secrets (key icon in sidebar)
#   Local: export ANTHROPIC_API_KEY="sk-ant-..."
# ════════════════════════════════════════════════════════════
FA_PROMPT_URL = (
    "https://raw.githubusercontent.com/gfky1356-ship-it/"
    "Stock-Financial-Analysis-Script/main/Prompt%20file%20for%20FA%20analysis"
)
FA_MODEL = "claude-sonnet-4-5"


def _get_fa_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            from google.colab import userdata
            key = userdata.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    return key or ""


def _fetch_fa_prompt():
    try:
        resp = requests.get(FA_PROMPT_URL, timeout=15)
        resp.raise_for_status()
        return resp.text
    except Exception as e:
        print(f"    ⚠️  Could not fetch FA prompt: {e}")
        return None


def _save_fa_html(ticker, company_name, market, price, currency, analysis_text, output_dir):
    ts = datetime.now().strftime("%Y%m%d")
    filepath = os.path.join(output_dir, f"{ticker}_FA_Claude_{ts}.html")
    escaped = analysis_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    now_str = datetime.now().strftime("%Y-%m-%d")
    now_dt  = datetime.now().strftime("%Y-%m-%d %H:%M")
    parts = [
        "<!DOCTYPE html><html lang='en'><head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1.0'>",
        "<title>" + ticker + " FA Analysis</title>",
        "<style>",
        "  body{background:#f5f5f0;color:#1a1a2e;font-family:monospace;font-size:13px;line-height:1.8;padding:16px}",
        "  .report{width:100%;max-width:900px;margin:0 auto;display:flex;flex-direction:column;gap:12px}",
        "  .hdr{background:#fff;border:1px solid #d0d0d8;border-left:4px solid #0a7c5c;padding:14px 16px;box-shadow:0 1px 4px rgba(0,0,0,.06)}",
        "  .tkr{font-size:24px;font-weight:600;color:#0a7c5c}",
        "  .sub{font-size:11px;color:#555568;margin-top:4px}",
        "  .bdy{background:#fff;border:1px solid #d8d8e0;padding:16px 20px;white-space:pre-wrap;box-shadow:0 1px 3px rgba(0,0,0,.04)}",
        "  .ftr{font-size:10px;color:#888898;text-align:right;padding-top:4px}",
        "</style></head><body><div class='report'>",
        "<div class='hdr'>",
        "  <div class='tkr'>" + ticker + " <span style='color:#555568;font-size:15px;font-weight:400'>@ " + currency + str(price) + "</span></div>",
        "  <div class='sub'>" + company_name + " · " + market + " · Analysis by " + FA_MODEL + " · " + now_str + "</div>",
        "</div>",
        "<div class='bdy'>" + escaped + "</div>",
        "<div class='ftr'>" + FA_MODEL + " · " + now_dt + " · For reference only, not investment advice</div>",
        "</div></body></html>",
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return filepath


# SCANNER FUNCTIONS
# ════════════════════════════════════════════════════════════
def _sma(series, length):
    return series.rolling(length, min_periods=length).mean()

def calc_rsi(series, period=14):
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def add_sma_gap_rs_buy_sell_signals(df, index_close,
    b_sma20_len=20, b_sma50_len=50, b_slope_lookback=5,
    b_slope_smooth_len=1, a_rs_sma_length=30, a_delta_ma_length=9):
    out = df.copy()
    index_close = index_close.reindex(out.index).astype(float)
    out["b_sma20"]       = _sma(out["close"], b_sma20_len)
    out["b_sma50"]       = _sma(out["close"], b_sma50_len)
    out["b_raw_slope20"] = (out["b_sma20"] - out["b_sma20"].shift(b_slope_lookback)) / out["b_sma20"].shift(b_slope_lookback) * 100.0
    out["b_raw_slope50"] = (out["b_sma50"] - out["b_sma50"].shift(b_slope_lookback)) / out["b_sma50"].shift(b_slope_lookback) * 100.0
    out["b_slope20"]     = _sma(out["b_raw_slope20"], b_slope_smooth_len)
    out["b_slope50"]     = _sma(out["b_raw_slope50"], b_slope_smooth_len)
    out["b_bull"]        = out["b_slope20"] > out["b_slope50"]
    stock_sma = _sma(out["close"], a_rs_sma_length)
    index_sma = _sma(index_close, a_rs_sma_length)
    out["delta_ab"]  = (out["close"] / stock_sma) - (index_close / index_sma)
    out["delta_ma"]  = _sma(out["delta_ab"], a_delta_ma_length)
    out["slope_gap"] = out["b_slope20"] - out["b_slope50"]
    out["a_two_day_green_rising"] = ((out["delta_ab"] >= 0) & (out["delta_ab"].shift(1) >= 0) & (out["delta_ab"] > out["delta_ab"].shift(1)))
    out["green_candle"]       = out["close"] > out["open"]
    out["red_candle"]         = out["close"] < out["open"]
    out["slope_gap_reducing"] = (out["b_slope20"] > out["b_slope50"]) & (out["slope_gap"] < out["slope_gap"].shift(1))
    out["buy_condition"]  = out["b_bull"] & out["a_two_day_green_rising"] & out["green_candle"]
    out["sell_condition"] = (out["delta_ab"] < out["delta_ma"]) & out["slope_gap_reducing"] & out["red_candle"]
    buy_signal = np.zeros(len(out), dtype=bool)
    sell_signal = np.zeros(len(out), dtype=bool)
    in_position = buy_ban = sell_ban = False
    for i in range(len(out)):
        buy  = bool(out["buy_condition"].iat[i])  and not buy_ban  and not in_position
        sell = bool(out["sell_condition"].iat[i]) and not sell_ban and in_position
        if buy:  buy_signal[i]=True;  in_position=True;  buy_ban=True;  sell_ban=False
        if sell: sell_signal[i]=True; in_position=False; sell_ban=True; buy_ban=False
    out["buy_signal"]  = buy_signal
    out["sell_signal"] = sell_signal
    return out

def add_breakout_buy_signals(df, lookback_n1=30, lookback_n2=60, lookback_n3=90,
    sma200_slope_lookback=5, max_upside_pct=5.0):
    out = df.copy()
    out["sma20"]  = _sma(out["close"], 20)
    out["sma50"]  = _sma(out["close"], 50)
    out["sma200"] = _sma(out["close"], 200)
    out["sma200_slope_positive"] = out["sma200"] > out["sma200"].shift(sma200_slope_lookback)
    out["green_candle"]  = out["close"] > out["open"]
    out["sma20_above50"] = out["sma20"] > out["sma50"]
    out["b_above20sma"]  = out["close"] > out["sma20"]
    for label, lookback in [("n1",lookback_n1),("n2",lookback_n2),("n3",lookback_n3)]:
        signals = []
        for pos in range(len(out)):
            if pos < 1: signals.append(False); continue
            end_i = min(lookback, pos)
            a_price, a_offset = float("nan"), None
            for off in range(1, end_i+1):
                h = out["high"].iat[pos-off]
                if pd.notna(h) and (pd.isna(a_price) or h > a_price): a_price, a_offset = h, off
            pb = False
            if a_offset:
                for off in range(1, end_i+1):
                    if off < a_offset:
                        if pd.notna(out["low"].iat[pos-off]) and pd.notna(out["sma20"].iat[pos-off]):
                            if out["low"].iat[pos-off] < out["sma20"].iat[pos-off]: pb = True
            a_above  = (a_offset is not None and pd.notna(a_price) and pd.notna(out["sma20"].iat[pos-a_offset]) and a_price > out["sma20"].iat[pos-a_offset])
            b_cross  = pd.notna(a_price) and out["close"].iat[pos] > a_price and out["close"].iat[pos-1] <= a_price
            b_within = pd.notna(a_price) and out["close"].iat[pos] <= a_price*(1+max_upside_pct/100)
            sig = (bool(out["sma200_slope_positive"].iat[pos]) and bool(out["green_candle"].iat[pos]) and
                   bool(out["sma20_above50"].iat[pos]) and a_above and bool(out["b_above20sma"].iat[pos]) and
                   pb and b_cross and b_within)
            signals.append(sig)
        out[f"{label}_signal"] = signals
    out["any_buy_signal"] = out["n1_signal"] | out["n2_signal"] | out["n3_signal"]
    return out

def add_obv_signals(df, short_sma_length=20, long_sma_length=100, confirm_days=1.0):
    out = df.copy()
    close, volume = out["close"], out["volume"]
    dv = np.where(close > close.shift(1), volume, np.where(close < close.shift(1), -volume, 0))
    out["obv"]            = pd.Series(dv, index=out.index).cumsum()
    out["obv_short_sma"]  = _sma(out["obv"], short_sma_length)
    out["obv_long_sma"]   = _sma(out["obv"], long_sma_length)
    out["long_sma_slope"] = out["obv_long_sma"] - out["obv_long_sma"].shift(1)
    out["sell_ban"]       = out["long_sma_slope"] > 0
    out["buy_ban"]        = out["long_sma_slope"] < 0
    out["obv_above_both"] = (out["obv"] > out["obv_short_sma"]) & (out["obv"] > out["obv_long_sma"])
    out["obv_below_both"] = (out["obv"] < out["obv_short_sma"]) & (out["obv"] < out["obv_long_sma"])
    buy_signal = np.zeros(len(out), dtype=bool)
    sell_signal = np.zeros(len(out), dtype=bool)
    above_start = below_start = None
    buy_triggered = sell_triggered = False
    for i in range(len(out)):
        above = bool(out["obv_above_both"].iat[i]) if pd.notna(out["obv_above_both"].iat[i]) else False
        below = bool(out["obv_below_both"].iat[i]) if pd.notna(out["obv_below_both"].iat[i]) else False
        if above:
            if above_start is None: above_start = i
            below_start = None; sell_triggered = False
        else: above_start = None; buy_triggered = False
        if below:
            if below_start is None: below_start = i
            above_start = None; buy_triggered = False
        else: below_start = None; sell_triggered = False
        def elapsed(start):
            idx = out.index
            return ((idx[i]-idx[start]).total_seconds()/86400.0 if isinstance(idx, pd.DatetimeIndex) else float(i-start))
        if above and above_start is not None and elapsed(above_start) > confirm_days:
            if not buy_triggered and not bool(out["buy_ban"].iat[i]):
                buy_signal[i] = True; buy_triggered = True
        if below and below_start is not None and elapsed(below_start) > confirm_days:
            if not sell_triggered and not bool(out["sell_ban"].iat[i]):
                sell_signal[i] = True; sell_triggered = True
    out["buy_signal"]  = buy_signal
    out["sell_signal"] = sell_signal
    return out

def save_benchmark_to_csv(close_series, name, directory):
    safe_name = name.replace("^", "")
    path = os.path.join(directory, f"{safe_name}.csv")
    close_series.to_frame(name="close").to_csv(path)
    print(f"    Benchmark saved → {path}")
    return path

def combine_fa_reports_to_pdf(html_paths, output_path):
    if not html_paths: return None
    try:
        from weasyprint import HTML as WH
        import pypdf
    except ImportError:
        print("    ⚠️  PDF combine skipped — run: !pip install weasyprint pypdf -q")
        return None
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_pdfs = []
    for hp in html_paths:
        pp = hp.replace(".html", "_tmp.pdf")
        try: WH(filename=hp).write_pdf(pp); tmp_pdfs.append(pp)
        except Exception as e: print(f"    ⚠️  {os.path.basename(hp)}: {e}")
    if not tmp_pdfs: return None
    try:
        merger = pypdf.PdfMerger()
        for pp in tmp_pdfs: merger.append(pp)
        merger.write(output_path); merger.close()
        print(f"    ✅ Combined FA PDF ({len(tmp_pdfs)} reports) → {output_path}")
        return output_path
    except Exception as e: print(f"    ❌ PDF merge failed: {e}"); return None
    finally:
        for pp in tmp_pdfs:
            try: os.remove(pp)
            except: pass

def run_fa_scan(matched, ticker_info):
    if not matched:
        print("    No confirmed signals — FA scan skipped.")
        return []

    api_key = _get_fa_api_key()
    if not api_key:
        print("    ⚠️  ANTHROPIC_API_KEY not set — FA scan skipped.")
        print("        Colab: add key to Colab Secrets (key icon in sidebar)")
        print("        Local: export ANTHROPIC_API_KEY='sk-ant-...'")
        return []

    print("    Fetching FA prompt from GitHub...", end=" ", flush=True)
    system_prompt = _fetch_fa_prompt()
    if not system_prompt:
        return []
    print("OK")

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)

    os.makedirs(FA_REPORTS_DIR, exist_ok=True)
    fa_reports = []
    fa_failures = []

    for r in matched:
        symbol   = r["symbol"]
        name     = r.get("name", symbol)
        price    = r["last_close"]
        currency = "S$"
        market   = "Singapore (SGX)"

        print(f"    FA → {symbol:<8} {name[:32]:<32}", end=" ", flush=True)

        user_msg = (
            f"Please perform a full FA analysis for the stock ticker: {symbol}\n"
            f"Company: {name}\n"
            f"Market: {market}\n"
            f"Latest stock price (already downloaded from market data): {currency}{price}\n\n"
            f"IMPORTANT: Use {currency}{price} as the current stock price for all PE ratio "
            f"calculations. Do not re-fetch or re-download the price.\n\n"
            "Follow the framework exactly:\n"
            "1. Apply SGX-specific benchmarks (EPS growth ≥0%, Revenue CAGR >10%, ROE >9%, etc.)\n"
            "2. Retrieve the latest financial data from SGX / Yahoo Finance\n"
            "3. Score each metric against the SGX benchmarks\n"
            "4. Provide the qualitative Phase 2 analysis\n"
            "5. Give your final FIRE / WAIT / AVOID recommendation with clear reasoning"
        )

        try:
            response_text = ""
            with client.messages.stream(
                model=FA_MODEL,
                max_tokens=4096,
                system=[{
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{"role": "user", "content": user_msg}],
            ) as stream:
                for chunk in stream.text_stream:
                    response_text += chunk

            path = _save_fa_html(symbol, name, market, price, currency, response_text, FA_REPORTS_DIR)
            fa_reports.append(path)

            verdict = ("FIRE" if "FIRE" in response_text.upper()
                       else ("AVOID" if "AVOID" in response_text.upper() else "WAIT"))
            print(f"✅  {verdict}  [Price:{currency}{price}]")

        except Exception as e:
            fa_failures.append(symbol)
            print(f"❌  {e}")
        time.sleep(0.3)

    print(f"\n    FA reports: {len(fa_reports)} saved to {FA_REPORTS_DIR}/")
    if fa_failures:
        print(f"    Failed: {', '.join(fa_failures)}")
    return fa_reports

# SGX SCAN
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("  SGX STOCK SCANNER")
print(f"  Filters: vol > {MIN_VOLUME:,}  |  price > S${MIN_PRICE}")
print(f"  Benchmark: {BENCHMARK}  (TradingView: TVC:STI)")
print("=" * 60)

# Step 1: Fetch SGX ticker list
print("\n[1/5] Fetching SGX stock list...")
resp = requests.get("https://api.sgx.com/securities/v1.1/",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
resp.raise_for_status()
stocks = [s for s in resp.json()["data"]["prices"] if s.get("type") == "stocks"]
ticker_info = {
    f"{s['nc']}.SI": {"symbol": s["nc"], "name": s.get("n",""), "market": s.get("m",""),
                      "sgx_pe": s.get("pe"), "sgx_pb": s.get("pb"),
                      "sgx_yld": s.get("yld") or s.get("dy"),
                      "sgx_eps": s.get("eps") or s.get("es"),
                      "sgx_mc":  s.get("mc")  or s.get("mktcap")}
    for s in stocks if s.get("nc")
}
print(f"    Found {len(ticker_info)} stocks")

# Step 2: Download benchmark + price data
end_date   = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
print(f"\n[2/5] Downloading data ({start_date} → {end_date})...")

bench_raw   = yf.download(BENCHMARK, start=start_date, end=end_date, auto_adjust=True, progress=False)
bench_close = bench_raw["Close"].squeeze()
bench_rsi14 = calc_rsi(bench_close)
save_benchmark_to_csv(bench_close, "STI", BENCHMARK_DATA_DIR)
print(f"    TVC:STI RSI14 (latest): {float(bench_rsi14.iloc[-1]):.2f}")

all_tickers = list(ticker_info.keys())
batches     = [all_tickers[i:i+BATCH_SIZE] for i in range(0, len(all_tickers), BATCH_SIZE)]
stock_data  = {}
for i, batch in enumerate(batches):
    print(f"    Batch {i+1}/{len(batches)} ...", end=" ", flush=True)
    try:
        raw = yf.download(batch, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=True)
        if len(batch) == 1:
            if not raw.empty: stock_data[batch[0]] = raw.rename(columns=str.lower)
        else:
            for t in batch:
                try:
                    df = raw.xs(t, level=1, axis=1).dropna(how="all")
                    if not df.empty: stock_data[t] = df.rename(columns=str.lower)
                except KeyError: pass
        print(f"ok ({sum(1 for t in batch if t in stock_data)}/{len(batch)})")
    except Exception as e: print(f"ERROR: {e}")
    time.sleep(0.3)
print(f"    Total with data: {len(stock_data)}")

# Step 2b: Filter
print(f"\n[2b/5] Filtering  vol > {MIN_VOLUME:,}  AND  price > S${MIN_PRICE} ...")
filtered = {t: df for t, df in stock_data.items()
            if float(df["volume"].iloc[-1]) > MIN_VOLUME and float(df["close"].iloc[-1]) > MIN_PRICE}
print(f"    Before: {len(stock_data)}  →  After: {len(filtered)}  ({len(stock_data)-len(filtered)} removed)")

# Step 3: Run scanners
print(f"\n[3/5] Running scanners on {len(filtered)} stocks...")
results     = []
idx_rsi_val = float(bench_rsi14.iloc[-1]) if pd.notna(bench_rsi14.iloc[-1]) else None

for idx, (yf_t, df) in enumerate(filtered.items()):
    if len(df) < MIN_BARS: continue
    info = ticker_info[yf_t]
    try:
        df_sma = add_sma_gap_rs_buy_sell_signals(df, bench_close)
        df_bo  = add_breakout_buy_signals(df)
        df_obv = add_obv_signals(df)
        sma_sig      = bool(df_sma["buy_signal"].iloc[-1])
        bo_sig       = bool(df_bo["any_buy_signal"].iloc[-1])
        obv_sell_ban = bool(df_obv["sell_ban"].iloc[-1])
        confirmed    = (sma_sig and obv_sell_ban) or (bo_sig and obv_sell_ban)
        stock_rsi_s  = calc_rsi(df["close"])
        stock_rsi14  = round(float(stock_rsi_s.iloc[-1]), 2) if pd.notna(stock_rsi_s.iloc[-1]) else None
        rsi_ratio    = round(stock_rsi14 / idx_rsi_val, 4) if stock_rsi14 and idx_rsi_val and idx_rsi_val > 0 else None
        results.append({
            "symbol": info["symbol"], "name": info["name"], "market": info["market"],
            "yf_ticker": yf_t, "signal_confirmed": confirmed,
            "sma_gap_buy": sma_sig, "breakout_buy": bo_sig, "obv_sell_ban": obv_sell_ban,
            "last_close": round(float(df["close"].iloc[-1]), 4),
            "last_date":  df.index[-1].strftime("%Y-%m-%d"), "bars": len(df),
            "rsi14": stock_rsi14, "index_rsi14": round(idx_rsi_val, 2) if idx_rsi_val else None,
            "rsi_ratio": rsi_ratio,
        })
    except Exception: pass
    if (idx+1) % 50 == 0:
        print(f"    {idx+1}/{len(filtered)} scanned | {sum(1 for r in results if r['signal_confirmed'])} confirmed")

matched = [r for r in results if r["signal_confirmed"]]
print(f"\n    Done: {len(results)} scanned, {len(matched)} confirmed signals")

# Step 4: Export
print(f"\n[4/5] Exporting results...")
fieldnames  = ["symbol","name","market","signal_confirmed","sma_gap_buy","breakout_buy",
               "obv_sell_ban","last_close","last_date","bars","rsi14","index_rsi14","rsi_ratio"]
rsi_fields  = ["tv_ticker","symbol","name","rsi14","index_rsi14","rsi_ratio","index_ticker","last_date"]

with open(OUTPUT_RESULTS, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
    w.writerows(sorted([{k:r[k] for k in fieldnames} for r in results], key=lambda r: r["signal_confirmed"], reverse=True))
print(f"    Full results  → {OUTPUT_RESULTS}")

with open(OUTPUT_WATCHLIST, "w", encoding="utf-8") as f:
    for r in matched: f.write(f"SGX:{r['symbol']}\n")
print(f"    TV watchlist  → {OUTPUT_WATCHLIST}  ({len(matched)} tickers)")

with open(OUTPUT_RSI, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rsi_fields); w.writeheader()
    for r in matched:
        w.writerow({"tv_ticker": f"SGX:{r['symbol']}", "symbol": r["symbol"], "name": r["name"],
                    "rsi14": r["rsi14"], "index_rsi14": r["index_rsi14"], "rsi_ratio": r["rsi_ratio"],
                    "index_ticker": "TVC:STI", "last_date": r["last_date"]})
print(f"    RSI comparison → {OUTPUT_RSI}")

print("\n" + "=" * 60)
print(f"  SGX CONFIRMED SIGNALS ({len(matched)} stocks)")
print("=" * 60)
if matched:
    print(f"  {'Symbol':<8} {'Name':<28} {'Close':>7} {'SMA':^4} {'BO':^4} {'OBV':^4} {'RSI14':>6} {'IdxRSI':>7} {'Ratio':>6}")
    print(f"  {'-'*8} {'-'*28} {'-'*7} {'-'*4} {'-'*4} {'-'*4} {'-'*6} {'-'*7} {'-'*6}")
    for r in matched:
        print(f"  {r['symbol']:<8} {r['name'][:28]:<28} S${r['last_close']:>5.3f} "
              f"{'Y' if r['sma_gap_buy'] else 'N':^4} {'Y' if r['breakout_buy'] else 'N':^4} "
              f"{'Y' if r['obv_sell_ban'] else 'N':^4} "
              f"{str(r['rsi14'] or 'N/A'):>6} {str(r['index_rsi14'] or 'N/A'):>7} {str(r['rsi_ratio'] or 'N/A'):>6}")
else:
    print("  No confirmed signals today.")
print("=" * 60)

# Step 5: FA scan + combine PDF
print(f"\n[5/5] FA scan on {len(matched)} confirmed SGX stocks...")
fa_reports = run_fa_scan(matched, ticker_info)
if fa_reports:
    print(f"\n[PDF] Combining {len(fa_reports)} FA reports...")
    combine_fa_reports_to_pdf(fa_reports, COMBINED_FA_PDF)

# Summary
print("\n\n" + "=" * 60)
print("  SGX SCAN COMPLETE")
print("=" * 60)
print(f"  All files saved to: {GDRIVE_BASE}/")
print(f"  Results    : {OUTPUT_RESULTS}")
print(f"  Watchlist  : {OUTPUT_WATCHLIST}  ({len(matched)} signals)")
print(f"  RSI report : {OUTPUT_RSI}  (vs TVC:STI)")
print(f"  Benchmark  : {BENCHMARK_DATA_DIR}/STI.csv")
print(f"  FA reports : {FA_REPORTS_DIR}/  ({len(fa_reports)} HTML)")
print(f"  FA PDF     : {COMBINED_FA_PDF}")
print("=" * 60)
print("\nImport into TradingView: Watchlist → ⋮ → Import → sgx_watchlist.txt")
print("RSI ratio > 1.0 → stock stronger than TVC:STI")
