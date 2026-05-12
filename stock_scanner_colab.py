# ============================================================
# SGX + US STOCK SCANNER — Colab Ready  (v7.0, fully self-contained)
#
# ── Quick Start (paste into Google Colab) ───────────────────
# Private repo (replace YOUR_TOKEN with your GitHub PAT):
#   !git clone -q https://YOUR_TOKEN@github.com/gfky1356-ship-it/sgx-data-download.git \
#     && pip install yfinance requests weasyprint pypdf anthropic -q
#   !python sgx-data-download/stock_scanner_colab.py
#
# Public repo (no token needed):
#   !wget -qO stock_scanner_colab.py https://raw.githubusercontent.com/gfky1356-ship-it/sgx-data-download/main/stock_scanner_colab.py \
#     && pip install yfinance requests weasyprint pypdf anthropic -q
#   !python stock_scanner_colab.py
# ─────────────────────────────────────────────────────────────
#
# SGX scan filters : volume > 100,000  AND  price > S$0.20
# US  scan filters : volume > 200,000  AND  price > US$10.00
#
# Scanners (shared for both markets):
#   1. SMA Gap Enlarge Buy
#   2. Breakout Buy
#   3. OBV (sell_ban confirmation)
#
# FA data sources (in priority order):
#   1. Yahoo Finance (yfinance) — all fields
#   2. SGX API                  — fallback P/E, P/B, yield, EPS
#
# Price used in FA report = last close from downloaded data.
#
# NEW in v7.0:
#   - Google Drive auto-mount; ALL outputs saved to Drive
#   - FA HTML reports combined into one PDF (weasyprint + pypdf)
#   - SPY (US) and ^STI (SG) benchmark data saved as CSV
#   - 14-period RSI computed per stock vs its market benchmark
#   - RSI ratio (stock RSI14 / index RSI14) in results + watchlist report
#   - data_adapter.py + screener_module.py for modular future use
#
# Outputs (all saved to Google Drive → MyDrive/StockScanner/):
#   sgx_scan_results.csv          us_scan_results.csv
#   sgx_watchlist.txt             us_watchlist.txt
#   sgx_rsi_comparison.csv        us_rsi_comparison.csv
#   benchmark_data/STI.csv        benchmark_data/SPY.csv
#   fa_reports/<TICKER>_FA_v6.0_<date>.html
#   fa_reports/ALL_FA_REPORTS_<date>.pdf
# ============================================================

# !pip install yfinance requests weasyprint pypdf anthropic -q

import csv, os, re, time, warnings
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

# ── SGX output paths ────────────────────────────────────────
SGX_OUTPUT_RESULTS   = os.path.join(GDRIVE_BASE, "sgx_scan_results.csv")
SGX_OUTPUT_WATCHLIST = os.path.join(GDRIVE_BASE, "sgx_watchlist.txt")
SGX_RSI_COMPARISON   = os.path.join(GDRIVE_BASE, "sgx_rsi_comparison.csv")

# ── US output paths ─────────────────────────────────────────
US_OUTPUT_RESULTS    = os.path.join(GDRIVE_BASE, "us_scan_results.csv")
US_OUTPUT_WATCHLIST  = os.path.join(GDRIVE_BASE, "us_watchlist.txt")
US_RSI_COMPARISON    = os.path.join(GDRIVE_BASE, "us_rsi_comparison.csv")

FA_REPORTS_DIR       = os.path.join(GDRIVE_BASE, "fa_reports")
COMBINED_FA_PDF      = os.path.join(
    FA_REPORTS_DIR, f"ALL_FA_REPORTS_{datetime.now().strftime('%Y%m%d')}.pdf"
)

# ── Benchmarks ──────────────────────────────────────────────
# US benchmark: SPY (S&P 500 ETF, matches TradingView SPY ticker)
# SG benchmark: ^STI (Straits Times Index, matches TradingView TVC:STI)
SGX_BENCHMARK        = "^STI"
US_BENCHMARK         = "SPY"

BATCH_SIZE           = 50

# ── Filters ─────────────────────────────────────────────────
SGX_MIN_VOLUME = 100_000
SGX_MIN_PRICE  = 0.20
US_MIN_VOLUME  = 200_000
US_MIN_PRICE   = 10.00


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


# SHARED SCANNER FUNCTIONS (used for both SGX and US)
# ════════════════════════════════════════════════════════════
def _sma(series, length):
    return series.rolling(length, min_periods=length).mean()


def calc_rsi(series, period=14):
    """Wilder's smoothed RSI."""
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
    stock_sma        = _sma(out["close"], a_rs_sma_length)
    index_sma        = _sma(index_close, a_rs_sma_length)
    out["delta_ab"]  = (out["close"] / stock_sma) - (index_close / index_sma)
    out["delta_ma"]  = _sma(out["delta_ab"], a_delta_ma_length)
    out["slope_gap"] = out["b_slope20"] - out["b_slope50"]
    out["a_two_day_green_rising"] = (
        (out["delta_ab"] >= 0) & (out["delta_ab"].shift(1) >= 0) &
        (out["delta_ab"] > out["delta_ab"].shift(1)))
    out["green_candle"]       = out["close"] > out["open"]
    out["red_candle"]         = out["close"] < out["open"]
    out["slope_gap_reducing"] = (out["b_slope20"] > out["b_slope50"]) & (out["slope_gap"] < out["slope_gap"].shift(1))
    out["buy_condition"]      = out["b_bull"] & out["a_two_day_green_rising"] & out["green_candle"]
    out["sell_condition"]     = (out["delta_ab"] < out["delta_ma"]) & out["slope_gap_reducing"] & out["red_candle"]
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
            a_above = (a_offset is not None and pd.notna(a_price) and
                       pd.notna(out["sma20"].iat[pos-a_offset]) and a_price > out["sma20"].iat[pos-a_offset])
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
            return ((idx[i]-idx[start]).total_seconds()/86400.0
                    if isinstance(idx, pd.DatetimeIndex) else float(i-start))
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
# BENCHMARK HELPERS
# ════════════════════════════════════════════════════════════
_TV_EXCHANGE_MAP = {
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ",
    "NasdaqGS": "NASDAQ", "NasdaqGM": "NASDAQ", "NasdaqCM": "NASDAQ",
    "NYQ": "NYSE", "PCX": "NYSE", "NYSEArca": "NYSE", "ASE": "NYSE",
}

def get_tv_ticker(symbol):
    """Return 'NASDAQ:SYMBOL' or 'NYSE:SYMBOL'; falls back to plain symbol on error."""
    try:
        fi = yf.Ticker(symbol).fast_info
        ex = fi.exchange if hasattr(fi, "exchange") else ""
        tv_ex = _TV_EXCHANGE_MAP.get(ex, "")
        return f"{tv_ex}:{symbol}" if tv_ex else symbol
    except Exception:
        return symbol

def save_benchmark_to_csv(close_series, name, directory):
    """Save benchmark close series to CSV in the benchmark data directory."""
    safe_name = name.replace("^", "")
    path = os.path.join(directory, f"{safe_name}.csv")
    close_series.to_frame(name="close").to_csv(path)
    print(f"    Benchmark saved → {path}")
    return path


def combine_fa_reports_to_pdf(html_paths, output_path):
    """Convert individual FA HTML reports to PDF and merge into one file."""
    if not html_paths:
        return None
    try:
        from weasyprint import HTML as WeasyprintHTML
        import pypdf
    except ImportError:
        print("    ⚠️  PDF combine skipped — run: !pip install weasyprint pypdf -q")
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_pdfs = []
    for hp in html_paths:
        pp = hp.replace(".html", "_tmp.pdf")
        try:
            WeasyprintHTML(filename=hp).write_pdf(pp)
            tmp_pdfs.append(pp)
        except Exception as e:
            print(f"    ⚠️  PDF convert failed for {os.path.basename(hp)}: {e}")

    if not tmp_pdfs:
        return None

    try:
        merger = pypdf.PdfMerger()
        for pp in tmp_pdfs:
            merger.append(pp)
        merger.write(output_path)
        merger.close()
        print(f"    ✅ Combined FA PDF ({len(tmp_pdfs)} reports) → {output_path}")
        return output_path
    except Exception as e:
        print(f"    ❌ PDF merge failed: {e}")
        return None
    finally:
        for pp in tmp_pdfs:
            try: os.remove(pp)
            except: pass


# ════════════════════════════════════════════════════════════
def run_fa_scan(matched, ticker_info, market_label):
    """Run FA scan for confirmed signals using Claude AI. Returns list of saved HTML report paths."""
    if not matched:
        print(f"    No confirmed signals — FA scan skipped.")
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
    fa_reports  = []
    fa_failures = []

    for r in matched:
        symbol   = r["symbol"]
        name     = r.get("name", symbol)
        price    = r["last_close"]
        is_sg    = r.get("yf_ticker", "").endswith(".SI")
        currency = "S$" if is_sg else "US$"
        market   = "Singapore (SGX)" if is_sg else "United States (S&P 500)"
        benchmarks = (
            "SGX-specific (EPS growth ≥0%, Revenue CAGR >10%, ROE >9%, etc.)"
            if is_sg else
            "US-specific (EPS growth ≥10%, Revenue CAGR >10%, ROE >15%, etc.)"
        )
        data_source = "SGX / Yahoo Finance" if is_sg else "SEC EDGAR / Yahoo Finance"

        print(f"    FA → {symbol:<8} {name[:32]:<32}", end=" ", flush=True)

        user_msg = (
            f"Please perform a full FA analysis for the stock ticker: {symbol}\n"
            f"Company: {name}\n"
            f"Market: {market}\n"
            f"Latest stock price (already downloaded from market data): {currency}{price}\n\n"
            f"IMPORTANT: Use {currency}{price} as the current stock price for all PE ratio "
            f"calculations. Do not re-fetch or re-download the price.\n\n"
            "Follow the framework exactly:\n"
            f"1. Apply {benchmarks}\n"
            f"2. Retrieve the latest financial data from {data_source}\n"
            "3. Score each metric against the correct market benchmarks\n"
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

    print(f"\n    {market_label} FA reports : {len(fa_reports)} saved to {FA_REPORTS_DIR}/")
    if fa_failures:
        print(f"    Failed : {', '.join(fa_failures)}")
    return fa_reports

# ██████████████  SGX SCAN  ██████████████
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("  SGX STOCK SCANNER")
print(f"  Filters: vol > {SGX_MIN_VOLUME:,}  |  price > S${SGX_MIN_PRICE}")
print("=" * 60)

# ── SGX Step 1: Fetch ticker list ────────────────────────────
print("\n[SGX 1/5] Fetching SGX stock list...")
resp = requests.get("https://api.sgx.com/securities/v1.1/",
                    headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
resp.raise_for_status()
stocks = [s for s in resp.json()["data"]["prices"] if s.get("type") == "stocks"]
sgx_ticker_info = {
    f"{s['nc']}.SI": {
        "symbol":  s["nc"],
        "name":    s.get("n", ""),
        "market":  s.get("m", ""),
        "sgx_pe":  s.get("pe"),
        "sgx_pb":  s.get("pb"),
        "sgx_yld": s.get("yld") or s.get("dy"),
        "sgx_eps": s.get("eps") or s.get("es"),
        "sgx_mc":  s.get("mc")  or s.get("mktcap"),
    }
    for s in stocks if s.get("nc")
}
print(f"    Found {len(sgx_ticker_info)} stocks")

# ── SGX Step 2: Download benchmark + price data ──────────────
end_date   = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
print(f"\n[SGX 2/5] Downloading data ({start_date} → {end_date})...")
print(f"    Benchmark: {SGX_BENCHMARK}  (TradingView: TVC:STI)")

sti_raw   = yf.download(SGX_BENCHMARK, start=start_date, end=end_date, auto_adjust=True, progress=False)
sti_close = sti_raw["Close"].squeeze()
sti_rsi14 = calc_rsi(sti_close)  # pre-compute benchmark RSI14
save_benchmark_to_csv(sti_close, "STI", BENCHMARK_DATA_DIR)
print(f"    TVC:STI RSI14 (latest): {float(sti_rsi14.iloc[-1]):.2f}")

all_sgx  = list(sgx_ticker_info.keys())
batches  = [all_sgx[i:i+BATCH_SIZE] for i in range(0, len(all_sgx), BATCH_SIZE)]
sgx_data = {}
for i, batch in enumerate(batches):
    print(f"    Batch {i+1}/{len(batches)} ...", end=" ", flush=True)
    try:
        raw = yf.download(batch, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=True)
        if len(batch) == 1:
            if not raw.empty: sgx_data[batch[0]] = raw.rename(columns=str.lower)
        else:
            for t in batch:
                try:
                    df = raw.xs(t, level=1, axis=1).dropna(how="all")
                    if not df.empty: sgx_data[t] = df.rename(columns=str.lower)
                except KeyError: pass
        print(f"ok ({sum(1 for t in batch if t in sgx_data)}/{len(batch)})")
    except Exception as e:
        print(f"ERROR: {e}")
    time.sleep(0.3)
print(f"    Total with data: {len(sgx_data)}")

# ── SGX Step 2b: Filter ──────────────────────────────────────
print(f"\n[SGX 2b/5] Filtering  vol > {SGX_MIN_VOLUME:,}  AND  price > S${SGX_MIN_PRICE} ...")
sgx_filtered = {t: df for t, df in sgx_data.items()
                if float(df["volume"].iloc[-1]) > SGX_MIN_VOLUME
                and float(df["close"].iloc[-1]) > SGX_MIN_PRICE}
print(f"    Before: {len(sgx_data)}  →  After: {len(sgx_filtered)}  ({len(sgx_data)-len(sgx_filtered)} removed)")

# ── SGX Step 3: Run scanners ─────────────────────────────────
print(f"\n[SGX 3/5] Running scanners on {len(sgx_filtered)} stocks...")
sgx_results = []
min_bars    = 110
idx_rsi_sgx = float(sti_rsi14.iloc[-1]) if pd.notna(sti_rsi14.iloc[-1]) else None

for idx, (yf_t, df) in enumerate(sgx_filtered.items()):
    if len(df) < min_bars: continue
    info = sgx_ticker_info[yf_t]
    try:
        df_sma = add_sma_gap_rs_buy_sell_signals(df, sti_close)
        df_bo  = add_breakout_buy_signals(df)
        df_obv = add_obv_signals(df)
        sma_sig      = bool(df_sma["buy_signal"].iloc[-1])
        bo_sig       = bool(df_bo["any_buy_signal"].iloc[-1])
        obv_sell_ban = bool(df_obv["sell_ban"].iloc[-1])
        confirmed    = (sma_sig and obv_sell_ban) or (bo_sig and obv_sell_ban)

        # RSI14 vs benchmark
        stock_rsi_s  = calc_rsi(df["close"])
        stock_rsi14  = round(float(stock_rsi_s.iloc[-1]), 2) if pd.notna(stock_rsi_s.iloc[-1]) else None
        rsi_ratio    = (round(stock_rsi14 / idx_rsi_sgx, 4)
                        if stock_rsi14 and idx_rsi_sgx and idx_rsi_sgx > 0 else None)

        sgx_results.append({
            "symbol": info["symbol"], "name": info["name"], "market": info["market"],
            "yf_ticker": yf_t,
            "signal_confirmed": confirmed, "sma_gap_buy": sma_sig,
            "breakout_buy": bo_sig, "obv_sell_ban": obv_sell_ban,
            "last_close": round(float(df["close"].iloc[-1]), 4),
            "last_date":  df.index[-1].strftime("%Y-%m-%d"), "bars": len(df),
            "rsi14":       stock_rsi14,
            "index_rsi14": round(idx_rsi_sgx, 2) if idx_rsi_sgx else None,
            "rsi_ratio":   rsi_ratio,
        })
    except Exception: pass
    if (idx+1) % 50 == 0:
        print(f"    {idx+1}/{len(sgx_filtered)} scanned | {sum(1 for r in sgx_results if r['signal_confirmed'])} confirmed")

sgx_matched = [r for r in sgx_results if r["signal_confirmed"]]
print(f"\n    Done: {len(sgx_results)} scanned, {len(sgx_matched)} confirmed signals")

# ── SGX Step 4: Export ───────────────────────────────────────
print(f"\n[SGX 4/5] Exporting results...")
fieldnames = [
    "symbol","name","market","signal_confirmed","sma_gap_buy","breakout_buy",
    "obv_sell_ban","last_close","last_date","bars","rsi14","index_rsi14","rsi_ratio"
]
with open(SGX_OUTPUT_RESULTS, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(sorted([{k:r[k] for k in fieldnames} for r in sgx_results],
                       key=lambda r: r["signal_confirmed"], reverse=True))
print(f"    Full results → {SGX_OUTPUT_RESULTS}")

with open(SGX_OUTPUT_WATCHLIST, "w", encoding="utf-8") as f:
    for r in sgx_matched: f.write(f"SGX:{r['symbol']}\n")
print(f"    TV watchlist → {SGX_OUTPUT_WATCHLIST}  ({len(sgx_matched)} tickers)")

# RSI comparison report for confirmed signals (companion to TV watchlist)
rsi_fields = ["tv_ticker","symbol","name","rsi14","index_rsi14","rsi_ratio","index_ticker","last_date"]
with open(SGX_RSI_COMPARISON, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rsi_fields)
    w.writeheader()
    for r in sgx_matched:
        w.writerow({
            "tv_ticker":    f"SGX:{r['symbol']}",
            "symbol":       r["symbol"],
            "name":         r["name"],
            "rsi14":        r["rsi14"],
            "index_rsi14":  r["index_rsi14"],
            "rsi_ratio":    r["rsi_ratio"],
            "index_ticker": "TVC:STI",
            "last_date":    r["last_date"],
        })
print(f"    RSI comparison → {SGX_RSI_COMPARISON}")

print("\n" + "=" * 60)
print(f"  SGX CONFIRMED SIGNALS ({len(sgx_matched)} stocks)")
print("=" * 60)
if sgx_matched:
    print(f"  {'Symbol':<8} {'Name':<28} {'Close':>7} {'SMA':^4} {'BO':^4} {'OBV':^4} {'RSI14':>6} {'IdxRSI':>7} {'Ratio':>6}")
    print(f"  {'-'*8} {'-'*28} {'-'*7} {'-'*4} {'-'*4} {'-'*4} {'-'*6} {'-'*7} {'-'*6}")
    for r in sgx_matched:
        rsi_str   = f"{r['rsi14']:.1f}" if r['rsi14']   else "  N/A"
        ratio_str = f"{r['rsi_ratio']:.3f}" if r['rsi_ratio'] else "  N/A"
        print(f"  {r['symbol']:<8} {r['name'][:28]:<28} S${r['last_close']:>5.3f} "
              f"{'Y' if r['sma_gap_buy'] else 'N':^4} "
              f"{'Y' if r['breakout_buy'] else 'N':^4} "
              f"{'Y' if r['obv_sell_ban'] else 'N':^4} "
              f"{rsi_str:>6} {r['index_rsi14'] or 'N/A':>7} {ratio_str:>6}")
else:
    print("  No confirmed signals today.")
print("=" * 60)

# ── SGX Step 5: FA scan ──────────────────────────────────────
print(f"\n[SGX 5/5] FA scan on {len(sgx_matched)} confirmed SGX stocks...")
print("    Note: price = last_close from downloaded batch data")
sgx_fa_reports = run_fa_scan(sgx_matched, sgx_ticker_info, "SGX")


# ════════════════════════════════════════════════════════════
# ██████████████  US SCAN (S&P 500)  ██████████████
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("  US STOCK SCANNER  (S&P 500)")
print(f"  Filters: vol > {US_MIN_VOLUME:,}  |  price > US${US_MIN_PRICE}")
print("=" * 60)

# ── US Step 1: Fetch S&P 500 ticker list ────────────────────
print("\n[US 1/5] Fetching S&P 500 stock list from Wikipedia...")
try:
    wiki_resp = requests.get(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"},
        timeout=30,
    )
    wiki_resp.raise_for_status()
    sp500_df = pd.read_html(wiki_resp.text)[0]
    us_ticker_info = {}
    for _, row in sp500_df.iterrows():
        raw_sym = str(row["Symbol"]).replace(".", "-")
        us_ticker_info[raw_sym] = {
            "symbol": raw_sym,
            "name":   str(row.get("Security", raw_sym)),
            "sector": str(row.get("GICS Sector", "")),
        }
    print(f"    Found {len(us_ticker_info)} stocks")
except Exception as e:
    print(f"    ERROR fetching S&P 500 list: {e}")
    us_ticker_info = {}

# ── US Step 2: Download benchmark + price data ───────────────
print(f"\n[US 2/5] Downloading data ({start_date} → {end_date})...")
print(f"    Benchmark: {US_BENCHMARK}  (TradingView: SPY)")

spy_raw   = yf.download(US_BENCHMARK, start=start_date, end=end_date, auto_adjust=True, progress=False)
spy_close = spy_raw["Close"].squeeze()
spy_rsi14 = calc_rsi(spy_close)  # pre-compute benchmark RSI14
save_benchmark_to_csv(spy_close, "SPY", BENCHMARK_DATA_DIR)
print(f"    SPY RSI14 (latest): {float(spy_rsi14.iloc[-1]):.2f}")

all_us  = list(us_ticker_info.keys())
batches = [all_us[i:i+BATCH_SIZE] for i in range(0, len(all_us), BATCH_SIZE)]
us_data = {}
for i, batch in enumerate(batches):
    print(f"    Batch {i+1}/{len(batches)} ...", end=" ", flush=True)
    try:
        raw = yf.download(batch, start=start_date, end=end_date, auto_adjust=True, progress=False, threads=True)
        if len(batch) == 1:
            if not raw.empty: us_data[batch[0]] = raw.rename(columns=str.lower)
        else:
            for t in batch:
                try:
                    df = raw.xs(t, level=1, axis=1).dropna(how="all")
                    if not df.empty: us_data[t] = df.rename(columns=str.lower)
                except KeyError: pass
        print(f"ok ({sum(1 for t in batch if t in us_data)}/{len(batch)})")
    except Exception as e:
        print(f"ERROR: {e}")
    time.sleep(0.3)
print(f"    Total with data: {len(us_data)}")

# ── US Step 2b: Filter ───────────────────────────────────────
print(f"\n[US 2b/5] Filtering  vol > {US_MIN_VOLUME:,}  AND  price > US${US_MIN_PRICE} ...")
us_filtered = {t: df for t, df in us_data.items()
               if float(df["volume"].iloc[-1]) > US_MIN_VOLUME
               and float(df["close"].iloc[-1]) > US_MIN_PRICE}
print(f"    Before: {len(us_data)}  →  After: {len(us_filtered)}  ({len(us_data)-len(us_filtered)} removed)")

# ── US Step 3: Run scanners ──────────────────────────────────
print(f"\n[US 3/5] Running scanners on {len(us_filtered)} stocks...")
us_results  = []
idx_rsi_us  = float(spy_rsi14.iloc[-1]) if pd.notna(spy_rsi14.iloc[-1]) else None

for idx, (sym, df) in enumerate(us_filtered.items()):
    if len(df) < min_bars: continue
    info = us_ticker_info[sym]
    try:
        df_sma = add_sma_gap_rs_buy_sell_signals(df, spy_close)
        df_bo  = add_breakout_buy_signals(df)
        df_obv = add_obv_signals(df)
        sma_sig      = bool(df_sma["buy_signal"].iloc[-1])
        bo_sig       = bool(df_bo["any_buy_signal"].iloc[-1])
        obv_sell_ban = bool(df_obv["sell_ban"].iloc[-1])
        confirmed    = (sma_sig and obv_sell_ban) or (bo_sig and obv_sell_ban)

        # RSI14 vs benchmark
        stock_rsi_s = calc_rsi(df["close"])
        stock_rsi14 = round(float(stock_rsi_s.iloc[-1]), 2) if pd.notna(stock_rsi_s.iloc[-1]) else None
        rsi_ratio   = (round(stock_rsi14 / idx_rsi_us, 4)
                       if stock_rsi14 and idx_rsi_us and idx_rsi_us > 0 else None)

        us_results.append({
            "symbol": info["symbol"], "name": info["name"], "market": "US",
            "yf_ticker": sym,
            "signal_confirmed": confirmed, "sma_gap_buy": sma_sig,
            "breakout_buy": bo_sig, "obv_sell_ban": obv_sell_ban,
            "last_close": round(float(df["close"].iloc[-1]), 4),
            "last_date":  df.index[-1].strftime("%Y-%m-%d"), "bars": len(df),
            "rsi14":       stock_rsi14,
            "index_rsi14": round(idx_rsi_us, 2) if idx_rsi_us else None,
            "rsi_ratio":   rsi_ratio,
        })
    except Exception: pass
    if (idx+1) % 50 == 0:
        print(f"    {idx+1}/{len(us_filtered)} scanned | {sum(1 for r in us_results if r['signal_confirmed'])} confirmed")

us_matched = [r for r in us_results if r["signal_confirmed"]]
print(f"\n    Done: {len(us_results)} scanned, {len(us_matched)} confirmed signals")

# ── US Step 4: Export ────────────────────────────────────────
print(f"\n[US 4/5] Exporting results...")
# Resolve correct TradingView exchange prefix (NASDAQ: vs NYSE:) for each confirmed stock
print(f"    Resolving TradingView exchange prefixes for {len(us_matched)} confirmed stocks...")
for r in us_matched:
    r["tv_ticker"] = get_tv_ticker(r["symbol"])
with open(US_OUTPUT_RESULTS, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(sorted([{k:r[k] for k in fieldnames} for r in us_results],
                       key=lambda r: r["signal_confirmed"], reverse=True))
print(f"    Full results → {US_OUTPUT_RESULTS}")

with open(US_OUTPUT_WATCHLIST, "w", encoding="utf-8") as f:
    for r in us_matched: f.write(f"{r['tv_ticker']}\n")
print(f"    TV watchlist → {US_OUTPUT_WATCHLIST}  ({len(us_matched)} tickers)")

with open(US_RSI_COMPARISON, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rsi_fields)
    w.writeheader()
    for r in us_matched:
        w.writerow({
            "tv_ticker":    r["tv_ticker"],
            "symbol":       r["symbol"],
            "name":         r["name"],
            "rsi14":        r["rsi14"],
            "index_rsi14":  r["index_rsi14"],
            "rsi_ratio":    r["rsi_ratio"],
            "index_ticker": "SPY",
            "last_date":    r["last_date"],
        })
print(f"    RSI comparison → {US_RSI_COMPARISON}")

print("\n" + "=" * 60)
print(f"  US CONFIRMED SIGNALS ({len(us_matched)} stocks)")
print("=" * 60)
if us_matched:
    print(f"  {'Symbol':<8} {'Name':<28} {'Close':>8} {'SMA':^4} {'BO':^4} {'OBV':^4} {'RSI14':>6} {'IdxRSI':>7} {'Ratio':>6}")
    print(f"  {'-'*8} {'-'*28} {'-'*8} {'-'*4} {'-'*4} {'-'*4} {'-'*6} {'-'*7} {'-'*6}")
    for r in us_matched:
        rsi_str   = f"{r['rsi14']:.1f}" if r['rsi14']   else "  N/A"
        ratio_str = f"{r['rsi_ratio']:.3f}" if r['rsi_ratio'] else "  N/A"
        print(f"  {r['symbol']:<8} {r['name'][:28]:<28} US${r['last_close']:>6.2f} "
              f"{'Y' if r['sma_gap_buy'] else 'N':^4} "
              f"{'Y' if r['breakout_buy'] else 'N':^4} "
              f"{'Y' if r['obv_sell_ban'] else 'N':^4} "
              f"{rsi_str:>6} {r['index_rsi14'] or 'N/A':>7} {ratio_str:>6}")
else:
    print("  No confirmed signals today.")
print("=" * 60)

# ── US Step 5: FA scan ───────────────────────────────────────
print(f"\n[US 5/5] FA scan on {len(us_matched)} confirmed US stocks...")
print("    Note: price = last_close from downloaded batch data")
us_fa_reports = run_fa_scan(us_matched, us_ticker_info, "US")


# ════════════════════════════════════════════════════════════
# COMBINE FA REPORTS INTO ONE PDF
# ════════════════════════════════════════════════════════════
all_fa_reports = sgx_fa_reports + us_fa_reports
if all_fa_reports:
    print(f"\n[PDF] Combining {len(all_fa_reports)} FA reports into single PDF...")
    combine_fa_reports_to_pdf(all_fa_reports, COMBINED_FA_PDF)
else:
    print("\n[PDF] No FA reports to combine.")


# ════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("  ALL DONE")
print("=" * 60)
print(f"  All files saved to: {GDRIVE_BASE}/")
print()
print(f"  SGX results      : {SGX_OUTPUT_RESULTS}")
print(f"  SGX watchlist    : {SGX_OUTPUT_WATCHLIST}  ({len(sgx_matched)} signals)")
print(f"  SGX RSI report   : {SGX_RSI_COMPARISON}  (TVC:STI benchmark)")
print(f"  US  results      : {US_OUTPUT_RESULTS}")
print(f"  US  watchlist    : {US_OUTPUT_WATCHLIST}  ({len(us_matched)} signals)")
print(f"  US  RSI report   : {US_RSI_COMPARISON}  (SPY benchmark)")
print(f"  Benchmarks       : {BENCHMARK_DATA_DIR}/STI.csv  |  SPY.csv")
print(f"  FA  reports (HTML): {FA_REPORTS_DIR}/  ({len(all_fa_reports)} total)")
print(f"  FA  combined PDF : {COMBINED_FA_PDF}")
print("=" * 60)
print("\nImport into TradingView:")
print("  Watchlist → ⋮ → Import watchlist → sgx_watchlist.txt / us_watchlist.txt")
print("\nRSI ratio columns: stock RSI14 / index RSI14")
print("  ratio > 1.0 → stock is relatively stronger than the index")
print("  ratio < 1.0 → stock is relatively weaker  than the index")
