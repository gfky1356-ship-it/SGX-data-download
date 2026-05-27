# ============================================================
# SGX — MOOMOO API HISTORICAL DATA DOWNLOADER
# Colab Ready
#
# Data source : Moomoo OpenAPI (free for Moomoo account holders)
#
# Requirements:
#   1. Moomoo desktop app with OpenD gateway running on your PC/Mac
#   2. For Colab: expose OpenD port via ngrok  (see SETUP below)
#   3. pip install moomoo-api -q
#
# Output: /content/sgx_historical.csv  +  /content/sti_close.csv
#   (same format as the yfinance downloader — all scanner cells work unchanged)
#
# ── COLAB SETUP ──────────────────────────────────────────────
#  On your PC/Mac (where Moomoo desktop is installed):
#    1. Open Moomoo → Settings → OpenAPI → Enable OpenD
#       (default port 11111, no password needed for local use)
#    2. Install ngrok: https://ngrok.com/download
#    3. Run in terminal:  ngrok tcp 11111
#    4. Copy the forwarding address, e.g.  tcp://0.tcp.ap.ngrok.io:12345
#    5. Set OPEND_HOST = "0.tcp.ap.ngrok.io"  and  OPEND_PORT = 12345  below
#
#  If running on the SAME machine as OpenD (not Colab):
#    Keep OPEND_HOST = "127.0.0.1"  and  OPEND_PORT = 11111
# ============================================================

# ── Colab install cell (paste into a Colab cell and run once) ─
# !pip install moomoo-api requests -q

import csv
import time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── CONFIG — edit these before running ───────────────────────
OPEND_HOST  = "127.0.0.1"          # ngrok host for Colab, e.g. "0.tcp.ap.ngrok.io"
OPEND_PORT  = 11111                 # ngrok port for Colab, e.g. 12345
SGX_HIST_CSV = "/content/sgx_historical.csv"
STI_CSV      = "/content/sti_close.csv"
STI_CODE     = "SG.ES3"            # SPDR STI ETF — closest tradeable STI proxy on Moomoo
DAYS_BACK    = 365
# ─────────────────────────────────────────────────────────────


def _date_range():
    end   = datetime.today()
    start = end - timedelta(days=DAYS_BACK)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def download_sgx_with_moomoo(
    opend_host=OPEND_HOST,
    opend_port=OPEND_PORT,
    sgx_hist_csv=SGX_HIST_CSV,
    sti_csv=STI_CSV,
):
    """
    Download SGX daily OHLCV data via Moomoo OpenAPI and save to CSV.
    Returns (sgx_stock_data, sgx_ticker_info, sti_close) dicts/series,
    matching the same variable names used by the scanner cells.
    """
    try:
        from moomoo import (
            OpenQuoteContext, Market, SecurityType,
            KLType, AuType, RET_OK,
        )
    except ImportError:
        raise ImportError(
            "moomoo-api is not installed. Run:  !pip install moomoo-api -q"
        )

    import pandas as pd

    start_date, end_date = _date_range()

    print("=" * 55)
    print("  SGX DATA DOWNLOAD  (Moomoo OpenAPI)")
    print("=" * 55)
    print(f"  OpenD  : {opend_host}:{opend_port}")
    print(f"  Period : {start_date}  →  {end_date}")

    # ── Connect ───────────────────────────────────────────────
    print("\n[1/4] Connecting to OpenD gateway...")
    try:
        quote_ctx = OpenQuoteContext(host=opend_host, port=opend_port)
    except Exception as e:
        raise ConnectionError(
            f"Cannot connect to OpenD at {opend_host}:{opend_port}\n"
            f"  • Is Moomoo desktop running with OpenD enabled?\n"
            f"  • For Colab: is ngrok tunnelling port 11111?\n"
            f"  Original error: {e}"
        )
    print("    Connected.")

    # ── Get SGX stock list ────────────────────────────────────
    print("\n[2/4] Fetching SGX stock list from Moomoo...")
    ret, data = quote_ctx.get_stock_basicinfo(Market.SG, SecurityType.STOCK)
    if ret != RET_OK:
        quote_ctx.close()
        raise RuntimeError(f"get_stock_basicinfo failed: {data}")

    stocks = data[data["stock_type"] == "STOCK"].copy() if "stock_type" in data.columns else data.copy()
    print(f"    Found {len(stocks)} SGX stocks")

    # Build ticker map:  moomoo_code → {symbol, name, market}
    ticker_map = {}
    for _, row in stocks.iterrows():
        code = str(row.get("code", ""))          # e.g. "SG.D05"
        sym  = code.replace("SG.", "").strip()   # e.g. "D05"
        if not sym:
            continue
        ticker_map[code] = {
            "symbol": sym,
            "name":   str(row.get("name", "")),
            "market": "Mainboard",               # Moomoo doesn't distinguish board; set default
        }

    # ── Download historical K-lines ───────────────────────────
    print(f"\n[3/4] Downloading 1-year daily history for {len(ticker_map)} stocks...")

    sgx_stock_data  = {}
    sgx_ticker_info = {}
    rows_written    = 0
    missing         = []

    with open(sgx_hist_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "name", "market", "date",
                         "open", "high", "low", "close", "volume"])

        codes = list(ticker_map.keys())
        for i, code in enumerate(codes):
            info = ticker_map[code]
            if (i + 1) % 50 == 0 or i == 0:
                print(f"    {i+1}/{len(codes)}  ({code})...")

            try:
                ret, kdata, _ = quote_ctx.request_history_kline(
                    code,
                    start=start_date,
                    end=end_date,
                    ktype=KLType.K_DAY,
                    autype=AuType.QFQ,   # forward-adjusted (same as yfinance auto_adjust)
                    max_count=1000,
                )
                if ret != RET_OK or kdata is None or kdata.empty:
                    missing.append(code)
                    time.sleep(0.05)
                    continue

                kdata["time_key"] = pd.to_datetime(kdata["time_key"])
                kdata = kdata.sort_values("time_key")

                df_rows = []
                for _, row in kdata.iterrows():
                    date_str = row["time_key"].strftime("%Y-%m-%d")
                    writer.writerow([
                        info["symbol"], info["name"], info["market"],
                        date_str,
                        round(float(row.get("open",   0) or 0), 4),
                        round(float(row.get("high",   0) or 0), 4),
                        round(float(row.get("low",    0) or 0), 4),
                        round(float(row.get("close",  0) or 0), 4),
                        int(row.get("volume", 0) or 0),
                    ])
                    df_rows.append({
                        "date":   row["time_key"],
                        "open":   float(row.get("open",   0) or 0),
                        "high":   float(row.get("high",   0) or 0),
                        "low":    float(row.get("low",    0) or 0),
                        "close":  float(row.get("close",  0) or 0),
                        "volume": int(row.get("volume", 0) or 0),
                    })
                    rows_written += 1

                if df_rows:
                    df = pd.DataFrame(df_rows).set_index("date").sort_index()
                    yf_key = f"{info['symbol']}.SI"
                    sgx_stock_data[yf_key]  = df
                    sgx_ticker_info[yf_key] = info

            except Exception as e:
                print(f"    ERROR {code}: {e}")
                missing.append(code)

            time.sleep(0.08)   # stay within Moomoo rate limits (~30 req/s allowed)

    # ── STI benchmark (SPDR STI ETF as proxy) ────────────────
    print(f"\n[4/4] Fetching STI benchmark ({STI_CODE})...")
    sti_close = None
    try:
        ret, sti_kdata, _ = quote_ctx.request_history_kline(
            STI_CODE,
            start=start_date,
            end=end_date,
            ktype=KLType.K_DAY,
            autype=AuType.QFQ,
            max_count=1000,
        )
        if ret == RET_OK and sti_kdata is not None and not sti_kdata.empty:
            sti_kdata["time_key"] = pd.to_datetime(sti_kdata["time_key"])
            sti_close = sti_kdata.set_index("time_key")["close"].sort_index()
            sti_close.index.name = "Date"
            sti_close.name = "Close"
            sti_close.to_csv(sti_csv)
            print(f"    STI benchmark saved: {len(sti_close)} rows → {sti_csv}")
        else:
            print(f"    WARNING: Could not fetch STI benchmark from Moomoo ({ret}: {sti_kdata})")
            print("    Falling back to yfinance for STI...")
            sti_close = _sti_fallback_yfinance(start_date, end_date, sti_csv)
    except Exception as e:
        print(f"    WARNING: STI fetch error: {e}")
        print("    Falling back to yfinance for STI...")
        sti_close = _sti_fallback_yfinance(start_date, end_date, sti_csv)

    quote_ctx.close()

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  Done!")
    print(f"  Stocks with data  : {len(sgx_stock_data)}")
    print(f"  Stocks no data    : {len(missing)}")
    print(f"  Total rows        : {rows_written:,}")
    print(f"  SGX CSV           : {sgx_hist_csv}")
    print(f"  STI CSV           : {sti_csv}")
    print(f"{'='*55}")

    return sgx_stock_data, sgx_ticker_info, sti_close


def _sti_fallback_yfinance(start_date, end_date, sti_csv):
    """Fetch STI benchmark from yfinance as a fallback."""
    try:
        import yfinance as yf
        import pandas as pd
        sti_raw = yf.download("^STI", start=start_date, end=end_date,
                              auto_adjust=True, progress=False)
        if not sti_raw.empty:
            sti_close = sti_raw["Close"].squeeze()
            sti_close.to_csv(sti_csv)
            print(f"    STI (yfinance fallback) saved: {len(sti_close)} rows → {sti_csv}")
            return sti_close
    except Exception as e:
        print(f"    yfinance STI fallback also failed: {e}")
    return None


# ── Run as standalone script ──────────────────────────────────
if __name__ == "__main__":
    sgx_stock_data, sgx_ticker_info, sti_close = download_sgx_with_moomoo()
