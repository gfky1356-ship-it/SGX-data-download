# ============================================================
# SGX — 1-YEAR HISTORICAL DATA DOWNLOADER
# Colab Ready
#
# Sources:
#   Ticker list : api.sgx.com  (SGX official API)
#   Price data  : Yahoo Finance (yfinance, .SI suffix)
#
# Output: /content/sgx_historical.csv
# ============================================================

# ── Cell 1: Install ─────────────────────────────────────────
# !pip install yfinance requests -q


# ── Cell 2: Download ────────────────────────────────────────
import csv, time, warnings
import requests
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

OUTPUT_FILE = "/content/sgx_historical.csv"
BATCH_SIZE  = 50


# ── Step 1: Fetch SGX stock list ─────────────────────────────
print("=" * 55)
print("  SGX HISTORICAL DATA DOWNLOADER")
print("=" * 55)
print("\n[1/3] Fetching SGX stock list...")

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


# ── Step 2: Download 1-year history in batches ───────────────
end_date   = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
print(f"\n[2/3] Downloading data ({start_date} → {end_date})...")

all_tickers = list(ticker_info.keys())
batches     = [all_tickers[i:i+BATCH_SIZE] for i in range(0, len(all_tickers), BATCH_SIZE)]
rows_written    = 0
tickers_found   = 0
tickers_missing = []

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["symbol", "name", "market", "date",
                     "open", "high", "low", "close", "volume"])

    for i, batch in enumerate(batches):
        print(f"    Batch {i+1}/{len(batches)} ({len(batch)} tickers)...", end=" ", flush=True)
        try:
            raw = yf.download(batch, start=start_date, end=end_date,
                              auto_adjust=True, progress=False, threads=True)
            if len(batch) == 1:
                results = {batch[0]: raw} if not raw.empty else {}
            else:
                results = {}
                for t in batch:
                    try:
                        df = raw.xs(t, level=1, axis=1).dropna(how="all")
                        if not df.empty:
                            results[t] = df
                    except KeyError:
                        pass

            for yf_t, df in results.items():
                info = ticker_info[yf_t]
                for date, row in df.iterrows():
                    writer.writerow([
                        info["symbol"], info["name"], info["market"],
                        date.strftime("%Y-%m-%d"),
                        round(float(row.get("Open",  0) or 0), 4),
                        round(float(row.get("High",  0) or 0), 4),
                        round(float(row.get("Low",   0) or 0), 4),
                        round(float(row.get("Close", 0) or 0), 4),
                        int(row.get("Volume", 0) or 0),
                    ])
                    rows_written += 1
                tickers_found += 1

            tickers_missing += [t for t in batch if t not in results]
            print(f"ok ({len(results)}/{len(batch)})")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.3)


# ── Step 3: Summary ──────────────────────────────────────────
print(f"\n[3/3] Done!")
print(f"    Tickers with data : {tickers_found}")
print(f"    Tickers no data   : {len(tickers_missing)}")
print(f"    Total rows        : {rows_written:,}")
print(f"    Saved to          : {OUTPUT_FILE}")
if tickers_missing:
    syms = [ticker_info[t]["symbol"] for t in tickers_missing]
    print(f"    Missing           : {', '.join(syms[:15])}", end="")
    print(f" +{len(syms)-15} more" if len(syms) > 15 else "")
