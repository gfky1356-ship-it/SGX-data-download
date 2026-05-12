# SGX + US Stock Scanner v7.0
## AI-Powered Technical & Fundamental Analysis Scanner

> **Automated dual-market stock scanner for Singapore (SGX) and US (S&P 500) with technical signal confirmation, fundamental analysis, and TradingView integration.**

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Scan Specifications](#scan-specifications)
- [Output Files](#output-files)
- [Technical Indicators](#technical-indicators)
- [Fundamental Analysis (FA) Matrix](#fundamental-analysis-fa-matrix)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Usage](#usage)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Troubleshooting](#troubleshooting)

---

## Overview

This is a **fully self-contained Python script** that:

1. **Scans thousands of stocks** across two markets (SGX + S&P 500)
2. **Applies three independent technical scanners** (SMA Gap, Breakout, OBV)
3. **Confirms signals** using multi-scanner consensus
4. **Computes 14-period RSI** vs market benchmarks (STI for SG, SPY for US)
5. **Runs fundamental analysis** (FA) on confirmed signals using GLOBAL Screener v6.0
6. **Generates professional HTML reports** with verdict badges (🔥 FIRE / ⏳ WAIT / 🚫 AVOID)
7. **Combines reports into a single PDF** for easy sharing
8. **Exports TradingView watchlists** for manual verification
9. **Saves to Google Drive** (if running in Colab, otherwise local `/content`)

**Language:** Chinese UI + International ticker support  
**Version:** v7.0 (Latest)  
**Status:** Production-ready for Google Colab

---

## Quick Start

### Option 1: Google Colab (Recommended)

```bash
# Private repo (replace YOUR_TOKEN with GitHub Personal Access Token):
!git clone -q https://YOUR_TOKEN@github.com/gfky1356-ship-it/sgx-data-download.git \
  && pip install yfinance requests weasyprint pypdf -q
!python sgx-data-download/stock_scanner_colab.py

# Public repo (no token needed):
!wget -qO stock_scanner_colab.py https://raw.githubusercontent.com/gfky1356-ship-it/sgx-data-download/main/stock_scanner_colab.py \
  && pip install yfinance requests weasyprint pypdf -q
!python stock_scanner_colab.py
```

### Option 2: Local Environment

```bash
# Clone repo
git clone https://github.com/gfky1356-ship-it/sgx-data-download.git
cd sgx-data-download

# Install dependencies
pip install yfinance requests weasyprint pypdf pandas numpy

# Run scanner
python stock_scanner_colab.py
```

### Option 3: One-Line Run (Public Repo)

```bash
wget -qO scanner.py https://raw.githubusercontent.com/gfky1356-ship-it/sgx-data-download/main/stock_scanner_colab.py && \
pip install yfinance requests weasyprint pypdf pandas numpy -q && \
python scanner.py
```

---

## Scan Specifications

### SGX (Singapore Exchange)

| Parameter | Value |
|-----------|-------|
| **Benchmark** | ^STI (Straits Times Index) |
| **TradingView Ticker** | TVC:STI |
| **Min Volume** | 100,000 shares/day |
| **Min Price** | S$0.20 |
| **Lookback Period** | 1 year (365 days) |
| **Output Watchlist** | `sgx_watchlist.txt` (format: `SGX:SYMBOL`) |

**Example confirmed signal:**
```
SGX:J36 @ S$3.240 | SMA Gap: Y | Breakout: N | OBV Sell Ban: Y | RSI14: 65.2 | Index RSI14: 42.1 | Ratio: 1.548
```

### US Market (S&P 500)

| Parameter | Value |
|-----------|-------|
| **Benchmark** | SPY (S&P 500 ETF) |
| **TradingView Ticker** | SPY (native) |
| **Min Volume** | 200,000 shares/day |
| **Min Price** | US$10.00 |
| **Lookback Period** | 1 year (365 days) |
| **Output Watchlist** | `us_watchlist.txt` (format: `NYSE:SYMBOL`) |

**Example confirmed signal:**
```
AAPL @ US$195.23 | SMA Gap: Y | Breakout: Y | OBV Sell Ban: Y | RSI14: 72.1 | Index RSI14: 58.3 | Ratio: 1.237
```

---

## Output Files

All outputs are saved to **Google Drive** (if Colab) or local directory:

```
StockScanner/
├── sgx_scan_results.csv              # Full SGX results (all scanned stocks)
├── sgx_watchlist.txt                 # Confirmed SGX signals (TradingView format)
├── sgx_rsi_comparison.csv            # SGX RSI14 vs index (STI)
├── us_scan_results.csv               # Full US results
├── us_watchlist.txt                  # Confirmed US signals (TradingView format)
├── us_rsi_comparison.csv             # US RSI14 vs index (SPY)
├── benchmark_data/
│   ├── STI.csv                       # STI close prices (1 year)
│   └── SPY.csv                       # SPY close prices (1 year)
├── fa_reports/
│   ├── J36_FA_v6.0_20260512.html    # Individual FA report (HTML)
│   ├── AAPL_FA_v6.0_20260512.html   # ...
│   ├── ...
│   └── ALL_FA_REPORTS_20260512.pdf  # Combined PDF (all reports merged)
```

### CSV Column Descriptions

#### `sgx_scan_results.csv` / `us_scan_results.csv`

| Column | Type | Description |
|--------|------|-------------|
| `symbol` | str | Stock ticker symbol |
| `name` | str | Company name |
| `market` | str | "SG" or "US" |
| `signal_confirmed` | bool | Multi-scanner consensus buy signal |
| `sma_gap_buy` | bool | SMA Gap Enlarge Buy signal triggered |
| `breakout_buy` | bool | Breakout Buy signal triggered |
| `obv_sell_ban` | bool | OBV showing accumulation phase (sell ban active) |
| `last_close` | float | Latest closing price |
| `last_date` | str | Date of latest data (YYYY-MM-DD) |
| `bars` | int | Number of trading days in history |
| `rsi14` | float | 14-period RSI for stock |
| `index_rsi14` | float | 14-period RSI for benchmark (STI/SPY) |
| `rsi_ratio` | float | stock_rsi14 / index_rsi14 |

---

## Technical Indicators

### 1. SMA Gap Enlarge Buy (Scanner A)

**Purpose:** Identify stocks where momentum is accelerating relative to the market benchmark.

**Logic:**
```
✓ 20-SMA slope > 50-SMA slope (bullish trend)
✓ 2 consecutive days of positive relative strength vs index
✓ Price > 20-SMA (above moving average)
✓ Green candle on signal day
→ BUY SIGNAL
```

**Parameters:**
- SMA lengths: 20, 50 days
- Slope lookback: 5 days
- Relative strength SMA: 30 days
- Delta MA smoothing: 9 days

---

### 2. Breakout Buy (Scanner B)

**Purpose:** Catch stocks breaking out of recent price consolidation with uptrend confirmation.

**Logic:**
```
✓ 200-SMA has positive slope (long-term uptrend)
✓ 20-SMA > 50-SMA (intermediate uptrend)
✓ Price > 20-SMA (above intermediate support)
✓ Green candle on signal day
✓ Price breaks above 30/60/90-day high (triple confirm)
✓ Pullback to 20-SMA happened before breakout
✓ Price within 5% above breakout level
→ BUY SIGNAL
```

**Parameters:**
- Lookback periods: 30, 60, 90 days (n1, n2, n3)
- 200-SMA slope lookback: 5 days
- Max upside after breakout: 5%

---

### 3. OBV (On-Balance Volume) Signal (Scanner C)

**Purpose:** Confirm accumulation phase (institutional buying without resistance).

**Logic:**
```
OBV = cumulative(volume × sign(close - prior_close))

✓ OBV > 20-SMA (short-term strength)
✓ OBV > 100-SMA (long-term confirmation)
→ SELL BAN ACTIVE (allow buy signals)

✓ OBV < 20-SMA (short-term weakness)
✓ OBV < 100-SMA (long-term weakness)
→ BUY BAN ACTIVE (suppress false signals)

Signal generated when OBV crosses above/below both SMAs
with 1+ day confirmation
```

**Parameters:**
- Short SMA: 20 days
- Long SMA: 100 days
- Confirmation period: 1 day minimum

---

### Signal Confirmation

**Buy Signal is CONFIRMED when:**
```
(SMA Gap Signal AND OBV Sell Ban Active) 
    OR 
(Breakout Signal AND OBV Sell Ban Active)
```

This multi-scanner approach reduces false positives significantly.

---

## Fundamental Analysis (FA) Matrix

### GLOBAL Screener v6.0 (Chinese UI)

Runs on all confirmed signals. Outputs HTML report with 10-point matrix:

| # | Metric (Chinese) | Metric (English) | SG Threshold | US Threshold | Status |
|---|---|---|---|---|---|
| 1 | EPS 同比增长 (%) | EPS Growth (%) | ≥ 0% | ≥ 10% | ✅/⚠️/❌ |
| 2 | 营收增速 / CAGR (%) | Revenue Growth / CAGR (%) | > 10% | > 10% | ✅/⚠️/❌ |
| 3 | ROE (TTM) (%) | Return on Equity (%) | > 9% | > 15% | ✅/⚠️/❌ |
| 4 | ROIC (%) | Return on Invested Capital (%) | ≥ 15% | ≥ 15% | ✅/⚠️/❌ |
| 5 | P/FCF (倍数) | Price / Free Cash Flow | < 35x | < 25x | ✅/⚠️/❌ |
| 6 | TTM 市盈率 | Trailing P/E | — | — | (Reference) |
| 7 | 前瞻市盈率 | Forward P/E | — | — | (Reference) |
| 8 | 净利润率 (%) | Net Profit Margin (%) | > 5% | > 9.5% | ✅/⚠️/❌ |
| 9 | 债务权益比 | Debt to Equity Ratio | < 1.0x | < 1.0x | ✅/⚠️/❌ |
| 10 | Rule of 40 (%) | Growth + Margin (%) | > 40% | > 40% | ✅/⚠️/❌ |

### Verdict Logic

| Scenario | Verdict | Badge | Color |
|----------|---------|-------|-------|
| 0 ❌ failures | 🔥 **FIRE** | Buy strongly | Green |
| 1-2 ❌ failures | ⏳ **WAIT** | Hold & watch | Amber |
| 3+ ❌ failures | 🚫 **AVOID** | Do not buy | Red |

**Warning flags:**
- ⚠️ **Edge**: Metric passes at threshold but fails at 90% level (fragile)
- ❌ **Fail**: Below threshold
- [NO DATA]: Missing data point

### Special Handling

**Banks & Financial Institutions:**
- P/FCF set to [NO DATA] (not applicable)
- ROIC set to [NO DATA] (banking models differ)
- All other metrics still evaluated

---

## Data Sources

### Primary Sources (In Priority Order)

| Data | Source 1 | Source 2 | Source 3 |
|------|----------|----------|----------|
| **Prices (OHLCV)** | yfinance | — | — |
| **Fundamentals** | Yahoo Finance | SGX API | Manual entry |
| **P/E, P/B, Yield** | Yahoo Finance | SGX API | — |
| **EPS, Growth** | Yahoo Finance | SEC EDGAR | — |
| **Sector/Industry** | Yahoo Finance | — | — |

### Price Data Currency

- **SGX**: Singapore Dollar (S$) – use `.SI` suffix in yfinance
- **US**: US Dollar (US$) – native ticker

### Benchmark Data

- **STI**: `^STI` ticker in yfinance → TradingView: `TVC:STI`
- **SPY**: `SPY` ticker in yfinance → TradingView: `SPY`

Both saved as CSV for reuse (enables `data_adapter.py` integration).

---

## Installation

### Requirements

- **Python 3.7+**
- **Internet connection** (API calls to yfinance, SGX, Wikipedia)
- **Google Colab** (recommended) OR local environment

### Dependencies

```
yfinance       # Stock price & fundamental data
requests       # API calls & web scraping
pandas         # Data manipulation
numpy          # Numerical operations
weasyprint     # HTML → PDF conversion
pypdf          # PDF merging
```

### Install via pip

```bash
pip install yfinance requests weasyprint pypdf pandas numpy
```

### Google Colab Setup

```python
!pip install yfinance requests weasyprint pypdf pandas numpy -q
```

---

## Usage

### Google Colab (Step by Step)

1. **Open Google Colab:** https://colab.research.google.com
2. **Paste one-liner:**
   ```python
   !wget -qO stock_scanner_colab.py https://raw.githubusercontent.com/gfky1356-ship-it/sgx-data-download/main/stock_scanner_colab.py && pip install yfinance requests weasyprint pypdf -q && python stock_scanner_colab.py
   ```
3. **Run cell** (Ctrl+Enter)
4. **Wait for completion** (10-30 minutes depending on market size)
5. **Check Google Drive:**
   - Navigate to `MyDrive → StockScanner/`
   - Download CSV files & PDF reports

### Local Environment

```bash
python stock_scanner_colab.py
```

Outputs saved to `/content/` (unless script modified).

### Programmatic Usage (Advanced)

```python
from data_adapter import DataAdapter
from stock_scanner_colab import GlobalScreenerV59_CN

# Use downloaded data via data_adapter
adapter = DataAdapter(
    source=DataAdapter.SOURCE_CSV,
    data_dir="/path/to/benchmark_data"
)
spy_close = adapter.load_benchmark("US")

# Run custom FA scan
fa_data = {
    "eps_growth": 12.5,
    "rev_cagr": 15.0,
    "roe": 18.5,
    # ... other fields
}
screener = GlobalScreenerV59_CN("AAPL", 195.23, fa_data)
html = screener.generate_html()
print(screener.get_verdict())  # ("fire", "🔥 FIRE — 符合买入标准", "fire")
```

---

## Architecture

### Module Structure

```
stock_scanner_colab.py (self-contained, ~1600 lines)
├── Setup & Configuration
│   ├── Google Drive mounting (Colab-aware)
│   ├── Output path definitions
│   ├── Scan filter parameters
│   └── Benchmark ticker definitions
│
├── GLOBAL Screener v6.0 (FA Engine)
│   ├── GlobalScreenerV59_CN class
│   ├── 10-point financial matrix
│   ├── HTML report generation
│   ├── Verdict logic (FIRE/WAIT/AVOID)
│   └── Chinese UI rendering
│
├── Scanner Functions (Shared)
│   ├── calc_rsi() — Wilder's smoothed RSI14
│   ├── add_sma_gap_rs_buy_sell_signals() — Scanner A
│   ├── add_breakout_buy_signals() — Scanner B
│   ├── add_obv_signals() — Scanner C
│   └── Helper functions
│
├── SGX Scan Pipeline (5 steps)
│   ├── Step 1: Fetch SGX stock list (SGX API)
│   ├── Step 2: Download OHLCV + benchmark (yfinance)
│   ├── Step 2b: Filter by volume & price
│   ├── Step 3: Run 3 scanners on filtered stocks
│   ├── Step 4: Export CSV + watchlist + RSI report
│   └── Step 5: Run FA scan on confirmed signals
│
├── US Scan Pipeline (5 steps)
│   ├── Step 1: Fetch S&P 500 list (Wikipedia)
│   ├── Step 2-5: (Identical to SGX)
│
├── PDF Generation
│   ├── combine_fa_reports_to_pdf() — weasyprint + pypdf
│   └── PDF cleanup
│
└── Summary Output
    └── Console report + file locations
```

### Data Flow

```
[SGX API] → [yfinance] → [OHLCV DataFrames]
                              ↓
                    [Filter by vol/price]
                              ↓
                    [Run 3 Scanners in parallel]
                              ↓
                    [Multi-scanner consensus]
                              ↓
        [Confirmed signals] → [FA Screener] → [HTML Report]
                ↓                                    ↓
            [CSV export]                        [PDF merge]
            [Watchlist]
            [RSI report]
```

---

## Key Features

### ✅ Market Coverage

- **SGX:** All stocks + STI benchmark (1,400+ stocks)
- **US:** S&P 500 (500 stocks) with SPY benchmark
- **Custom:** Easily add new markets by extending scan pipelines

### ✅ Technical Analysis

- **3 independent scanners** (SMA, Breakout, OBV) reduce false signals
- **14-period RSI** with benchmark comparison (ratio-based, not absolute)
- **Multi-timeframe analysis** (20/50/200-day SMAs)
- **Volume confirmation** (OBV accumulation phase detection)

### ✅ Fundamental Analysis

- **10-point FA matrix** with market-specific thresholds (SG vs US)
- **Automated data fetching** from Yahoo Finance + SGX API
- **Intelligent fallbacks** when data missing
- **Bank-aware adjustments** (P/FCF & ROIC excluded for financial institutions)
- **Rule of 40** screening (Growth + Margin ≥ 40%)

### ✅ Reporting

- **HTML FA reports** with professional styling (light theme, monospace data)
- **PDF combination** – single consolidated PDF for all screened stocks
- **TradingView watchlists** – direct import format (SGX:SYMBOL or NYSE:SYMBOL)
- **RSI comparison CSV** – benchmark ratio tracking
- **Console summaries** – real-time progress + signal counts

### ✅ Integration

- **Google Drive support** – outputs auto-saved to Drive (if Colab)
- **data_adapter.py compatible** – modular future expansion
- **CSV exports** – easy downstream processing
- **TradingView integration** – watchlist import ready
- **Extensible** – add new scanners, markets, or FA rules

### ✅ Production Hardening

- **Batch processing** – yfinance calls in groups of 50 to avoid rate limits
- **Error handling** – graceful fallbacks for API failures
- **Caching** – benchmark data saved locally (RSI reuse)
- **Timeout management** – configurable sleep between requests
- **Empty data handling** – skips stocks with insufficient history

---

## Troubleshooting

### Q: "No data available from yfinance"
**A:** Check internet connection + ensure yfinance library is up-to-date:
```bash
pip install --upgrade yfinance
```

### Q: "Google Drive mounting failed"
**A:** Script falls back to `/content`. Files still saved locally in Colab.

### Q: "PDF combine skipped — run: !pip install weasyprint pypdf"
**A:** Install missing dependencies:
```bash
!pip install weasyprint pypdf -q
```

### Q: "Confirmed signals = 0"
**A:** This is normal! Multi-scanner consensus is strict:
- Need BOTH (SMA OR Breakout) AND OBV Sell Ban
- Only high-confidence signals pass
- Try lowering min volume filters in config section
- Check that scan date ≠ market holidays

### Q: "AttributeError: 'DataFrame' has no attribute 'iat'"
**A:** Outdated pandas version. Update:
```bash
pip install --upgrade pandas
```

### Q: "Connection timeout (SGX API)"
**A:** Temporary API issue. Retry later or skip SGX:
```python
# Comment out SGX scan section (lines ~667-833)
```

### Q: "Output files not in Google Drive"
**A:** Verify Drive mounting succeeded:
```python
import os
print(os.listdir("/content/drive/MyDrive"))  # Should list folders
```

### Q: "Why is SGX missing my ticker?"
**A:** Verify:
1. Ticker exists on SGX website
2. Volume > 100,000 shares/day in last trading day
3. Price > S$0.20
4. Check `sgx_scan_results.csv` (full results) for why signal didn't confirm

### Q: "HTML FA report looks broken"
**A:** Use a modern browser (Chrome/Firefox/Safari). Some mobile browsers have rendering issues.

---

## Changelog

### v7.0 (Latest - 2026-05-12)
- Google Drive auto-mount for Colab
- FA HTML reports combined into single PDF
- SPY + STI benchmark data exported as CSV
- 14-period RSI computed per stock vs index
- RSI ratio column added to results
- data_adapter.py + screener_module.py compatibility layer
- Chinese UI for FA Screener v6.0
- Production-hardened error handling

### v6.0 (2026-05-10)
- GlobalScreenerV59_CN with HTML report generation
- Financial data transparency block
- Market-specific FA thresholds (SG vs US)
- Bank/REIT special handling

### v5.9 (2026-05-06)
- First HTML report version
- Light background theme (#f5f5f0)
- Full Chinese labels

### Earlier Versions
- Pure Markdown output
- CLI-only interface

---

## License

This project is provided as-is for educational and research purposes. No warranty implied.

---

## Support & Contributing

- **Issues:** GitHub Issues tab
- **Discussions:** GitHub Discussions
- **PRs:** Welcome for improvements!

---

## Disclaimer

⚠️ **This scanner is for research purposes only. NOT investment advice.**

- Confirmed signals do NOT guarantee profitable trades
- Always conduct your own due diligence (DD)
- Risk management is your responsibility
- Past performance ≠ future results
- Consult a licensed financial advisor before investing

---

**Last Updated:** 2026-05-12  
**Version:** 7.0 (Colab-ready, production)  
**Author:** gfky1356-ship-it  
**Repository:** https://github.com/gfky1356-ship-it/SGX-data-download
