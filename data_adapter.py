"""
data_adapter.py  —  Module 1: Data Source Adapter
===================================================
Tells downstream screener modules WHERE to get source data.

This module decouples the data source from the screener logic.
Screener modules call load_ohlcv() and load_benchmark() without
knowing whether data comes from yfinance, local CSV files, a
database, or any future source.

Supported sources
-----------------
  DataAdapter.SOURCE_YFINANCE  — live download via yfinance (default)
  DataAdapter.SOURCE_CSV       — read pre-downloaded CSV files

Quick start
-----------
  from data_adapter import DataAdapter, default_adapter

  # Default: fresh download from yfinance
  df    = default_adapter.load_ohlcv("AAPL")
  bench = default_adapter.load_benchmark("US")   # SPY close Series

  # CSV source (after stock_scanner_colab.py has run and saved data):
  from data_adapter import DataAdapter
  adapter = DataAdapter(
      source   = DataAdapter.SOURCE_CSV,
      data_dir = "/content/drive/MyDrive/StockScanner/benchmark_data",
  )
  spy_close = adapter.load_benchmark("US")

Adding a new source in future
------------------------------
  1. Add a SOURCE_* class constant (e.g. SOURCE_PARQUET = "parquet")
  2. Add a _fetch_parquet() method
  3. Dispatch in _fetch()
  No changes required in screener modules.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


class DataAdapter:
    SOURCE_YFINANCE = "yfinance"
    SOURCE_CSV      = "csv"

    # Default benchmark tickers per market
    BENCHMARKS = {
        "US": "SPY",    # S&P 500 ETF  (TradingView: SPY)
        "SG": "^STI",   # Straits Times Index (TradingView: TVC:STI)
    }

    def __init__(
        self,
        source: str     = SOURCE_YFINANCE,
        data_dir: str   = None,
        start_date: str = None,
        end_date: str   = None,
    ):
        """
        Parameters
        ----------
        source     : SOURCE_YFINANCE or SOURCE_CSV
        data_dir   : directory for CSV files (used when source=SOURCE_CSV)
        start_date : history start, default = 1 year ago
        end_date   : history end,   default = today
        """
        self.source     = source
        self.data_dir   = data_dir or "/content/drive/MyDrive/StockScanner/benchmark_data"
        self.end_date   = end_date   or datetime.today().strftime("%Y-%m-%d")
        self.start_date = start_date or (
            datetime.today() - timedelta(days=365)
        ).strftime("%Y-%m-%d")
        self._cache: dict = {}

    # ── Public interface ─────────────────────────────────────

    def load_ohlcv(self, ticker: str) -> pd.DataFrame:
        """
        Return OHLCV DataFrame (columns lower-cased: open, high, low, close, volume)
        for a given ticker symbol.  Results are cached within the session.
        """
        key = ticker.upper()
        if key in self._cache:
            return self._cache[key]
        df = self._fetch(key)
        self._cache[key] = df
        return df

    def load_benchmark(self, market: str = "US") -> pd.Series:
        """
        Return the close price Series for the market benchmark.

        market = "US"  → SPY
        market = "SG"  → ^STI
        """
        ticker = self.BENCHMARKS.get(market.upper())
        if ticker is None:
            raise ValueError(
                f"Unknown market {market!r}. Supported: {list(self.BENCHMARKS)}"
            )
        df = self.load_ohlcv(ticker)
        col = "close" if "close" in df.columns else "Close"
        return df[col]

    def clear_cache(self):
        """Discard all in-memory cached data."""
        self._cache.clear()

    def describe(self) -> dict:
        """Return a summary dict of this adapter's configuration."""
        return {
            "source":         self.source,
            "start_date":     self.start_date,
            "end_date":       self.end_date,
            "data_dir":       self.data_dir,
            "cached_tickers": sorted(self._cache.keys()),
        }

    def __repr__(self):
        d = self.describe()
        return (
            f"DataAdapter(source={d['source']!r}, "
            f"{d['start_date']} → {d['end_date']}, "
            f"cached={len(d['cached_tickers'])})"
        )

    # ── Private ──────────────────────────────────────────────

    def _fetch(self, ticker: str) -> pd.DataFrame:
        if self.source == self.SOURCE_YFINANCE:
            return self._fetch_yfinance(ticker)
        if self.source == self.SOURCE_CSV:
            return self._fetch_csv(ticker)
        raise ValueError(
            f"Unknown source {self.source!r}. "
            f"Use {self.SOURCE_YFINANCE!r} or {self.SOURCE_CSV!r}."
        )

    def _fetch_yfinance(self, ticker: str) -> pd.DataFrame:
        raw = yf.download(
            ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )
        if raw.empty:
            raise ValueError(f"No yfinance data returned for {ticker!r}")
        return raw.rename(columns=str.lower)

    def _fetch_csv(self, ticker: str) -> pd.DataFrame:
        # Try a few filename conventions (e.g. "^STI" saved as "STI.csv")
        safe = ticker.replace("^", "").replace("/", "_")
        candidates = [
            os.path.join(self.data_dir, f"{safe}.csv"),
            os.path.join(self.data_dir, f"{ticker}.csv"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                return df.rename(columns=str.lower)
        raise FileNotFoundError(
            f"CSV file not found for {ticker!r}. "
            f"Searched: {candidates}. "
            f"Run stock_scanner_colab.py first to download and save benchmark data."
        )


# ── Module-level default instance (yfinance, last 1 year) ────
default_adapter = DataAdapter()
