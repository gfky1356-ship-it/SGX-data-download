"""
screener_module.py  —  Module 2: Independent Screener Unit
===========================================================
Contains screener conditions ONLY.  Data is sourced via
data_adapter.py (Module 1) by default — the screener never
hardcodes where data comes from.

Architecture
------------
  Module 1  data_adapter.py     → WHERE is the data?
  Module 2  screener_module.py  → WHAT are the conditions?

  Future screeners (screener_rsi.py, screener_macd.py, …)
  follow the same pattern: import DataAdapter, implement
  a scan() method, done.  Module 1 stays untouched.

Quick start
-----------
  # Default — pulls fresh data from yfinance:
  from screener_module import StockScreener
  screener = StockScreener()
  results  = screener.scan_us(["AAPL", "MSFT", "GOOGL"])
  for r in screener.confirmed(results):
      print(r["symbol"], r["rsi_ratio"])

  # Custom adapter — read from pre-downloaded CSVs:
  from data_adapter import DataAdapter
  adapter  = DataAdapter(
      source   = DataAdapter.SOURCE_CSV,
      data_dir = "/content/drive/MyDrive/StockScanner/benchmark_data",
  )
  screener = StockScreener(adapter=adapter)
  results  = screener.scan_sg(["D05.SI", "Z74.SI", "U11.SI"])

Result dict fields
------------------
  symbol            str    ticker passed to scan()
  signal_confirmed  bool   True if any scanner fired + OBV sell-ban
  sma_gap_buy       bool   SMA Gap Enlarge Buy signal
  breakout_buy      bool   Breakout Buy signal
  obv_sell_ban      bool   OBV long-SMA slope > 0 (uptrend)
  last_close        float  most recent closing price
  last_date         str    date of last bar  (YYYY-MM-DD)
  bars              int    number of bars in history
  rsi14             float  stock's 14-period RSI (latest bar)
  index_rsi14       float  benchmark's 14-period RSI (latest bar)
  rsi_ratio         float  rsi14 / index_rsi14
                           > 1.0 → stock stronger than index
                           < 1.0 → stock weaker  than index
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from data_adapter import DataAdapter, default_adapter
except ImportError:
    # Allow importing this module even without data_adapter on the path,
    # but instantiating StockScreener() without an adapter will raise.
    DataAdapter    = None
    default_adapter = None


# ════════════════════════════════════════════════════════════
# RSI UTILITY
# ════════════════════════════════════════════════════════════
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's smoothed RSI."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


# ════════════════════════════════════════════════════════════
# TECHNICAL INDICATOR HELPERS
# ════════════════════════════════════════════════════════════
def _sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()


def _sma_gap_signals(df: pd.DataFrame, index_close: pd.Series) -> pd.DataFrame:
    """SMA Gap Enlarge Buy / Sell signals."""
    out = df.copy()
    ic  = index_close.reindex(out.index).astype(float)
    out["b_sma20"]       = _sma(out["close"], 20)
    out["b_sma50"]       = _sma(out["close"], 50)
    out["b_raw_slope20"] = (out["b_sma20"] - out["b_sma20"].shift(5)) / out["b_sma20"].shift(5) * 100.0
    out["b_raw_slope50"] = (out["b_sma50"] - out["b_sma50"].shift(5)) / out["b_sma50"].shift(5) * 100.0
    out["b_slope20"]     = out["b_raw_slope20"]  # smooth_len=1 → no-op
    out["b_slope50"]     = out["b_raw_slope50"]
    out["b_bull"]        = out["b_slope20"] > out["b_slope50"]
    stock_sma        = _sma(out["close"], 30)
    index_sma        = _sma(ic, 30)
    out["delta_ab"]  = (out["close"] / stock_sma) - (ic / index_sma)
    out["delta_ma"]  = _sma(out["delta_ab"], 9)
    out["slope_gap"] = out["b_slope20"] - out["b_slope50"]
    out["a_two_day_green_rising"] = (
        (out["delta_ab"] >= 0) & (out["delta_ab"].shift(1) >= 0) &
        (out["delta_ab"] > out["delta_ab"].shift(1))
    )
    out["green_candle"]       = out["close"] > out["open"]
    out["red_candle"]         = out["close"] < out["open"]
    out["slope_gap_reducing"] = (
        (out["b_slope20"] > out["b_slope50"]) &
        (out["slope_gap"] < out["slope_gap"].shift(1))
    )
    out["buy_condition"]  = out["b_bull"] & out["a_two_day_green_rising"] & out["green_candle"]
    out["sell_condition"] = (out["delta_ab"] < out["delta_ma"]) & out["slope_gap_reducing"] & out["red_candle"]

    buy_sig = np.zeros(len(out), dtype=bool)
    sell_sig = np.zeros(len(out), dtype=bool)
    in_pos = buy_ban = sell_ban = False
    for i in range(len(out)):
        b = bool(out["buy_condition"].iat[i])  and not buy_ban  and not in_pos
        s = bool(out["sell_condition"].iat[i]) and not sell_ban and in_pos
        if b: buy_sig[i]  = True; in_pos = True;  buy_ban = True;  sell_ban = False
        if s: sell_sig[i] = True; in_pos = False; sell_ban = True; buy_ban  = False
    out["buy_signal"]  = buy_sig
    out["sell_signal"] = sell_sig
    return out


def _breakout_signals(df: pd.DataFrame, max_upside_pct: float = 5.0) -> pd.DataFrame:
    """Breakout Buy signals (3 lookback windows: 30, 60, 90 bars)."""
    out = df.copy()
    out["sma20"]  = _sma(out["close"], 20)
    out["sma50"]  = _sma(out["close"], 50)
    out["sma200"] = _sma(out["close"], 200)
    out["sma200_slope_positive"] = out["sma200"] > out["sma200"].shift(5)
    out["green_candle"]  = out["close"] > out["open"]
    out["sma20_above50"] = out["sma20"]  > out["sma50"]
    out["b_above20sma"]  = out["close"]  > out["sma20"]

    for label, lookback in [("n1", 30), ("n2", 60), ("n3", 90)]:
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
                        lo  = out["low"].iat[pos - off]
                        s20 = out["sma20"].iat[pos - off]
                        if pd.notna(lo) and pd.notna(s20) and lo < s20:
                            pb = True
            a_above = (
                a_offset is not None and pd.notna(a_price) and
                pd.notna(out["sma20"].iat[pos - a_offset]) and
                a_price > out["sma20"].iat[pos - a_offset]
            )
            b_cross  = pd.notna(a_price) and out["close"].iat[pos] > a_price and out["close"].iat[pos - 1] <= a_price
            b_within = pd.notna(a_price) and out["close"].iat[pos] <= a_price * (1 + max_upside_pct / 100)
            sig = (
                bool(out["sma200_slope_positive"].iat[pos]) and
                bool(out["green_candle"].iat[pos]) and
                bool(out["sma20_above50"].iat[pos]) and
                a_above and
                bool(out["b_above20sma"].iat[pos]) and
                pb and b_cross and b_within
            )
            signals.append(sig)
        out[f"{label}_signal"] = signals
    out["any_buy_signal"] = out["n1_signal"] | out["n2_signal"] | out["n3_signal"]
    return out


def _obv_signals(df: pd.DataFrame) -> pd.DataFrame:
    """OBV sell_ban / buy_ban signals."""
    out   = df.copy()
    close = out["close"]
    vol   = out["volume"]
    dv    = np.where(close > close.shift(1), vol,
                     np.where(close < close.shift(1), -vol, 0))
    out["obv"]           = pd.Series(dv, index=out.index).cumsum()
    out["obv_short_sma"] = _sma(out["obv"], 20)
    out["obv_long_sma"]  = _sma(out["obv"], 100)
    slope                = out["obv_long_sma"] - out["obv_long_sma"].shift(1)
    out["sell_ban"]      = slope > 0
    out["buy_ban"]       = slope < 0
    above = (out["obv"] > out["obv_short_sma"]) & (out["obv"] > out["obv_long_sma"])
    below = (out["obv"] < out["obv_short_sma"]) & (out["obv"] < out["obv_long_sma"])
    out["obv_above_both"] = above
    out["obv_below_both"] = below

    buy_sig = np.zeros(len(out), dtype=bool)
    sell_sig = np.zeros(len(out), dtype=bool)
    above_start = below_start = None
    buy_triggered = sell_triggered = False
    for i in range(len(out)):
        ab = bool(above.iat[i]) if pd.notna(above.iat[i]) else False
        bl = bool(below.iat[i]) if pd.notna(below.iat[i]) else False
        if ab:
            above_start = above_start if above_start is not None else i
            below_start = None; sell_triggered = False
        else:
            above_start = None; buy_triggered = False
        if bl:
            below_start = below_start if below_start is not None else i
            above_start = None; buy_triggered = False
        else:
            below_start = None; sell_triggered = False

        def _elapsed(start):
            ix = out.index
            return (
                (ix[i] - ix[start]).total_seconds() / 86400.0
                if isinstance(ix, pd.DatetimeIndex) else float(i - start)
            )

        if ab and above_start is not None and _elapsed(above_start) > 1.0:
            if not buy_triggered and not bool(out["buy_ban"].iat[i]):
                buy_sig[i] = True; buy_triggered = True
        if bl and below_start is not None and _elapsed(below_start) > 1.0:
            if not sell_triggered and not bool(out["sell_ban"].iat[i]):
                sell_sig[i] = True; sell_triggered = True

    out["buy_signal"]  = buy_sig
    out["sell_signal"] = sell_sig
    return out


# ════════════════════════════════════════════════════════════
# SCREENER CLASS
# ════════════════════════════════════════════════════════════
class StockScreener:
    """
    Independent screener.  Controls the data source via the
    DataAdapter (Module 1).  Does not hard-code any API calls.

    To use a different screener strategy in the future, subclass
    this or create a new class with the same scan() interface.
    """

    MIN_BARS = 110  # minimum history bars required

    def __init__(self, adapter: "DataAdapter | None" = None):
        """
        Parameters
        ----------
        adapter : DataAdapter instance.  Defaults to default_adapter
                  (yfinance, last 1 year) from data_adapter.py.
        """
        if adapter is None:
            if default_adapter is None:
                raise ImportError(
                    "data_adapter.py not found on the Python path. "
                    "Either install it or pass an explicit DataAdapter instance."
                )
            adapter = default_adapter
        self.adapter = adapter

    # ── Public API ───────────────────────────────────────────

    def scan(self, tickers: list, market: str = "US") -> list:
        """
        Scan a list of tickers against the given market benchmark.

        Parameters
        ----------
        tickers : list of ticker strings (as recognised by the adapter)
        market  : "US" (SPY benchmark) or "SG" (^STI benchmark)

        Returns
        -------
        List of result dicts — one per successfully scanned ticker.
        See module docstring for field descriptions.
        """
        bench_close = self.adapter.load_benchmark(market)
        bench_rsi   = calc_rsi(bench_close)
        idx_rsi_val = (
            float(bench_rsi.iloc[-1])
            if pd.notna(bench_rsi.iloc[-1]) else None
        )

        results = []
        for ticker in tickers:
            try:
                df = self.adapter.load_ohlcv(ticker)
                if len(df) < self.MIN_BARS:
                    continue
                result = self._screen_one(ticker, df, bench_close, idx_rsi_val)
                results.append(result)
            except Exception as exc:
                print(f"  ⚠️  {ticker}: {exc}")
        return results

    def scan_us(self, tickers: list) -> list:
        """Convenience wrapper: scan US tickers against SPY."""
        return self.scan(tickers, market="US")

    def scan_sg(self, tickers: list) -> list:
        """Convenience wrapper: scan SG tickers against TVC:STI."""
        return self.scan(tickers, market="SG")

    def confirmed(self, results: list) -> list:
        """Filter scan results to only confirmed (signal_confirmed=True) rows."""
        return [r for r in results if r["signal_confirmed"]]

    # ── Private ──────────────────────────────────────────────

    def _screen_one(
        self,
        ticker: str,
        df: pd.DataFrame,
        bench_close: pd.Series,
        idx_rsi_val: float | None,
    ) -> dict:
        df_sma = _sma_gap_signals(df, bench_close)
        df_bo  = _breakout_signals(df)
        df_obv = _obv_signals(df)

        sma_sig      = bool(df_sma["buy_signal"].iloc[-1])
        bo_sig       = bool(df_bo["any_buy_signal"].iloc[-1])
        obv_sell_ban = bool(df_obv["sell_ban"].iloc[-1])
        confirmed    = (sma_sig and obv_sell_ban) or (bo_sig and obv_sell_ban)

        stock_rsi_s = calc_rsi(df["close"])
        stock_rsi14 = (
            round(float(stock_rsi_s.iloc[-1]), 2)
            if pd.notna(stock_rsi_s.iloc[-1]) else None
        )
        rsi_ratio = (
            round(stock_rsi14 / idx_rsi_val, 4)
            if stock_rsi14 and idx_rsi_val and idx_rsi_val > 0
            else None
        )

        return {
            "symbol":           ticker,
            "signal_confirmed": confirmed,
            "sma_gap_buy":      sma_sig,
            "breakout_buy":     bo_sig,
            "obv_sell_ban":     obv_sell_ban,
            "last_close":       round(float(df["close"].iloc[-1]), 4),
            "last_date":        df.index[-1].strftime("%Y-%m-%d"),
            "bars":             len(df),
            "rsi14":            stock_rsi14,
            "index_rsi14":      round(idx_rsi_val, 2) if idx_rsi_val else None,
            "rsi_ratio":        rsi_ratio,
        }


# ── Standalone usage demo ────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  screener_module.py — standalone demo")
    print("=" * 55)

    screener = StockScreener()
    print(f"  Adapter : {screener.adapter}")

    print("\n  Scanning AAPL, MSFT, NVDA (US market) ...")
    us_results = screener.scan_us(["AAPL", "MSFT", "NVDA"])
    print(f"\n  {'Symbol':<8} {'Confirmed':^10} {'RSI14':>6} {'IdxRSI':>7} {'Ratio':>7}")
    print(f"  {'-'*8} {'-'*10} {'-'*6} {'-'*7} {'-'*7}")
    for r in us_results:
        rsi_s   = f"{r['rsi14']:.1f}"  if r["rsi14"]    else "  N/A"
        ratio_s = f"{r['rsi_ratio']:.3f}" if r["rsi_ratio"] else "  N/A"
        print(f"  {r['symbol']:<8} {'YES' if r['signal_confirmed'] else 'no':^10} "
              f"{rsi_s:>6} {r['index_rsi14'] or 'N/A':>7} {ratio_s:>7}")

    print(f"\n  Confirmed signals: {len(screener.confirmed(us_results))}/{len(us_results)}")
    print("=" * 55)
