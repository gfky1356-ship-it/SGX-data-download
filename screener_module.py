"""
screener_module.py  —  Module 2: Independent Screener Unit
===========================================================
Architecture
  Module 1  data_adapter.py     →  WHERE is the data?
  Module 2  screener_module.py  →  WHAT are the conditions?

This file is divided into 3 sections (separated by dot lines):

  SECTION A  [FIXED]    Default link to Module 1.
                        Do NOT change this section.

  SECTION B  [REPLACE]  Your screener indicator functions.
                        Replace this section when creating
                        a new screener (e.g. screener_macd.py).

  SECTION C  [FIXED]    StockScreener class — scan loop and
                        result output.  Do NOT change scan(),
                        scan_us(), scan_sg(), or confirmed().
                        Only update _screen_one() to call your
                        new functions from Section B.

To create a new screener
  1. Copy this file → e.g. screener_macd.py
  2. In Section B, delete the 3 existing functions and write yours
  3. In Section C, update _screen_one() to call your new functions
  4. Section A and the rest of Section C stay exactly the same

Result dict fields (returned by scan)
  symbol            str    ticker symbol
  signal_confirmed  bool   True if any buy signal fired + OBV sell-ban
  sma_gap_buy       bool   SMA Gap Enlarge Buy signal
  breakout_buy      bool   Breakout Buy signal
  obv_sell_ban      bool   OBV long-SMA slope > 0 (uptrend confirmed)
  last_close        float  most recent closing price
  last_date         str    date of last bar  (YYYY-MM-DD)
  bars              int    number of bars in downloaded history
  rsi14             float  stock 14-period RSI (latest bar)
  index_rsi14       float  benchmark 14-period RSI (latest bar)
  rsi_ratio         float  rsi14 / index_rsi14
                           > 1.0 → stock stronger than index
                           < 1.0 → stock weaker  than index
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════
#  SECTION A  ·  FIXED  ·  DEFAULT LINK TO MODULE 1 (data_adapter)
# ════════════════════════════════════════════════════════════════════
#
#  DEFAULT SETTING:
#    Data is sourced from yfinance via data_adapter.py (Module 1).
#    When you run StockScreener(), it automatically connects here.
#
#  To use a different data source (e.g. pre-downloaded CSVs):
#    from data_adapter import DataAdapter
#    adapter = DataAdapter(source=DataAdapter.SOURCE_CSV, data_dir="...")
#    screener = StockScreener(adapter=adapter)
#
#  DO NOT edit this section when creating a new screener.
# ════════════════════════════════════════════════════════════════════

try:
    from data_adapter import DataAdapter, default_adapter   # ← Module 1 link
except ImportError:
    DataAdapter     = None   # allows import without data_adapter on path;
    default_adapter = None   # StockScreener() will raise if adapter=None


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's smoothed RSI — shared utility, keep as-is."""
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _sma(series: pd.Series, length: int) -> pd.Series:
    """Simple moving average — shared utility, keep as-is."""
    return series.rolling(length, min_periods=length).mean()


# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
#  SECTION B  ·  REPLACE THIS SECTION FOR A NEW SCREENER
# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
#
#  CURRENT SCREENER:  SMA Gap Enlarge Buy  +  Breakout Buy  +  OBV
#
#  HOW TO CREATE A NEW SCREENER (e.g. screener_macd.py):
#    Step 1 — Copy this file and rename it
#    Step 2 — Delete the 3 functions below
#    Step 3 — Write your own indicator functions here
#    Step 4 — In Section C, update _screen_one() to call your functions
#    Step 5 — Section A and the rest of Section C stay exactly the same
#
#  The function names below (_sma_gap_signals, _breakout_signals,
#  _obv_signals) are just names — change them to whatever you like.
#  Just make sure _screen_one() in Section C calls the new names.
# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·

def _sma_gap_signals(df: pd.DataFrame, index_close: pd.Series) -> pd.DataFrame:
    """
    Indicator 1 of 3 — SMA Gap Enlarge Buy / Sell
    Fires buy_signal when SMA20 > SMA50 (bull), RS rising 2 days, green candle.
    """
    out = df.copy()
    ic  = index_close.reindex(out.index).astype(float)
    out["b_sma20"]       = _sma(out["close"], 20)
    out["b_sma50"]       = _sma(out["close"], 50)
    out["b_raw_slope20"] = (out["b_sma20"] - out["b_sma20"].shift(5)) / out["b_sma20"].shift(5) * 100.0
    out["b_raw_slope50"] = (out["b_sma50"] - out["b_sma50"].shift(5)) / out["b_sma50"].shift(5) * 100.0
    out["b_slope20"]     = out["b_raw_slope20"]
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
    """
    Indicator 2 of 3 — Breakout Buy (3 lookback windows: 30, 60, 90 bars)
    Fires any_buy_signal when price breaks prior swing high above SMA20,
    SMA200 sloping up, with a prior pullback below SMA20.
    """
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
            if pos < 1: signals.append(False); continue
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
            a_above  = (a_offset is not None and pd.notna(a_price) and
                        pd.notna(out["sma20"].iat[pos - a_offset]) and
                        a_price > out["sma20"].iat[pos - a_offset])
            b_cross  = pd.notna(a_price) and out["close"].iat[pos] > a_price and out["close"].iat[pos - 1] <= a_price
            b_within = pd.notna(a_price) and out["close"].iat[pos] <= a_price * (1 + max_upside_pct / 100)
            sig = (bool(out["sma200_slope_positive"].iat[pos]) and bool(out["green_candle"].iat[pos]) and
                   bool(out["sma20_above50"].iat[pos]) and a_above and
                   bool(out["b_above20sma"].iat[pos]) and pb and b_cross and b_within)
            signals.append(sig)
        out[f"{label}_signal"] = signals
    out["any_buy_signal"] = out["n1_signal"] | out["n2_signal"] | out["n3_signal"]
    return out


def _obv_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indicator 3 of 3 — OBV sell_ban / buy_ban (trend confirmation)
    sell_ban = True means OBV long-SMA is rising → uptrend confirmed.
    Used as a gate: buy signals only count when sell_ban is active.
    """
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
            return ((ix[i] - ix[start]).total_seconds() / 86400.0
                    if isinstance(ix, pd.DatetimeIndex) else float(i - start))
        if ab and above_start is not None and _elapsed(above_start) > 1.0:
            if not buy_triggered and not bool(out["buy_ban"].iat[i]):
                buy_sig[i] = True; buy_triggered = True
        if bl and below_start is not None and _elapsed(below_start) > 1.0:
            if not sell_triggered and not bool(out["sell_ban"].iat[i]):
                sell_sig[i] = True; sell_triggered = True
    out["buy_signal"]  = buy_sig
    out["sell_signal"] = sell_sig
    return out

# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
#  END OF SECTION B  ·  REPLACEABLE SCREENER CONDITIONS
# · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·


# ════════════════════════════════════════════════════════════════════
#  SECTION C  ·  FIXED  ·  SCREENER CLASS — SCAN LOOP & OUTPUT
# ════════════════════════════════════════════════════════════════════
#
#  DO NOT change scan(), scan_us(), scan_sg(), or confirmed().
#  They handle data loading, RSI ratio, and result formatting
#  for ALL screeners — they are the same in every screener file.
#
#  The ONLY method to update when writing a new screener is
#  _screen_one() — change it to call your Section B functions.
# ════════════════════════════════════════════════════════════════════

class StockScreener:

    MIN_BARS = 110  # minimum history bars required to run indicators

    def __init__(self, adapter: "DataAdapter | None" = None):
        """
        adapter : DataAdapter from Module 1.
                  Leave as None to use the default (yfinance, last 1 year).
        """
        if adapter is None:
            if default_adapter is None:
                raise ImportError(
                    "data_adapter.py not found. "
                    "Pass an explicit DataAdapter instance or add data_adapter.py to your path."
                )
            adapter = default_adapter
        self.adapter = adapter   # ← always points to Module 1

    # ── FIXED — do not change these methods ─────────────────

    def scan(self, tickers: list, market: str = "US") -> list:
        """Scan a list of tickers. market = 'US' (SPY) or 'SG' (^STI)."""
        bench_close = self.adapter.load_benchmark(market)
        bench_rsi   = calc_rsi(bench_close)
        idx_rsi_val = float(bench_rsi.iloc[-1]) if pd.notna(bench_rsi.iloc[-1]) else None
        results = []
        for ticker in tickers:
            try:
                df = self.adapter.load_ohlcv(ticker)
                if len(df) < self.MIN_BARS:
                    continue
                results.append(self._screen_one(ticker, df, bench_close, idx_rsi_val))
            except Exception as exc:
                print(f"  ⚠️  {ticker}: {exc}")
        return results

    def scan_us(self, tickers: list) -> list:
        """Scan US tickers — benchmark: SPY."""
        return self.scan(tickers, market="US")

    def scan_sg(self, tickers: list) -> list:
        """Scan SG tickers — benchmark: TVC:STI."""
        return self.scan(tickers, market="SG")

    def confirmed(self, results: list) -> list:
        """Return only rows where signal_confirmed = True."""
        return [r for r in results if r["signal_confirmed"]]

    # ── UPDATE THIS METHOD when writing a new screener ───────
    #    Call your Section B functions here, then build the
    #    confirmed flag and return the result dict.
    #    The result dict keys must stay the same so that
    #    downstream code (exports, reports) keeps working.

    def _screen_one(
        self,
        ticker: str,
        df: pd.DataFrame,
        bench_close: pd.Series,
        idx_rsi_val: float | None,
    ) -> dict:
        # ── call Section B indicator functions ───────────────
        df_sma = _sma_gap_signals(df, bench_close)   # Indicator 1
        df_bo  = _breakout_signals(df)               # Indicator 2
        df_obv = _obv_signals(df)                    # Indicator 3

        sma_sig      = bool(df_sma["buy_signal"].iloc[-1])
        bo_sig       = bool(df_bo["any_buy_signal"].iloc[-1])
        obv_sell_ban = bool(df_obv["sell_ban"].iloc[-1])

        # confirmed = (Indicator1 OR Indicator2) AND Indicator3
        confirmed = (sma_sig and obv_sell_ban) or (bo_sig and obv_sell_ban)

        # RSI14 ratio vs benchmark (keep this block in your new screener)
        stock_rsi_s = calc_rsi(df["close"])
        stock_rsi14 = (round(float(stock_rsi_s.iloc[-1]), 2)
                       if pd.notna(stock_rsi_s.iloc[-1]) else None)
        rsi_ratio   = (round(stock_rsi14 / idx_rsi_val, 4)
                       if stock_rsi14 and idx_rsi_val and idx_rsi_val > 0 else None)

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


# ── Standalone demo (runs only when you execute this file directly) ──
if __name__ == "__main__":
    print("=" * 55)
    print("  screener_module.py — standalone demo")
    print("=" * 55)
    screener = StockScreener()   # uses default_adapter → yfinance
    print(f"  Adapter : {screener.adapter}")
    print("\n  Scanning AAPL, MSFT, NVDA (US) ...")
    results = screener.scan_us(["AAPL", "MSFT", "NVDA"])
    print(f"\n  {'Symbol':<8} {'Confirmed':^10} {'RSI14':>6} {'IdxRSI':>7} {'Ratio':>7}")
    print(f"  {'-'*8} {'-'*10} {'-'*6} {'-'*7} {'-'*7}")
    for r in results:
        rsi_s   = f"{r['rsi14']:.1f}"     if r["rsi14"]    else "  N/A"
        ratio_s = f"{r['rsi_ratio']:.3f}" if r["rsi_ratio"] else "  N/A"
        print(f"  {r['symbol']:<8} {'YES' if r['signal_confirmed'] else 'no':^10} "
              f"{rsi_s:>6} {r['index_rsi14'] or 'N/A':>7} {ratio_s:>7}")
    print(f"\n  Confirmed: {len(screener.confirmed(results))}/{len(results)}")
    print("=" * 55)
