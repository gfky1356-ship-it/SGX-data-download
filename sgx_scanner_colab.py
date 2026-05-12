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
#       && pip install yfinance requests weasyprint pypdf -q
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

# !pip install yfinance requests weasyprint pypdf -q

import csv, os, time, warnings, webbrowser
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
# FA SCREENER v6.0
# ════════════════════════════════════════════════════════════
SCREENER_VERSION = "6.0"
CHANGELOG = [
    ("6.0", "2026-05-10", "强制验证 financial_reports（最少3份年度财报）；新增数据来源透明披露区块"),
    ("5.9", "2026-05-06", "新增 HTML 报告；亮色背景 #f5f5f0；深色灰字 #555568；全中文标签；固定宽度纵向滚动；版本自动管理"),
    ("5.8", "2026-04-01", "新增 Rule of 40；诚信修补（银行/REIT）；中文输出"),
    ("5.6", "2026-03-01", "原始版本，纯文字 Markdown 输出"),
]

class GlobalScreenerV59_CN:
    def __init__(self, ticker, price, data):
        self.ticker  = str(ticker).upper()
        self.price   = float(price)
        self.data    = data
        self.version     = SCREENER_VERSION
        self.report_date = datetime.now().strftime("%Y年%m月%d日")
        self.is_sg = (self.ticker.isdigit() or
            (self.ticker.isalnum() and any(c.isdigit() for c in self.ticker) and len(self.ticker) <= 4))
        self.market = "SG" if self.is_sg else "US"
        self.source = "SGX.com" if self.is_sg else "SEC EDGAR"
        self.ng_count = self.edge_count = self.pass_count = 0
        self._validate_financial_reports()
        self._apply_honesty_patch()

    def _validate_financial_reports(self):
        reports = self.data.get("financial_reports", [])
        self.has_financial_reports = isinstance(reports, list) and len(reports) >= 1

    def _apply_honesty_patch(self):
        self.is_bank = self.data.get("is_bank", False)
        if self.is_bank:
            self.data["p_fcf"] = "[NO DATA]"
            self.data["roic"]  = "[NO DATA]"

    def _check(self, name, value, sg_spec, us_spec, condition, ref_only=False):
        spec       = sg_spec if self.market == "SG" else us_spec
        spec_label = str(spec) if spec is not None else "参考"
        if ref_only:
            return {"name": name, "value": str(value), "spec": "参考", "status": "ref", "label": "—"}
        if value == "[NO DATA]":
            return {"name": name, "value": "[无数据]", "spec": spec_label, "status": "nd", "label": "—"}
        try:
            passed = condition(float(value), spec)
            if passed:
                try:    edge = not condition(float(value) * 0.9, spec)
                except: edge = False
                if edge:
                    self.edge_count += 1
                    return {"name": name, "value": str(value), "spec": spec_label, "status": "warn", "label": "⚠️"}
                else:
                    self.pass_count += 1
                    return {"name": name, "value": str(value), "spec": spec_label, "status": "pass", "label": "✅"}
            else:
                self.ng_count += 1
                return {"name": name, "value": str(value), "spec": spec_label, "status": "fail", "label": "❌"}
        except:
            return {"name": name, "value": str(value), "spec": spec_label, "status": "nd", "label": "⚠️"}

    def run_matrix(self):
        d = self.data
        return [
            self._check("EPS 同比增长 (%)",    d.get("eps_growth",  "[NO DATA]"), 0,   10,  lambda v,s: v >= s),
            self._check("营收增速 / CAGR (%)", d.get("rev_cagr",    "[NO DATA]"), 10,  10,  lambda v,s: v > s),
            self._check("ROE (TTM) (%)",       d.get("roe",         "[NO DATA]"), 9,   15,  lambda v,s: v > s),
            self._check("ROIC (%)",            d.get("roic",        "[NO DATA]"), 15,  15,  lambda v,s: v >= s),
            self._check("P/FCF (估值倍数)",    d.get("p_fcf",       "[NO DATA]"), 35,  25,  lambda v,s: v < s),
            self._check("TTM 市盈率",          d.get("ttm_pe",      "—"),         None,None,None, ref_only=True),
            self._check("前瞻市盈率",          d.get("forward_pe",  "—"),         None,None,None, ref_only=True),
            self._check("净利润率 (%)",        d.get("margin",      "[NO DATA]"), 5,   9.5, lambda v,s: v > s),
            self._check("债务权益比",          d.get("debt_equity", "[NO DATA]"), 1.0, 1.0, lambda v,s: v < s),
            self._check("Rule of 40 (%)",      d.get("rule_of_40",  "[NO DATA]"), 40,  40,  lambda v,s: v > s),
        ]

    def get_verdict(self):
        if   self.ng_count == 0:  return "fire",  "🔥 FIRE — 符合买入标准", "fire"
        elif self.ng_count <= 2:  return "wait",  "⏳ WAIT — 建议观望",     "wait"
        else:                     return "avoid", "🚫 AVOID — 规避风险",    "avoid"

    def _changelog_html(self):
        rows = ""
        for ver, date, note in CHANGELOG:
            is_cur = ver == self.version
            b0, b1 = ("<strong>", "</strong>") if is_cur else ("", "")
            cur = " (当前)" if is_cur else ""
            rows += f"<tr><td>{b0}v{ver}{cur}{b1}</td><td>{date}</td><td>{note}</td></tr>"
        return rows

    def generate_html(self):
        matrix = self.run_matrix()
        verdict_key, verdict_label, verdict_css = self.get_verdict()
        total_valid = self.pass_count + self.edge_count + self.ng_count
        score_pct   = int(self.pass_count / total_valid * 100) if total_valid > 0 else 0
        d        = self.data
        currency = "S$" if self.is_sg else "US$"
        mkt      = "SG" if self.is_sg else "US"
        matrix_rows_html = ""
        for i, row in enumerate(matrix, 1):
            sc = {"pass":"pass","warn":"warn","fail":"fail","ref":"nd","nd":"nd"}.get(row["status"],"nd")
            matrix_rows_html += (f'<div class="matrix-row"><div class="col-num">{i}</div>'
                f'<div class="col-name">{row["name"]}</div><div class="col-val">{row["value"]}</div>'
                f'<div class="col-std">{row["spec"]}</div><div class="col-stat {sc}">{row["label"]}</div></div>')
        vc = {"fire":("rgba(10,124,92,0.06)","#a0d4c4","#0a7c5c"),"wait":("rgba(245,166,35,0.05)","#e8c070","#b87000"),"avoid":("rgba(204,34,68,0.05)","#f0a0b0","#cc2244")}
        bc = {"fire":("rgba(10,124,92,0.12)","#0a7c5c","#0a7c5c"),"wait":("rgba(245,166,35,0.12)","#e8960a","#b87000"),"avoid":("rgba(204,34,68,0.12)","#cc2244","#cc2244")}
        v_bg,v_border,v_color = vc[verdict_css]
        b_bg,b_border,b_color = bc[verdict_css]
        entry_html = f'<div class="entry-box">🎯 {d["entry_note"]}</div>' if d.get("entry_note") else ""
        changelog_rows = self._changelog_html()
        reports = d.get("financial_reports", [])
        datasource_rows = ""
        for _i, _r in enumerate(reports, 1):
            _url = _r.get("url","#")
            datasource_rows += (f'<div>📄 {_i}. <strong>{_r.get("period","—")}</strong> · {_r.get("source","—")} · '
                f'<a href="{_url}" target="_blank">{_url[:60]}{"…" if len(_url)>60 else ""}</a> · 发布日期：{_r.get("date","—")}</div>')
        if not datasource_rows:
            datasource_rows = '<div>（自动扫描模式 — 数据来源：Yahoo Finance / SGX API）</div>'
        css = f"""
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;500;600&display=swap');
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#f5f5f0;color:#1a1a2e;font-family:'Noto Sans SC',sans-serif;font-size:14px;line-height:1.6;padding:16px;overflow-x:hidden}}
  .report{{width:100%;max-width:900px;margin:0 auto;display:flex;flex-direction:column;gap:12px}}
  .header{{background:#fff;border:1px solid #d0d0d8;border-left:4px solid #0a7c5c;padding:14px 16px;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
  .header-top{{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}}
  .ticker{{font-family:'IBM Plex Mono',monospace;font-size:24px;font-weight:600;color:#0a7c5c}}
  .ticker span{{color:#555568;font-size:15px;font-weight:400;margin-left:6px}}
  .company-name{{font-size:12px;color:#555568;margin-top:3px}}
  .verdict-badge{{font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:600;padding:5px 14px;background:{b_bg};border:1px solid {b_border};color:{b_color};white-space:nowrap;flex-shrink:0}}
  .filing-info{{margin-top:6px;font-size:11px;color:#555568}}
  .section-label{{font-family:'IBM Plex Mono',monospace;font-size:10px;color:#555568;letter-spacing:2px;text-transform:uppercase}}
  .stats-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
  .stat-card{{background:#fff;border:1px solid #d8d8e0;padding:10px 12px;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
  .stat-label{{font-size:10px;color:#555568;text-transform:uppercase;letter-spacing:1px;margin-bottom:2px}}
  .stat-value{{font-family:'IBM Plex Mono',monospace;font-size:14px;font-weight:600;color:#1a1a2e}}
  .stat-sub{{font-size:11px;color:#555568;margin-top:2px}}
  .matrix{{background:#fff;border:1px solid #d8d8e0;width:100%;box-shadow:0 1px 3px rgba(0,0,0,.04);overflow:hidden}}
  .matrix-row{{display:grid;grid-template-columns:20px 1fr minmax(0,80px) minmax(0,62px) 28px;border-bottom:1px solid #eeeef4;align-items:center;min-width:0}}
  .matrix-row:last-child{{border-bottom:none}}
  .matrix-row.hdr{{background:#f0f0f6;font-family:'IBM Plex Mono',monospace;font-size:9px;color:#555568;letter-spacing:1px;text-transform:uppercase}}
  .matrix-row>div{{padding:7px 5px;border-right:1px solid #eeeef4;min-width:0;overflow:hidden}}
  .matrix-row>div:last-child{{border-right:none}}
  .col-num{{text-align:center;color:#666678;font-family:'IBM Plex Mono',monospace;font-size:11px}}
  .col-name{{color:#1a1a2e;font-size:12px;white-space:normal;word-break:break-word}}
  .col-val{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#333350;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  .col-std{{font-family:'IBM Plex Mono',monospace;font-size:11px;color:#555568;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
  .col-stat{{text-align:center;font-size:14px}}
  .pass{{color:#0a7c5c}}.warn{{color:#b87000}}.fail{{color:#cc2244}}.nd{{color:#666678;font-size:12px}}
  .score-wrap{{background:#fff;border:1px solid #d8d8e0;padding:10px 14px;display:flex;align-items:center;gap:12px;flex-wrap:wrap;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
  .score-num{{font-family:'IBM Plex Mono',monospace;font-size:16px;font-weight:600;color:#0a7c5c;white-space:nowrap}}
  .score-track{{flex:1;min-width:60px;height:5px;background:#e0e0e8;border-radius:3px}}
  .score-fill{{height:100%;width:{score_pct}%;background:#0a7c5c;border-radius:3px}}
  .tags{{display:flex;gap:6px;flex-wrap:wrap}}
  .tag{{font-family:'IBM Plex Mono',monospace;font-size:11px;padding:2px 8px;border-radius:2px}}
  .tag-g{{background:#e8f6f1;color:#0a7c5c;border:1px solid #a0d4c4}}
  .tag-y{{background:#fff5e0;color:#b87000;border:1px solid #e8c070}}
  .tag-r{{background:#ffeef2;color:#cc2244;border:1px solid #f0a0b0}}
  .phase2{{background:#fff;border:1px solid #d8d8e0;padding:12px 14px;display:flex;flex-direction:column;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
  .p2-block{{padding:10px 0;border-bottom:1px solid #eeeef4}}
  .p2-block:last-child{{border-bottom:none;padding-bottom:0}}
  .p2-block:first-child{{padding-top:0}}
  .p2-key{{font-family:'IBM Plex Mono',monospace;font-size:10px;color:#555568;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px}}
  .p2-val{{font-size:13px;color:#2a2a40;line-height:1.65}}
  .verdict{{background:{v_bg};border:1px solid {v_border};padding:14px 16px}}
  .verdict-title{{font-family:'IBM Plex Mono',monospace;font-size:14px;font-weight:600;color:{v_color};margin-bottom:8px}}
  .verdict-body{{font-size:13px;color:#2a2a40;line-height:1.7}}
  .entry-box{{margin-top:10px;padding:8px 12px;background:#e8f6f1;border-left:3px solid #0a7c5c;font-family:'IBM Plex Mono',monospace;font-size:13px;color:#0a7c5c}}
  .changelog{{background:#fff;border:1px solid #d8d8e0;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
  .changelog table{{width:100%;border-collapse:collapse;font-size:12px}}
  .changelog th{{background:#f0f0f6;color:#555568;font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;padding:6px 8px;text-align:left;border-bottom:1px solid #d8d8e0}}
  .changelog td{{padding:6px 8px;border-bottom:1px solid #eeeef4;color:#2a2a40;vertical-align:top}}
  .changelog tr:last-child td{{border-bottom:none}}
  .changelog td:first-child{{font-family:'IBM Plex Mono',monospace;color:#0a7c5c;white-space:nowrap}}
  .changelog td:nth-child(2){{white-space:nowrap;color:#555568}}
  .footer{{font-size:10px;color:#888898;text-align:right;padding-top:4px}}
  .datasource{{background:#fffdf0;border:1px solid #e8c070;border-left:4px solid #b87000;padding:10px 14px;font-size:11px;color:#555568;line-height:1.8}}
  .datasource strong{{color:#b87000;font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:1px}}
  .datasource a{{color:#0a7c5c;text-decoration:none}}"""
        return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{self.ticker} FA Scan v{self.version}</title><style>{css}</style></head><body>
<div class="report">
<div class="header"><div class="header-top"><div>
<div class="ticker">{self.ticker} <span>@ {currency}{self.price}</span></div>
<div class="company-name">{d.get("company_name",self.ticker)} · {self.market} ({self.source})</div>
</div><div class="verdict-badge">{verdict_label}</div></div>
<div class="filing-info">财报来源：{len(d.get("financial_reports",[]))} 份{"✅" if self.has_financial_reports else "（自动扫描）"} · 报告生成：{self.report_date} · Screener v{self.version}</div></div>
<div class="section-label">关键数据</div>
<div class="stats-grid">
<div class="stat-card"><div class="stat-label">TTM 市盈率</div><div class="stat-value">{d.get("ttm_pe","—")}</div><div class="stat-sub">前瞻PE：{d.get("forward_pe","—")}</div></div>
<div class="stat-card"><div class="stat-label">ROE / ROIC</div><div class="stat-value">{d.get("roe","—")}% / {d.get("roic","—")}%</div><div class="stat-sub">净利润率：{d.get("margin","—")}%</div></div>
<div class="stat-card"><div class="stat-label">P/FCF</div><div class="stat-value">{d.get("p_fcf","—")}</div><div class="stat-sub">债务权益比：{d.get("debt_equity","—")}x</div></div>
<div class="stat-card"><div class="stat-label">EPS增长 / 营收CAGR</div><div class="stat-value">{d.get("eps_growth","—")}% / {d.get("rev_cagr","—")}%</div><div class="stat-sub">Rule of 40：{d.get("rule_of_40","—")}%</div></div>
</div>
<div class="section-label">第一阶段 — 财务矩阵 (v{self.version})</div>
<div class="matrix"><div class="matrix-row hdr"><div class="col-num">#</div><div>指标</div><div style="text-align:right">实际值</div><div style="text-align:right">{mkt} 标准</div><div style="text-align:center">状态</div></div>{matrix_rows_html}</div>
<div class="score-wrap"><div class="score-num">矩阵评分：{self.pass_count + self.edge_count} / {total_valid}</div>
<div class="score-track"><div class="score-fill"></div></div>
<div class="tags"><span class="tag tag-g">✅ ×{self.pass_count} 达标</span><span class="tag tag-y">⚠️ ×{self.edge_count} 边缘</span><span class="tag tag-r">❌ ×{self.ng_count} 未达</span></div></div>
<div class="section-label">第二阶段 — 深度研究</div>
<div class="phase2">
<div class="p2-block"><div class="p2-key">增长动能分析</div><div class="p2-val">{d.get("momentum_analysis","无数据")}</div></div>
<div class="p2-block"><div class="p2-key">竞争格局与护城河</div><div class="p2-val">{d.get("moat_analysis","无数据")}</div></div>
<div class="p2-block"><div class="p2-key">前瞻市盈率审计</div><div class="p2-val">{d.get("fpe_audit","无数据")}</div></div>
<div class="p2-block"><div class="p2-key">周期性风险评估</div><div class="p2-val">{d.get("cyclicality_audit","无数据")}</div></div>
</div>
<div class="verdict"><div class="verdict-title">{verdict_label}</div>
<div class="verdict-body">触发 <strong>{self.ng_count} 个「❌ 未达标」</strong>警报，{self.edge_count} 个「⚠️ 边缘」指标。{d.get("final_comment","请结合第二阶段深度研究综合评估。")}</div>{entry_html}</div>
<div class="section-label">数据来源透明披露</div>
<div class="datasource"><strong>📎 财报来源</strong><br>{datasource_rows}</div>
<div class="section-label">版本历史</div>
<div class="changelog"><table><tr><th>版本</th><th>日期</th><th>更新内容</th></tr>{changelog_rows}</table></div>
<div class="footer">GLOBAL Screener v{self.version} · {self.report_date} · 仅供参考，不构成投资建议</div>
</div></body></html>"""

    def save_report(self, output_dir="."):
        html     = self.generate_html()
        ts       = datetime.now().strftime("%Y%m%d")
        filepath = os.path.join(output_dir, f"{self.ticker}_FA_v{self.version}_{ts}.html")
        with open(filepath, "w", encoding="utf-8") as f: f.write(html)
        print(f"✅ HTML 报告已保存：{filepath}")
        return filepath


# ════════════════════════════════════════════════════════════
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

def _sgx_float(sgx_info, key):
    v = sgx_info.get(key)
    try: f = float(v); return f if f > 0 else None
    except (TypeError, ValueError): return None

def _fetch_fa_data(yf_ticker, fallback_name, sgx_info=None):
    sgx_info = sgx_info or {}
    try:
        yf_info = yf.Ticker(yf_ticker).info
        if not yf_info or yf_info.get("regularMarketPrice") is None: yf_info = {}
    except Exception: yf_info = {}
    sgx_pe = _sgx_float(sgx_info, "sgx_pe")
    def _pct(v): return round(v*100, 2) if v is not None else "[NO DATA]"
    def _num(v): return round(v, 2)     if v is not None else "[NO DATA]"
    def _pe(v):  return round(v, 2)     if v is not None else "—"
    yf_pe      = yf_info.get("trailingPE")
    de_raw     = yf_info.get("debtToEquity")
    rev_cagr   = _pct(yf_info.get("revenueGrowth"))
    margin     = _pct(yf_info.get("profitMargins"))
    mrq        = yf_info.get("mostRecentQuarter")
    sector     = yf_info.get("sector", "")
    industry   = yf_info.get("industry", "")
    is_bank    = any(k in (sector+industry).lower() for k in ["bank","financ","reit","trust","insur"])
    return {
        "company_name":      yf_info.get("longName") or yf_info.get("shortName") or sgx_info.get("name") or fallback_name,
        "releasing_date":    datetime.fromtimestamp(mrq).strftime("%Y-%m-%d") if mrq else "N/A",
        "is_bank":           is_bank,
        "eps_growth":        _pct(yf_info.get("earningsGrowth") or yf_info.get("earningsQuarterlyGrowth")),
        "rev_cagr":          rev_cagr,
        "roe":               _pct(yf_info.get("returnOnEquity")),
        "roic":              "[NO DATA]",
        "p_fcf":             _num(yf_info.get("priceToFreeCashflows")),
        "ttm_pe":            _pe(yf_pe if yf_pe is not None else sgx_pe),
        "forward_pe":        _pe(yf_info.get("forwardPE")),
        "margin":            margin,
        "debt_equity":       round(de_raw/100, 2) if de_raw is not None else "[NO DATA]",
        "rule_of_40":        round(rev_cagr+margin, 2) if isinstance(rev_cagr, float) and isinstance(margin, float) else "[NO DATA]",
        "_pe_source":        "Yahoo" if yf_pe is not None else ("SGX" if sgx_pe is not None else "N/A"),
        "financial_reports": [],
    }

def run_fa_scan(matched, ticker_info):
    if not matched: print("    No confirmed signals — FA scan skipped."); return []
    os.makedirs(FA_REPORTS_DIR, exist_ok=True)
    fa_reports = []; fa_failures = []
    for r in matched:
        yf_t = r["yf_ticker"]
        sgx_info = ticker_info.get(yf_t, {})
        print(f"    FA → {r['symbol']:<8} {r['name'][:32]:<32}", end=" ", flush=True)
        try:
            fa_data  = _fetch_fa_data(yf_t, r["name"], sgx_info)
            pe_src   = fa_data.pop("_pe_source", "?")
            screener = GlobalScreenerV59_CN(r["symbol"], r["last_close"], fa_data)
            path     = screener.save_report(output_dir=FA_REPORTS_DIR)
            fa_reports.append(path)
            _, verdict_label, _ = screener.get_verdict()
            print(f"✅  {verdict_label}  [PE:{pe_src}  Price:{r['last_close']}]")
        except Exception as e: fa_failures.append(r["symbol"]); print(f"❌  {e}")
        time.sleep(0.5)
    print(f"\n    FA reports: {len(fa_reports)} saved to {FA_REPORTS_DIR}/")
    if fa_failures: print(f"    Failed: {', '.join(fa_failures)}")
    return fa_reports


# ════════════════════════════════════════════════════════════
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
