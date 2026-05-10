"""
GLOBAL Screener v5.9 — HTML 报告输出版（含版本自动管理）
作者：Hua (via Claude)
更新日期：2026年5月

变更历史：
  v5.6  原始版本，纯文字 Markdown 输出
  v5.8  新增 Rule of 40；诚信修补（银行/REIT）；中文输出
  v5.9  新增 HTML 报告生成；亮色背景；深色灰字；全中文标签；纵向滚动；版本自动管理

使用说明：
  1. 直接运行此文件 → 检测版本 → 自动生成带版本号的 .py 副本 → 生成示例报告
  2. 每次修改 SCREENER_VERSION + CHANGELOG → 重新运行 → 自动生成新版本文件
  3. 手动调用：screener.export_versioned_script(output_dir=".")
"""

import datetime
import os
import re
import webbrowser

# ─────────────────────────────────────────────
# 版本配置（每次更新只需改这里）
# ─────────────────────────────────────────────
SCREENER_VERSION = "5.9"

CHANGELOG = [
    ("5.9", "2026-05-06", "新增 HTML 报告；亮色背景 #f5f5f0；深色灰字 #555568；全中文标签；固定宽度纵向滚动；版本自动管理"),
    ("5.8", "2026-04-01", "新增 Rule of 40；诚信修补（银行/REIT）；中文输出"),
    ("5.6", "2026-03-01", "原始版本，纯文字 Markdown 输出"),
]


class GlobalScreenerV59_CN:
    """GLOBAL Screener v5.9 — HTML 报告 + 版本自动管理"""

    def __init__(self, ticker, price, data):
        self.ticker = str(ticker).upper()
        self.price = float(price)
        self.data = data
        self.version = SCREENER_VERSION
        self.report_date = datetime.datetime.now().strftime("%Y年%m月%d日")

        self.is_sg = (
            self.ticker.isdigit()
            or (self.ticker.isalnum() and any(c.isdigit() for c in self.ticker) and len(self.ticker) <= 4)
        )
        self.market = "SG" if self.is_sg else "US"
        self.source = "SGX.com" if self.is_sg else "SEC EDGAR"

        self.ng_count = 0
        self.edge_count = 0
        self.pass_count = 0
        self._apply_honesty_patch()

    def _apply_honesty_patch(self):
        self.is_bank = self.data.get("is_bank", False)
        if self.is_bank:
            self.data["p_fcf"] = "[NO DATA]"
            self.data["roic"] = "[NO DATA]"

    def _check(self, name, value, sg_spec, us_spec, condition, ref_only=False):
        spec = sg_spec if self.market == "SG" else us_spec
        spec_label = str(spec) if spec is not None else "参考"

        if ref_only:
            return {"name": name, "value": str(value), "spec": "参考", "status": "ref", "label": "—"}
        if value == "[NO DATA]":
            return {"name": name, "value": "[无数据]", "spec": spec_label, "status": "nd", "label": "—"}
        try:
            passed = condition(float(value), spec)
            if passed:
                try:
                    edge = not condition(float(value) * 0.9, spec)
                except Exception:
                    edge = False
                if edge:
                    self.edge_count += 1
                    return {"name": name, "value": str(value), "spec": spec_label, "status": "warn", "label": "⚠️"}
                else:
                    self.pass_count += 1
                    return {"name": name, "value": str(value), "spec": spec_label, "status": "pass", "label": "✅"}
            else:
                self.ng_count += 1
                return {"name": name, "value": str(value), "spec": spec_label, "status": "fail", "label": "❌"}
        except Exception:
            return {"name": name, "value": str(value), "spec": spec_label, "status": "nd", "label": "⚠️"}

    def run_matrix(self):
        d = self.data
        return [
            self._check("EPS 同比增长 (%)",    d.get("eps_growth",  "[NO DATA]"), 0,    10,   lambda v, s: v >= s),
            self._check("营收增速 / CAGR (%)", d.get("rev_cagr",    "[NO DATA]"), 10,   10,   lambda v, s: v > s),
            self._check("ROE (TTM) (%)",       d.get("roe",         "[NO DATA]"), 9,    15,   lambda v, s: v > s),
            self._check("ROIC (%)",            d.get("roic",        "[NO DATA]"), 15,   15,   lambda v, s: v >= s),
            self._check("P/FCF (估值倍数)",    d.get("p_fcf",       "[NO DATA]"), 35,   25,   lambda v, s: v < s),
            self._check("TTM 市盈率",          d.get("ttm_pe",      "—"),         None, None, None, ref_only=True),
            self._check("前瞻市盈率",          d.get("forward_pe",  "—"),         None, None, None, ref_only=True),
            self._check("净利润率 (%)",        d.get("margin",      "[NO DATA]"), 5,    9.5,  lambda v, s: v > s),
            self._check("债务权益比",          d.get("debt_equity", "[NO DATA]"), 1.0,  1.0,  lambda v, s: v < s),
            self._check("Rule of 40 (%)",      d.get("rule_of_40",  "[NO DATA]"), 40,   40,   lambda v, s: v > s),
        ]

    def get_verdict(self):
        if self.ng_count == 0:
            return "fire",  "🔥 FIRE — 符合买入标准", "fire"
        elif self.ng_count <= 2:
            return "wait",  "⏳ WAIT — 建议观望",     "wait"
        else:
            return "avoid", "🚫 AVOID — 规避风险",    "avoid"

    def _changelog_html(self):
        rows = ""
        for ver, date, note in CHANGELOG:
            is_cur = ver == self.version
            b0 = "<strong>" if is_cur else ""
            b1 = "</strong>" if is_cur else ""
            cur = " (当前)" if is_cur else ""
            rows += f"<tr><td>{b0}v{ver}{cur}{b1}</td><td>{date}</td><td>{note}</td></tr>"
        return rows

    def generate_html(self):
        matrix = self.run_matrix()
        verdict_key, verdict_label, verdict_css = self.get_verdict()
        total_valid = self.pass_count + self.edge_count + self.ng_count
        score_pct = int(self.pass_count / total_valid * 100) if total_valid > 0 else 0
        d = self.data
        currency = "S$" if self.is_sg else "US$"
        mkt = "SG" if self.is_sg else "US"

        matrix_rows_html = ""
        for i, row in enumerate(matrix, 1):
            sc = {"pass": "pass", "warn": "warn", "fail": "fail", "ref": "nd", "nd": "nd"}.get(row["status"], "nd")
            matrix_rows_html += (
                f'<div class="matrix-row">' +
                f'<div class="col-num">{i}</div>' +
                f'<div class="col-name">{row["name"]}</div>' +
                f'<div class="col-val">{row["value"]}</div>' +
                f'<div class="col-std">{row["spec"]}</div>' +
                f'<div class="col-stat {sc}">{row["label"]}</div>' +
                f'</div>'
            )

        vc = {"fire": ("rgba(10,124,92,0.06)", "#a0d4c4", "#0a7c5c"),
              "wait": ("rgba(245,166,35,0.05)", "#e8c070", "#b87000"),
              "avoid": ("rgba(204,34,68,0.05)", "#f0a0b0", "#cc2244")}
        bc = {"fire": ("rgba(10,124,92,0.12)", "#0a7c5c", "#0a7c5c"),
              "wait": ("rgba(245,166,35,0.12)", "#e8960a", "#b87000"),
              "avoid": ("rgba(204,34,68,0.12)", "#cc2244", "#cc2244")}
        v_bg, v_border, v_color = vc[verdict_css]
        b_bg, b_border, b_color = bc[verdict_css]
        entry_html = f'<div class="entry-box">🎯 {d["entry_note"]}</div>' if d.get("entry_note") else ""
        changelog_rows = self._changelog_html()

        css = f"""
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Noto+Sans+SC:wght@400;500;600&display=swap');
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{background:#f5f5f0;color:#1a1a2e;font-family:'Noto Sans SC',sans-serif;font-size:14px;line-height:1.6;padding:16px;overflow-x:hidden}}
  .report{{width:100%;max-width:100%;display:flex;flex-direction:column;gap:12px}}
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
  .matrix{{background:#fff;border:1px solid #d8d8e0;width:100%;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
  .matrix-row{{display:grid;grid-template-columns:26px 1fr auto auto 38px;border-bottom:1px solid #eeeef4;align-items:center}}
  .matrix-row:last-child{{border-bottom:none}}
  .matrix-row.hdr{{background:#f0f0f6;font-family:'IBM Plex Mono',monospace;font-size:10px;color:#555568;letter-spacing:1px;text-transform:uppercase}}
  .matrix-row>div{{padding:9px 8px;border-right:1px solid #eeeef4}}
  .matrix-row>div:last-child{{border-right:none}}
  .col-num{{text-align:center;color:#666678;font-family:'IBM Plex Mono',monospace;font-size:11px}}
  .col-name{{color:#1a1a2e;font-size:13px}}
  .col-val{{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#333350;text-align:right;white-space:nowrap}}
  .col-std{{font-family:'IBM Plex Mono',monospace;font-size:12px;color:#555568;text-align:right;white-space:nowrap}}
  .col-stat{{text-align:center;font-size:15px}}
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
  .p2-val strong{{color:#1a1a2e}}
  .verdict{{background:{v_bg};border:1px solid {v_border};padding:14px 16px}}
  .verdict-title{{font-family:'IBM Plex Mono',monospace;font-size:14px;font-weight:600;color:{v_color};margin-bottom:8px}}
  .verdict-body{{font-size:13px;color:#2a2a40;line-height:1.7}}
  .verdict-body strong{{color:#1a1a2e}}
  .entry-box{{margin-top:10px;padding:8px 12px;background:#e8f6f1;border-left:3px solid #0a7c5c;font-family:'IBM Plex Mono',monospace;font-size:13px;color:#0a7c5c}}
  .changelog{{background:#fff;border:1px solid #d8d8e0;padding:12px 14px;box-shadow:0 1px 3px rgba(0,0,0,.04)}}
  .changelog table{{width:100%;border-collapse:collapse;font-size:12px}}
  .changelog th{{background:#f0f0f6;color:#555568;font-family:'IBM Plex Mono',monospace;font-size:10px;letter-spacing:1px;text-transform:uppercase;padding:6px 8px;text-align:left;border-bottom:1px solid #d8d8e0}}
  .changelog td{{padding:6px 8px;border-bottom:1px solid #eeeef4;color:#2a2a40;vertical-align:top}}
  .changelog tr:last-child td{{border-bottom:none}}
  .changelog td:first-child{{font-family:'IBM Plex Mono',monospace;color:#0a7c5c;white-space:nowrap}}
  .changelog td:nth-child(2){{white-space:nowrap;color:#555568}}
  .footer{{font-size:10px;color:#888898;text-align:right;padding-top:4px}}
"""
        return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{self.ticker} FA Scan v{self.version}</title>
<style>{css}</style></head><body>
<div class="report">
<div class="header"><div class="header-top"><div>
<div class="ticker">{self.ticker} <span>@ {currency}{self.price}</span></div>
<div class="company-name">{d.get("company_name", self.ticker)} · {self.market} ({self.source})</div>
</div><div class="verdict-badge">{verdict_label}</div></div>
<div class="filing-info">最新财报：{d.get("releasing_date","未提供")} · 报告生成：{self.report_date} · Screener v{self.version}</div></div>

<div class="section-label">关键数据</div>
<div class="stats-grid">
<div class="stat-card"><div class="stat-label">TTM 市盈率</div><div class="stat-value">{d.get("ttm_pe","—")}</div><div class="stat-sub">前瞻PE：{d.get("forward_pe","—")}</div></div>
<div class="stat-card"><div class="stat-label">ROE / ROIC</div><div class="stat-value">{d.get("roe","—")}% / {d.get("roic","—")}%</div><div class="stat-sub">净利润率：{d.get("margin","—")}%</div></div>
<div class="stat-card"><div class="stat-label">P/FCF</div><div class="stat-value">{d.get("p_fcf","—")}</div><div class="stat-sub">债务权益比：{d.get("debt_equity","—")}x</div></div>
<div class="stat-card"><div class="stat-label">EPS增长 / 营收CAGR</div><div class="stat-value">{d.get("eps_growth","—")}% / {d.get("rev_cagr","—")}%</div><div class="stat-sub">Rule of 40：{d.get("rule_of_40","—")}%</div></div>
</div>

<div class="section-label">第一阶段 — 财务矩阵 (v{self.version})</div>
<div class="matrix">
<div class="matrix-row hdr"><div class="col-num">#</div><div>指标</div><div style="text-align:right">实际值</div><div style="text-align:right">{mkt} 标准</div><div style="text-align:center">状态</div></div>
{matrix_rows_html}
</div>

<div class="score-wrap">
<div class="score-num">矩阵评分：{self.pass_count + self.edge_count} / {total_valid}</div>
<div class="score-track"><div class="score-fill"></div></div>
<div class="tags"><span class="tag tag-g">✅ ×{self.pass_count} 达标</span><span class="tag tag-y">⚠️ ×{self.edge_count} 边缘</span><span class="tag tag-r">❌ ×{self.ng_count} 未达</span></div>
</div>

<div class="section-label">第二阶段 — 深度研究</div>
<div class="phase2">
<div class="p2-block"><div class="p2-key">增长动能分析</div><div class="p2-val">{d.get("momentum_analysis","无数据")}</div></div>
<div class="p2-block"><div class="p2-key">竞争格局与护城河</div><div class="p2-val">{d.get("moat_analysis","无数据")}</div></div>
<div class="p2-block"><div class="p2-key">前瞻市盈率审计</div><div class="p2-val">{d.get("fpe_audit","无数据")}</div></div>
<div class="p2-block"><div class="p2-key">周期性风险评估</div><div class="p2-val">{d.get("cyclicality_audit","无数据")}</div></div>
</div>

<div class="verdict">
<div class="verdict-title">{verdict_label}</div>
<div class="verdict-body">触发 <strong>{self.ng_count} 个「❌ 未达标」</strong>警报，{self.edge_count} 个「⚠️ 边缘」指标。{d.get("final_comment","请结合第二阶段深度研究综合评估。")}</div>
{entry_html}
</div>

<div class="section-label">版本历史</div>
<div class="changelog"><table>
<tr><th>版本</th><th>日期</th><th>更新内容</th></tr>
{changelog_rows}
</table></div>

<div class="footer">GLOBAL Screener v{self.version} · {self.report_date} · 仅供参考，不构成投资建议</div>
</div></body></html>"""

    def save_report(self, output_dir="."):
        html = self.generate_html()
        ts = datetime.datetime.now().strftime("%Y%m%d")
        filepath = os.path.join(output_dir, f"{self.ticker}_FA_v{self.version}_{ts}.html")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"✅ HTML 报告已保存：{filepath}")
        return filepath

    def open_report(self, output_dir="."):
        filepath = self.save_report(output_dir)
        webbrowser.open(f"file://{os.path.abspath(filepath)}")
        return filepath

    # ── 版本管理：自动生成新版本 .py 文件 ──────────────
    def export_versioned_script(self, output_dir="."):
        """
        将当前运行的 .py 文件复制为带版本号的新文件：
        Global_FA_Scan_CN_5_9.py（当前版本示例）
        并在终端打印文件路径提示。
        """
        src = os.path.abspath(__file__)
        ver_clean = self.version.replace(".", "_")
        dest_name = f"Global_FA_Scan_CN_{ver_clean}.py"
        dest_path = os.path.join(os.path.abspath(output_dir), dest_name)

        with open(src, "r", encoding="utf-8") as f:
            content = f.read()

        today = datetime.datetime.now().strftime("%Y年%m月")
        content = re.sub(r"(更新日期：)\S+", f"\g<1>{today}", content, count=1)

        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(content)

        print("\n" + "=" * 58)
        print(f"  📦  新版本脚本已生成")
        print(f"  文件名：{dest_name}")
        print(f"  路径：  {dest_path}")
        print(f"  版本：  v{self.version}")
        print(f"  日期：  {datetime.datetime.now().strftime('%Y-%m-%d')}")
        print("=" * 58 + "\n")
        return dest_path


# ─────────────────────────────────────────────
# 每次运行自动检测版本，如有新版则生成 .py 文件
# ─────────────────────────────────────────────
def _auto_export_on_run(output_dir="."):
    ver_clean = SCREENER_VERSION.replace(".", "_")
    dest_name = f"Global_FA_Scan_CN_{ver_clean}.py"
    dest_path = os.path.join(os.path.abspath(output_dir), dest_name)

    src = os.path.abspath(__file__)
    # 只有当源文件和目标文件不是同一个文件时才导出
    if os.path.abspath(src) != dest_path and not os.path.exists(dest_path):
        dummy = GlobalScreenerV59_CN("__EXPORT__", 0, {})
        dummy.export_versioned_script(output_dir)
    elif os.path.abspath(src) == dest_path:
        print(f"✅ 当前运行文件已是版本命名文件：{os.path.basename(src)}")
    else:
        print(f"✅ 版本 v{SCREENER_VERSION} 文件已存在：{dest_name}")


# ─────────────────────────────────────────────
# 示例用法
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # 自动版本检测与导出
    _auto_export_on_run(output_dir=".")

    # OV8 升菘集团（SG股）示例
    ov8_data = {
        "company_name":      "Sheng Siong Group Ltd / 升菘集团",
        "releasing_date":    "FY2025 全年 (27 Feb 2026) + Q1 FY2026 (29 Apr 2026)",
        "is_bank":           False,
        "eps_growth":        8.5,
        "rev_cagr":          7.0,
        "roe":               25.17,
        "roic":              14.49,
        "p_fcf":             15.69,
        "ttm_pe":            30.5,
        "forward_pe":        21.75,
        "margin":            9.5,
        "debt_equity":       0.20,
        "rule_of_40":        "[NO DATA]",
        "momentum_analysis": "<strong>稳健上行。</strong>营收 S$1.34B → S$1.43B → S$1.57B；净利润 S$137M → S$137M → S$149M。毛利率稳定维持在 ~31%。Q1 2026 延续势头：营收 +12.4%、净利润 +12% YoY。",
        "moat_analysis":     "新加坡最大本土超市连锁之一，心巢区位优势明显，自有品牌（约1600款）提供额外利润空间。主要竞争者：FairPrice、Giant、RedMart。",
        "fpe_audit":         "TTM PE 30.5x 明显高于行业均值 16.7x。前瞻 PE 21.75x 相对合理，但仍高于历史均值 19.6x。当前股价已超过分析师共识目标 S$3.04。",
        "cyclicality_audit": "<strong>低风险。</strong>民生必需品，抗经济周期。Beta 仅 0.06。风险：市场饱和、S$520M 资本支出压制 FCF、昆明业务亏损。",
        "final_comment":     "优质防御性标的，但当前价位 S$3.13（TTM PE 30.5x）已明显溢价，且高于分析师共识目标 S$3.04。",
        "entry_note":        "理想入场区间：S$2.70 – S$2.85 · 对应 PE 约 24–26x · 较当前低约 9–14%",
    }
    screener = GlobalScreenerV59_CN("OV8", 3.13, ov8_data)
    screener.open_report(output_dir=".")

    # NFLX Netflix（US股，取消注释使用）
    # nflx_data = {
    #     "company_name":   "Netflix Inc.",
    #     "releasing_date": "2026年4月16日 (Q1 2026)",
    #     "is_bank":        False,
    #     "eps_growth":     86.08, "rev_cagr": 11.3, "roe": 48.49, "roic": 29.21,
    #     "p_fcf":          33.4,  "ttm_pe":   24.5,  "forward_pe": 28.9,
    #     "margin":         32.3,  "debt_equity": 0.45, "rule_of_40": 52.3,
    #     "momentum_analysis": "Q1数据强劲，广告套餐用户持续增长。",
    #     "moat_analysis":     "内容护城河深厚，竞争格局趋于稳定。",
    #     "fpe_audit":         "前瞻PE 28.9x，利润率扩张支撑估值。",
    #     "cyclicality_audit": "流媒体成熟期，货币化成为核心驱动。",
    #     "final_comment":     "基本面优秀，等待回调后入场。",
    #     "entry_note":        "参考入场：PE 回落至 22–24x（约 US$850–920）时建仓",
    # }
    # screener2 = GlobalScreenerV59_CN("NFLX", 1050.0, nflx_data)
    # screener2.open_report(output_dir=".")
