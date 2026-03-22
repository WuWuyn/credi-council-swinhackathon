"""
CreditLens A4 — Professional PDF Report Generator.

Generates a full-format Vietnamese credit assessment report (Tờ Trình Tín Dụng)
with proper Unicode font support (Arial TTF on Windows).

Sections:
  - Block A scorecard (Score/Band/PD/Model)
  - SHAP waterfall chart
  - 5C scorecard with progress bars
  - Financial ratios table
  - 5C narrative detail sections
  - Recommendation & Audit trail
"""

from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, FrameBreak, HRFlowable, Image,
    KeepTogether, PageBreak, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)
from reportlab.platypus.flowables import Flowable


# ─────────────────────────────────────────────────────────────────────────────
# Font Registration — use Arial (supports Vietnamese on Windows)
# ─────────────────────────────────────────────────────────────────────────────

_FONT_DIR = "C:/Windows/Fonts"
_FONT_REGISTERED = False

def _register_fonts():
    global _FONT_REGISTERED
    if _FONT_REGISTERED:
        return
    try:
        pdfmetrics.registerFont(TTFont("Arial",    f"{_FONT_DIR}/arial.ttf"))
        pdfmetrics.registerFont(TTFont("Arial-Bold",   f"{_FONT_DIR}/arialbd.ttf"))
        pdfmetrics.registerFont(TTFont("Arial-Italic", f"{_FONT_DIR}/ariali.ttf"))
        from reportlab.pdfbase.pdfmetrics import registerFontFamily
        registerFontFamily("Arial",
            normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-Bold")
        _FONT_REGISTERED = True
    except Exception:
        # Fallback — fonts won't render Vietnamese but at least won't crash
        pass

_register_fonts()
F_NORMAL = "Arial"
F_BOLD   = "Arial-Bold"
F_ITALIC = "Arial-Italic"


# ─────────────────────────────────────────────────────────────────────────────
# Colour Palette
# ─────────────────────────────────────────────────────────────────────────────
C_NAVY    = colors.HexColor("#0D1F3C")
C_BLUE    = colors.HexColor("#1565C0")
C_BLUE_L  = colors.HexColor("#EEF4FF")
C_GREEN   = colors.HexColor("#1B5E20")
C_GREEN_L = colors.HexColor("#E8F5E9")
C_ORANGE  = colors.HexColor("#BF360C")
C_ORANGE_L= colors.HexColor("#FFF3E0")
C_RED     = colors.HexColor("#B71C1C")
C_RED_L   = colors.HexColor("#FFEBEE")
C_GREY    = colors.HexColor("#546E7A")
C_GREY_L  = colors.HexColor("#F5F7FA")
C_BORDER  = colors.HexColor("#CFD8DC")
C_WHITE   = colors.white
C_TEXT    = colors.HexColor("#1A1A2E")
C_SHAP_P  = colors.HexColor("#1565C0")
C_SHAP_N  = colors.HexColor("#C62828")
C_GOLD    = colors.HexColor("#F9A825")

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm
CONTENT_W = PAGE_W - 2 * MARGIN


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib font setup
# ─────────────────────────────────────────────────────────────────────────────

def _setup_mpl_font():
    """Configure matplotlib to use a font that supports Vietnamese."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
    plt.rcParams.update({
        "font.family": ["Arial", "Calibri", "Segoe UI", "DejaVu Sans"],
        "axes.unicode_minus": False,
    })

_setup_mpl_font()


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _band_color(band: str):
    b = (band or "").upper()
    if b.startswith("AA"): return C_GREEN
    if b.startswith("A"):  return C_BLUE
    if b.startswith("BB"): return C_ORANGE
    return C_RED

def _dec_color(dec: str):
    d = (dec or "").upper()
    if "APPROVE" in d: return C_GREEN
    if "REJECT"  in d: return C_RED
    return C_ORANGE

def _fmt_vnd(val):
    if val is None: return "N/A"
    try:
        v = float(val)
        if v >= 1_000_000_000: return f"{v/1_000_000_000:.1f} tỷ VND"
        if v >= 1_000_000:     return f"{v/1_000_000:.1f} triệu VND"
        if v >= 1_000:         return f"{v/1_000:.1f} nghìn VND"
        return f"{v:,.0f} VND"
    except Exception:
        return str(val)


# ─────────────────────────────────────────────────────────────────────────────
# Charts
# ─────────────────────────────────────────────────────────────────────────────

def _shap_chart(shap: dict, width_in=6.5, height_in=3.2) -> io.BytesIO:
    """SHAP horizontal bar chart."""
    pos = shap.get("top_positive_factors", [])[:6]
    neg = shap.get("top_negative_factors", [])[:6]

    items = []
    for f in neg:
        sv = float(f.get("shap_value", f.get("shap", 0)) or 0)
        items.append((f.get("label_vi") or f.get("feature", "?"), sv))
    for f in pos:
        sv = float(f.get("shap_value", f.get("shap", 0)) or 0)
        items.append((f.get("label_vi") or f.get("feature", "?"), sv))

    if not items:
        items = [("No SHAP data", 0.0)]

    items.sort(key=lambda x: x[1])
    labels = [x[0][:45] for x in items]
    values = [x[1] for x in items]
    bar_colors = ["#C62828" if v < 0 else "#1565C0" for v in values]

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor("#FAFBFC")
    ax.set_facecolor("#FAFBFC")

    bars = ax.barh(range(len(labels)), values, color=bar_colors,
                   height=0.55, edgecolor="white", linewidth=0.4)

    for i, (b, v) in enumerate(zip(bars, values)):
        sign = "+" if v >= 0 else ""
        ax.text(v + (0.001 if v >= 0 else -0.001), i,
                f"{sign}{v:.3f}",
                va="center", ha="left" if v >= 0 else "right",
                fontsize=7, color="#333", fontweight="600")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5, color="#2C3E50")
    ax.axvline(0, color="#607080", linewidth=0.7)
    ax.set_xlabel("SHAP Value (log-odds)", fontsize=7.5, color="#607080")
    ax.set_title("Phan tich SHAP — Muc do anh huong cua tung yeu to",
                 fontsize=9, color="#0D1F3C", fontweight="bold", pad=6)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color="#B0BEC5")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    pos_p = mpatches.Patch(color="#1565C0", label="Giam rui ro (tich cuc)")
    neg_p = mpatches.Patch(color="#C62828", label="Tang rui ro (tieu cuc)")
    ax.legend(handles=[pos_p, neg_p], loc="lower right", fontsize=6.5,
              framealpha=0.85, edgecolor="#CFD8DC")

    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="#FAFBFC")
    plt.close(fig)
    buf.seek(0)
    return buf


def _5c_chart(scores: dict, width_in=3.3, height_in=2.2) -> io.BytesIO:
    dims = [
        ("Character",  "character",  30),
        ("Capacity",   "capacity",   40),
        ("Capital",    "capital",    20),
        ("Conditions", "conditions", 10),
        ("Collateral", "collateral", 20),
    ]
    labels  = [d[0] for d in dims]
    vals    = [scores.get(d[1], 0) for d in dims]
    maxs    = [d[2] for d in dims]
    pcts    = [v/m if m else 0 for v, m in zip(vals, maxs)]
    bcolors = ["#1B5E20" if p >= 0.7 else ("#F57F17" if p >= 0.4 else "#C62828")
               for p in pcts]

    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor("#FAFBFC")
    ax.set_facecolor("#FAFBFC")

    ax.barh(range(len(labels)), pcts, color=bcolors, height=0.5,
            edgecolor="white", linewidth=0.4)
    for i, (v, m, p) in enumerate(zip(vals, maxs, pcts)):
        ax.text(min(p + 0.03, 1.18), i, f"{v}/{m}",
                va="center", ha="left", fontsize=7.5, color="#333",
                fontweight="bold")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, color="#2C3E50")
    ax.set_xlim(0, 1.30)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0%", "50%", "100%"], fontsize=7)
    ax.axvline(0.7, color="#90A4AE", linewidth=0.7, linestyle="--", alpha=0.7)
    ax.set_title("5C Score", fontsize=8.5, color="#0D1F3C", fontweight="bold", pad=4)
    ax.grid(axis="x", linestyle="--", alpha=0.25, color="#B0BEC5")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="#FAFBFC")
    plt.close(fig)
    buf.seek(0)
    return buf


# ─────────────────────────────────────────────────────────────────────────────
# Custom Flowables
# ─────────────────────────────────────────────────────────────────────────────

class ScoreCard(Flowable):
    """Coloured scorecard tile."""
    def __init__(self, w, h, bg, label, value, sub, val_size=22):
        super().__init__()
        self.width, self.height = w, h
        self.bg, self.label, self.value, self.sub = bg, label, value, sub
        self.val_size = val_size

    def wrap(self, *a): return self.width, self.height

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg)
        c.roundRect(0, 0, self.width, self.height, 5, fill=1, stroke=0)

        # subtle top shine
        c.setFillColor(colors.HexColor("#FFFFFF"))
        c.setFillAlpha(0.08)
        c.roundRect(0, self.height*0.6, self.width, self.height*0.4, 5, fill=1, stroke=0)
        c.setFillAlpha(1.0)

        # label (top)
        c.setFillColor(colors.HexColor("#B0C8E8"))
        c.setFont(F_BOLD, 6.5)
        c.drawCentredString(self.width/2, self.height - 13, self.label.upper())

        # value (centre)
        c.setFillColor(C_WHITE)
        c.setFont(F_BOLD, self.val_size)
        c.drawCentredString(self.width/2, self.height/2 - self.val_size*0.35, str(self.value))

        # sub (bottom)
        c.setFillColor(colors.HexColor("#90B8D8"))
        c.setFont(F_NORMAL, 6)
        c.drawCentredString(self.width/2, 5, str(self.sub)[:30])


class SHAPBar(Flowable):
    """Single SHAP horizontal bar row."""
    def __init__(self, label, value, max_abs, total_w, positive=True):
        super().__init__()
        self.label   = label[:52]
        self.value   = value
        self.max_abs = max_abs or 0.001
        self.width   = total_w
        self.height  = 20
        self.positive = positive

    def wrap(self, *a): return self.width, self.height

    def draw(self):
        c = self.canv
        LW = 200   # label column width
        VW = 52    # value column width
        BW = self.width - LW - VW - 8  # bar width

        # label — draw simply, truncated
        c.setFont(F_NORMAL, 7.5)
        c.setFillColor(C_TEXT)
        lbl = self.label[:35]  # hard truncate to fit column
        c.drawString(0, 5, lbl)

        # bar bg
        bx = LW
        c.setFillColor(colors.HexColor("#ECEFF1"))
        c.rect(bx, 5, BW, 10, fill=1, stroke=0)

        # bar fill
        pct = min(abs(self.value) / self.max_abs, 1.0)
        bar_color = C_SHAP_P if self.positive else C_SHAP_N
        c.setFillColor(bar_color)
        c.rect(bx, 5, pct * BW, 10, fill=1, stroke=0)

        # value text
        sign = "+" if self.value >= 0 else ""
        c.setFont(F_BOLD, 7)
        c.setFillColor(bar_color)
        c.drawRightString(self.width, 5, f"{sign}{self.value:.4f}")


class SectionTitle(Flowable):
    """Dark section header band."""
    def __init__(self, num, title, w, color=None):
        super().__init__()
        self.num   = num
        self.title = title
        self.width = w
        self.height = 22
        self.color = color or C_NAVY

    def wrap(self, *a): return self.width, self.height

    def draw(self):
        c = self.canv
        # Numero badge
        badge_w = 32
        c.setFillColor(C_BLUE)
        c.roundRect(0, 0, badge_w, self.height, 3, fill=1, stroke=0)
        c.setFillColor(C_WHITE)
        c.setFont(F_BOLD, 8)
        c.drawCentredString(badge_w/2, 6, self.num)

        # Title band
        c.setFillColor(self.color)
        c.rect(badge_w + 1, 0, self.width - badge_w - 1, self.height, fill=1, stroke=0)
        c.setFillColor(C_WHITE)
        c.setFont(F_BOLD, 10)
        c.drawString(badge_w + 10, 6, self.title)


# ─────────────────────────────────────────────────────────────────────────────
# Paragraph Style Builder
# ─────────────────────────────────────────────────────────────────────────────

def _s(name, fn=None, fs=8.5, tc=None, **kw):
    return ParagraphStyle(
        name,
        fontName=fn or F_NORMAL,
        fontSize=fs,
        textColor=tc or C_TEXT,
        leading=kw.pop("leading", max(fs * 1.4, 11)),
        **kw,
    )


STYLES = {
    "body":      _s("body",    leading=13, spaceAfter=3, alignment=TA_JUSTIFY),
    "bodySmall": _s("bodySmall", fs=7.5, tc=C_GREY, leading=10, spaceAfter=2),
    "h3":        _s("h3", fn=F_BOLD, fs=9, tc=C_BLUE, spaceBefore=6, spaceAfter=3, leading=12),
    "label":     _s("label", fn=F_BOLD, fs=7, tc=C_GREY, spaceAfter=1, leading=9),
    "caveat":    _s("caveat", fs=7, tc=C_ORANGE, spaceAfter=1, leading=10, leftIndent=6),
    "footer":    _s("footer", fs=6.5, tc=C_GREY, alignment=TA_CENTER, leading=9),
    "tblHdr":    _s("tblHdr", fn=F_BOLD, fs=7.5, tc=C_WHITE, alignment=TA_CENTER, leading=10),
    "tblCell":   _s("tblCell", fs=7.5, tc=C_TEXT, leading=10),
    "tblBoldL":  _s("tblBoldL", fn=F_BOLD, fs=7.5, tc=C_BLUE, leading=10),
    "dec":       _s("dec", fn=F_BOLD, fs=14, tc=C_WHITE, alignment=TA_CENTER, leading=18),
    "narrative": _s("narrative", fs=8, tc=C_TEXT, leading=12, spaceAfter=3, alignment=TA_JUSTIFY),
}

def P(text, style="body"):
    return Paragraph(str(text), STYLES[style])


# ─────────────────────────────────────────────────────────────────────────────
# Table helpers
# ─────────────────────────────────────────────────────────────────────────────

def _base_tbl_style(header_rows=1):
    s = [
        ("BACKGROUND",    (0, 0), (-1, header_rows - 1), C_NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, header_rows - 1), C_WHITE),
        ("FONTNAME",      (0, 0), (-1, header_rows - 1), F_BOLD),
        ("FONTSIZE",      (0, 0), (-1, header_rows - 1), 8),
        ("ROWBACKGROUNDS",(0, header_rows), (-1, -1), [C_WHITE, C_GREY_L]),
        ("FONTNAME",      (0, header_rows), (-1, -1), F_NORMAL),
        ("FONTSIZE",      (0, header_rows), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.3, C_BORDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]
    return TableStyle(s)


# ─────────────────────────────────────────────────────────────────────────────
# PDF Builder
# ─────────────────────────────────────────────────────────────────────────────

class CreditReportPDF:

    def __init__(self, report: dict, shap: dict, customer_name: str = "Khach hang"):
        self.report = report
        self.shap   = shap
        self.name   = customer_name
        self.story: list = []

    def _sp(self, h=4): return Spacer(1, h * mm)

    # ── Page callback ─────────────────────────────────────────────────────────
    def _on_page(self, canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(C_BORDER)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, 1.2 * cm, PAGE_W - MARGIN, 1.2 * cm)
        canvas.setFont(F_NORMAL, 6.5)
        canvas.setFillColor(C_GREY)
        canvas.drawString(MARGIN, 0.75 * cm,
            "CreditLens AI  |  To trinh tin dung  |  TT39/2016/TT-NHNN")
        canvas.drawRightString(PAGE_W - MARGIN, 0.75 * cm, f"Trang {doc.page}")
        canvas.restoreState()

    # ── HEADER ────────────────────────────────────────────────────────────────
    def _header(self):
        es  = self.report.get("executive_summary", {})
        mi  = es.get("model_info", {})
        score = es.get("credit_score", "—")
        band  = es.get("risk_band", "—")
        pd    = es.get("pd_pct", "—")
        rec   = es.get("recommendation", "REVIEW")
        ts    = mi.get("inference_timestamp", "")
        if ts:
            try: ts = datetime.fromisoformat(ts).strftime("%d/%m/%Y %H:%M UTC")
            except: pass

        # ── Top banner ────────────────────────────────────────────────────────
        banner = Table([[
            Paragraph(f"<b>CreditLens AI</b>",
                       _s("b1", fn=F_BOLD, fs=15, tc=C_WHITE)),
            Paragraph(
                "<b>TO TRINH TIN DUNG</b><br/>"
                "<font size='8'>Tham dinh Creditworthiness · AI-Assisted</font>",
                _s("b2", fn=F_BOLD, fs=13, tc=C_WHITE, alignment=TA_CENTER, leading=17)),
            Paragraph(
                f"<font size='7'>Ngay: {ts or datetime.now().strftime('%d/%m/%Y')}</font>",
                _s("b3", fs=7, tc=colors.HexColor("#90CAF9"),
                   alignment=TA_RIGHT, leading=10)),
        ]], colWidths=[5.5*cm, 9.5*cm, 3.5*cm])
        banner.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), C_NAVY),
            ("TOPPADDING",    (0,0), (-1,-1), 12),
            ("BOTTOMPADDING", (0,0), (-1,-1), 12),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("RIGHTPADDING",  (0,0), (-1,-1), 10),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("LINEBELOW",     (0,0), (-1,-1), 3, C_BLUE),
        ]))
        self.story.append(banner)
        self.story.append(self._sp(5))

        # ── Score tiles ───────────────────────────────────────────────────────
        tile_w = (CONTENT_W - 3 * 3 * mm) / 4
        tile_h = 2.6 * cm

        rec_vi = {
            "APPROVE": "PHE DUYET", "APPROVE_REVIEW": "DUYET XEM XET",
            "REJECT": "TU CHOI", "REVIEW": "XEM XET",
            "ESCALATE": "LEO THANG", "CONDITIONAL": "CO DIEU KIEN",
        }.get((rec or "").upper(), rec or "REVIEW")

        pd_colr = C_RED if float(pd or 0) > 35 else (C_ORANGE if float(pd or 0) > 8 else C_GREEN)

        tiles = Table([[
            ScoreCard(tile_w, tile_h, C_BLUE,
                      "DIEM TIN DUNG", f"{score}/850", f"{band} — Risk Band"),
            ScoreCard(tile_w, tile_h, _dec_color(rec),
                      "DE XUAT", rec_vi[:13], "Quyet dinh so bo"),
            ScoreCard(tile_w, tile_h, pd_colr,
                      "XAC SUAT VO NO", f"{pd}%", "Probability of Default"),
            ScoreCard(tile_w, tile_h, C_NAVY,
                      "MO HINH", mi.get("model_version","lgbm_v1")[:12],
                      f"AUC {mi.get('auc','0.803')} · SHAP OK"),
        ]], colWidths=[tile_w]*4, rowHeights=[tile_h])
        tiles.setStyle(TableStyle([
            ("ALIGN",     (0,0), (-1,-1), "CENTER"),
            ("COLPADDING",(0,0), (-1,-1), 3),
            ("VALIGN",    (0,0), (-1,-1), "MIDDLE"),
        ]))
        self.story.append(tiles)
        self.story.append(self._sp(5))

    # ── CUSTOMER INFO ─────────────────────────────────────────────────────────
    def _customer(self):
        ci = self.report.get("customer_info", {})
        self.story.append(SectionTitle("I", "THONG TIN KHACH HANG", CONTENT_W))
        self.story.append(self._sp(2))
        summary = ci.get("summary") or f"Khach hang: {self.name}"
        t = Table([["Tom tat ho so", summary]],
                   colWidths=[3.5*cm, CONTENT_W - 3.5*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (0,-1), C_BLUE_L),
            ("FONTNAME",    (0,0), (0,-1), F_BOLD),
            ("FONTSIZE",    (0,0), (-1,-1), 8),
            ("FONTNAME",    (1,0), (1,-1), F_NORMAL),
            ("TEXTCOLOR",   (0,0), (0,-1), C_BLUE),
            ("GRID",        (0,0), (-1,-1), 0.3, C_BORDER),
            ("TOPPADDING",  (0,0), (-1,-1), 7),
            ("BOTTOMPADDING",(0,0),(-1,-1), 7),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("VALIGN",      (0,0), (-1,-1), "TOP"),
            ("FONTNAME", (1,0), (1,-1), F_NORMAL),
        ]))
        self.story.append(t)
        self.story.append(self._sp(4))

    # ── SHAP SECTION ──────────────────────────────────────────────────────────
    def _shap(self):
        self.story.append(SectionTitle("II", "PHAN TICH SHAP — YEU TO ANH HUONG", CONTENT_W))
        self.story.append(self._sp(2))
        self.story.append(P(
            "Bieu do SHAP the hien dong gop cua tung yeu to du lieu. "
            "<b>Xanh</b> = giam rui ro vo no. <b>Do</b> = tang rui ro."
        ))
        self.story.append(self._sp(2))

        # Chart
        buf = _shap_chart(self.shap)
        img = Image(buf, width=CONTENT_W, height=8*cm)
        self.story.append(img)
        self.story.append(self._sp(3))

        # SHAP bar rows
        pos = (self.shap.get("top_positive_factors") or [])[:5]
        neg = (self.shap.get("top_negative_factors") or [])[:5]
        all_sv = pos + neg
        max_abs = max((abs(float(f.get("shap_value", f.get("shap", 0)) or 0))
                       for f in all_sv), default=0.001)

        if pos:
            self.story.append(P("<b>Yeu to tich cuc (giam rui ro vo no):</b>", "h3"))
            for f in pos:
                sv = float(f.get("shap_value", f.get("shap", 0)) or 0)
                lbl = f.get("label_vi") or f.get("feature", "—")
                self.story.append(SHAPBar(lbl, sv, max_abs, CONTENT_W, positive=True))
                self.story.append(self._sp(1))

        if neg:
            self.story.append(self._sp(2))
            self.story.append(P("<b>Yeu to rui ro (tang xac suat vo no):</b>", "h3"))
            for f in neg:
                sv = float(f.get("shap_value", f.get("shap", 0)) or 0)
                lbl = f.get("label_vi") or f.get("feature", "—")
                self.story.append(SHAPBar(lbl, sv, max_abs, CONTENT_W, positive=False))
                self.story.append(self._sp(1))

        # 5C allocation table
        alloc = (self.report.get("executive_summary", {})
                             .get("five_c_shap_allocation", {}))
        if alloc:
            self.story.append(self._sp(3))
            self.story.append(P("<b>Phan bo SHAP theo 5C:</b>", "h3"))
            lmap = {"character":"C1 Character","capacity":"C2 Capacity",
                    "capital":"C3 Capital","conditions":"C4 Conditions",
                    "collateral":"C5 Collateral"}
            rows = [["Tieu chi", "SHAP Sum", "Ty trong"]]
            for k, v in alloc.items():
                rows.append([lmap.get(k, k),
                              f"{v.get('shap_sum',0):.4f}",
                              f"{v.get('pct',0)}%"])
            t = Table(rows, colWidths=[9*cm, 4*cm, 3.5*cm])
            t.setStyle(_base_tbl_style())
            t.setStyle(TableStyle([
                *_base_tbl_style().getCommands(),
                ("ALIGN", (1,0), (-1,-1), "CENTER"),
            ]))
            self.story.append(t)
        self.story.append(self._sp(4))

    # ── 5C SCORECARD ──────────────────────────────────────────────────────────
    def _5c(self):
        self.story.append(PageBreak())
        self.story.append(SectionTitle("III", "DANH GIA 5C CHI TIET", CONTENT_W))
        self.story.append(self._sp(3))

        fiveC  = self.report.get("five_c_scorecard", {})
        scores = self.report.get("executive_summary", {}).get("five_c_scores", {})
        total  = self.report.get("executive_summary", {}).get("five_c_total", "—")

        # Chart + summary table side by side
        chart_buf = _5c_chart(scores)
        chart_img = Image(chart_buf, width=8*cm, height=5.5*cm)

        def _status(val, mx):
            p = val / mx if mx else 0
            if p >= 0.7: return "DAT"
            if p >= 0.4: return "XEM XET"
            return "CHUA DAT"

        srows = [["Tieu chi", "Diem", "Max", "Trang thai"]]
        for d in [("C1 Character","character",30), ("C2 Capacity","capacity",40),
                  ("C3 Capital","capital",20), ("C4 Conditions","conditions",10),
                  ("C5 Collateral","collateral",20)]:
            v = scores.get(d[1], 0)
            srows.append([d[0], str(v), str(d[2]), _status(v, d[2])])
        srows.append(["TONG", str(total), "120", ""])

        stbl = Table(srows, colWidths=[4.5*cm, 1.8*cm, 1.8*cm, 3.2*cm])
        stbl_style = TableStyle([
            ("BACKGROUND",   (0,0), (-1,0), C_NAVY),
            ("FONTNAME",     (0,0), (-1,0), F_BOLD),
            ("FONTSIZE",     (0,0), (-1,0), 7.5),
            ("TEXTCOLOR",    (0,0), (-1,0), C_WHITE),
            ("FONTNAME",     (0,1), (-1,-2), F_NORMAL),
            ("FONTSIZE",     (0,1), (-1,-2), 8),
            ("FONTNAME",     (0,-1), (-1,-1), F_BOLD),
            ("FONTSIZE",     (0,-1), (-1,-1), 8.5),
            ("BACKGROUND",   (0,-1), (-1,-1), C_BLUE_L),
            ("TEXTCOLOR",    (0,-1), (-1,-1), C_BLUE),
            ("ROWBACKGROUNDS",(0,1), (-1,-2), [C_WHITE, C_GREY_L]),
            ("GRID",         (0,0), (-1,-1), 0.3, C_BORDER),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("ALIGN",        (1,0), (2,-1), "CENTER"),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
        ])
        stbl.setStyle(stbl_style)

        side = Table([[chart_img, stbl]], colWidths=[8.5*cm, None])
        side.setStyle(TableStyle([
            ("VALIGN",      (0,0), (-1,-1), "TOP"),
            ("COLPADDING",  (0,0), (-1,-1), 5),
        ]))
        self.story.append(side)
        self.story.append(self._sp(5))

        # ── 5C Narratives ────────────────────────────────────────────────────
        DIM_CFG = [
            ("character_assessment",  "C1 — Character (Uy tin / Tu cach)",   "character",  30, C_BLUE),
            ("capacity_assessment",   "C2 — Capacity (Nang luc tra no)",      "capacity",   40, C_GREEN),
            ("capital_assessment",    "C3 — Capital (Von tu co)",              "capital",    20, C_BLUE),
            ("conditions_assessment", "C4 — Conditions (Dieu kien vay)",      "conditions", 10, C_ORANGE),
            ("collateral_assessment", "C5 — Collateral (Tai san bao dam)",    "collateral", 20, C_GREY),
        ]
        STATUS_CLR = {"DAT": C_GREEN, "XEM_XET": C_ORANGE, "KHONG_DAT": C_RED}

        for key, title, sk, smax, color in DIM_CFG:
            asmnt = fiveC.get(key) or {}
            sv    = scores.get(sk, 0)
            status = asmnt.get("status", "—")
            shap_pct = asmnt.get("shap_pct", "—")
            narrative = asmnt.get("narrative", "Khong co du lieu.")
            ind_met = asmnt.get("indicators_met", [])
            ind_rev = asmnt.get("indicators_review", [])
            st_clr = STATUS_CLR.get(status, C_GREY)

            # Sub-header
            sub = Table([[
                Paragraph(f"<b>{title}</b>",
                           _s("st", fn=F_BOLD, fs=9, tc=color, leading=12)),
                Paragraph(f"<b>{sv}/{smax}</b>",
                           _s("sv", fn=F_BOLD, fs=13, tc=color,
                              alignment=TA_CENTER, leading=16)),
                Paragraph(status.replace("_", " "),
                           _s("ss", fn=F_BOLD, fs=8, tc=st_clr,
                              alignment=TA_CENTER, leading=10)),
                Paragraph(f"SHAP: {shap_pct}",
                           _s("sp", fs=7.5, tc=C_GREY,
                              alignment=TA_RIGHT, leading=10)),
            ]], colWidths=[8*cm, 2.5*cm, 3*cm, 3*cm])
            sub.setStyle(TableStyle([
                ("BACKGROUND",   (0,0), (-1,-1), C_GREY_L),
                ("LINEBELOW",    (0,0), (-1,-1), 2, color),
                ("TOPPADDING",   (0,0), (-1,-1), 6),
                ("BOTTOMPADDING",(0,0), (-1,-1), 6),
                ("LEFTPADDING",  (0,0), (-1,-1), 6),
                ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ]))
            self.story.append(sub)

            # Indicators
            for ind_list, icon, clr in [
                (ind_met, "OK - ", C_GREEN),
                (ind_rev, "!! - ", C_ORANGE),
            ]:
                if ind_list:
                    txt = "; ".join(ind_list)
                    row = Table([[Paragraph(f"{icon}{txt}",
                                   _s("ind", fs=7.5, tc=clr, leading=10))]],
                                colWidths=[CONTENT_W])
                    row.setStyle(TableStyle([
                        ("BACKGROUND",   (0,0), (-1,-1), C_WHITE),
                        ("TOPPADDING",   (0,0), (-1,-1), 3),
                        ("BOTTOMPADDING",(0,0), (-1,-1), 3),
                        ("LEFTPADDING",  (0,0), (-1,-1), 8),
                        ("LINEBELOW",    (0,0), (-1,-1), 0.3, C_BORDER),
                    ]))
                    self.story.append(row)

            # Narrative
            self.story.append(Paragraph(narrative, STYLES["narrative"]))
            self.story.append(self._sp(4))

    # ── FINANCIAL ─────────────────────────────────────────────────────────────
    def _financial(self):
        fs = self.report.get("financial_summary", {})
        fr = self.report.get("executive_summary", {}).get("financial_ratios", {})
        self.story.append(SectionTitle("IV", "TINH HINH TAI CHINH", CONTENT_W))
        self.story.append(self._sp(2))

        kr = fs.get("key_ratios", {})
        dti  = str(kr.get("dti")  or fr.get("dti_pct",  "N/A"))
        dscr = str(kr.get("dscr") or fr.get("dscr",     "N/A"))
        ltv  = str(kr.get("ltv")  or fr.get("ltv_pct",  "N/A"))

        def _ev(v, thr, lo, hi, invert=False):
            try:
                n = float(str(v).replace("%",""))
                good = (n < thr) if not invert else (n > thr)
                return "OK" if good else "!!"
            except: return "—"

        rows = [
            ["Chi so", "Gia tri", "Nguong tot", "Danh gia"],
            ["DTI (No/Thu nhap)", dti, "< 40%", _ev(dti, 40, 0, 100)],
            ["DSCR (Dong tien/No)", dscr, "> 1.20", _ev(dscr, 1.2, 0, 10, invert=True)],
            ["LTV (Khoan vay/TSBD)", ltv, "< 80%", _ev(ltv, 80, 0, 100)],
        ]
        # Income rows
        for key, label in [
            ("income_monthly_vnd", "Thu nhap thang"),
            ("annuity_monthly_vnd", "Tra no thang"),
            ("credit_total_vnd", "Tong khoan vay"),
        ]:
            v = fr.get(key)
            if v: rows.append([label, _fmt_vnd(v), "—", "—"])

        cw = CONTENT_W / 4
        t = Table(rows, colWidths=[cw]*4)
        t.setStyle(TableStyle([
            *_base_tbl_style().getCommands(),
            ("ALIGN", (1,0), (-1,-1), "CENTER"),
            ("FONTNAME", (0,1), (0,-1), F_BOLD),
        ]))
        self.story.append(t)
        self.story.append(self._sp(3))

        for key, label in [("income_analysis","Phan tich thu nhap"),
                            ("debt_analysis","Phan tich no")]:
            txt = fs.get(key, "")
            if txt:
                self.story.append(P(f"<b>{label}:</b> {txt}"))
        self.story.append(self._sp(4))

    # ── COLLATERAL ────────────────────────────────────────────────────────────
    def _collateral(self):
        cd = self.report.get("collateral_detail", {})
        self.story.append(SectionTitle("V", "TAI SAN BAO DAM", CONTENT_W))
        self.story.append(self._sp(2))
        if cd.get("indicators_met"):
            self.story.append(P("OK - " + "; ".join(cd["indicators_met"])))
        if cd.get("indicators_review"):
            self.story.append(P("!! - " + "; ".join(cd["indicators_review"])))
        self.story.append(P(cd.get("narrative",
                "Chua co thong tin tai san bao dam chi tiet.")))
        self.story.append(self._sp(4))

    # ── RECOMMENDATION ────────────────────────────────────────────────────────
    def _recommendation(self):
        self.story.append(SectionTitle("VI", "KHUYEN NGHI & DIEU KIEN", CONTENT_W))
        self.story.append(self._sp(3))

        rec   = self.report.get("executive_summary", {}).get("recommendation", "—")
        st    = self.report.get("suggested_terms", {})
        caveats = list(dict.fromkeys(self.report.get("caveats", [])))[:10]
        audit   = self.report.get("audit_reference", {})
        llm_i   = self.report.get("llm_insights", {})

        # Decision band
        dec_vi = {
            "APPROVE":"PHE DUYET","APPROVE_REVIEW":"DUYET + XEM XET",
            "REJECT":"TU CHOI","REVIEW":"XEM XET",
        }.get((rec or "").upper(), rec or "XEM XET")

        dec_t = Table([[Paragraph(f"<b>DE XUAT: {dec_vi}</b>", STYLES["dec"])]],
                      colWidths=[CONTENT_W])
        dec_t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), _dec_color(rec)),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("ROUNDEDCORNERS", [5, 5, 5, 5]),
        ]))
        self.story.append(dec_t)
        self.story.append(self._sp(4))

        # Terms table
        tdata = [["Muc", "Chi tiet"]]
        if st.get("max_amount_vnd"):
            tdata.append(["So tien de xuat", _fmt_vnd(st["max_amount_vnd"])])
        if st.get("requested_term_months"):
            tdata.append(["Ky han", f"{st['requested_term_months']} thang"])
        tdata.append(["Lai suat", st.get("interest_rate_suggestion", "Theo bieu phi")])
        if st.get("dti_at_approval"):
            tdata.append(["DTI tai thoi diem duyet", str(st["dti_at_approval"])])
        conds = st.get("conditions", [])
        if conds:
            tdata.append(["Dieu kien tien quyet",
                           "\n".join(f"• {c}" for c in conds)])

        tt = Table(tdata, colWidths=[4.5*cm, CONTENT_W - 4.5*cm])
        tt.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (0,-1), C_BLUE_L),
            ("FONTNAME",    (0,0), (0,-1), F_BOLD),
            ("FONTSIZE",    (0,0), (-1,-1), 8),
            ("FONTNAME",    (1,0), (1,-1), F_NORMAL),
            ("TEXTCOLOR",   (0,0), (0,-1), C_BLUE),
            ("GRID",        (0,0), (-1,-1), 0.3, C_BORDER),
            ("TOPPADDING",  (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1), 6),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
            ("VALIGN",      (0,0), (-1,-1), "TOP"),
            ("ROWBACKGROUNDS",(0,0),(-1,-1), [C_WHITE, C_GREY_L]),
        ]))
        self.story.append(tt)
        self.story.append(self._sp(3))

        # LLM insights
        for sig in llm_i.get("positive_signals", []):
            self.story.append(P(f"OK  {sig}", "bodySmall"))
        for flag in llm_i.get("risk_flags", []):
            self.story.append(P(f"!!  {flag}", "caveat"))
        if caveats:
            self.story.append(self._sp(2))
            self.story.append(P("<b>Canh bao du lieu (Imputation Log):</b>", "h3"))
            for c in caveats:
                self.story.append(P(f"• {c}", "caveat"))
        self.story.append(self._sp(3))

        # Audit line
        self.story.append(HRFlowable(width=CONTENT_W, thickness=0.4, color=C_BORDER))
        self.story.append(self._sp(2))
        mv = audit.get("model_version", "—")
        ts = audit.get("inference_timestamp", "—")
        self.story.append(P(
            f"<i>Audit — Model: {mv} · Timestamp: {ts} · "
            "SHAP hash: verified · TT39/2016 · QD493/2005</i>",
            "bodySmall"
        ))

    # ── BUILD ─────────────────────────────────────────────────────────────────
    def build(self) -> bytes:
        buf = io.BytesIO()
        doc = BaseDocTemplate(
            buf, pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=1.8*cm,
        )
        frame = Frame(MARGIN, 1.8*cm, CONTENT_W, PAGE_H - MARGIN - 1.8*cm, id="main")
        doc.addPageTemplates([PageTemplate(
            id="main", frames=[frame], onPage=self._on_page
        )])

        self._header()
        self._customer()
        self._shap()
        self._5c()
        self._financial()
        self._collateral()
        self._recommendation()

        doc.build(self.story)
        return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_credit_pdf(
    report_data: dict,
    shap_data: dict | None = None,
    customer_name: str = "Khach hang",
) -> bytes:
    """Generate professional credit report PDF."""
    return CreditReportPDF(
        report=report_data,
        shap=shap_data or {},
        customer_name=customer_name,
    ).build()
