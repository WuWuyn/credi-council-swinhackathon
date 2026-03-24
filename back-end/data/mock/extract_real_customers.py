"""
Extract 50 real customers from the Home Credit dataset for demo purposes.

Selects 25 customers with TARGET=0 (repaid OK) and 25 with TARGET=1 (defaulted),
choosing those with rich data across all related tables.

Outputs per-customer:
  - Full feature dict (application_row.json)
  - CIC API JSON (07_cic_api_response.json)
  - Internal DB JSON (08_internal_db.json)
  - PDFs (01-05) via generate_pdfs_for_customer()

Usage:
    conda activate swinburn_hackathon
    cd d:\\project\\swinburn_new
    python data/mock/extract_real_customers.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "home-credit-default-risk"
OUTPUT_DIR = Path(__file__).resolve().parent  # data/mock/

# Minimum records in related tables to qualify as "rich" customer
MIN_BUREAU = 2
MIN_PREV_APP = 1
MIN_POS = 3
MIN_INST = 3


def load_tables():
    """Load all 7 tables with minimal columns for selection."""
    print("Loading tables...")
    app = pd.read_csv(DATA_DIR / "application_train.csv")
    print(f"  application_train: {len(app)} rows, {app.shape[1]} cols")

    bureau = pd.read_csv(DATA_DIR / "bureau.csv")
    print(f"  bureau: {len(bureau)} rows")

    bb = pd.read_csv(DATA_DIR / "bureau_balance.csv")
    print(f"  bureau_balance: {len(bb)} rows")

    prev = pd.read_csv(DATA_DIR / "previous_application.csv")
    print(f"  previous_application: {len(prev)} rows")

    pos = pd.read_csv(DATA_DIR / "POS_CASH_balance.csv")
    print(f"  POS_CASH_balance: {len(pos)} rows")

    inst = pd.read_csv(DATA_DIR / "installments_payments.csv")
    print(f"  installments_payments: {len(inst)} rows")

    cc = pd.read_csv(DATA_DIR / "credit_card_balance.csv")
    print(f"  credit_card_balance: {len(cc)} rows")

    return app, bureau, bb, prev, pos, inst, cc


def select_customers(app, bureau, prev, pos, inst, cc, n_pass=25, n_fail=25):
    """Select 50 customers with diverse model scores for demo.

    Strategy:
    1. Filter for customers with rich data across all tables
    2. Compute proxy PD score from EXT_SOURCE features
    3. Pick n_pass from TARGET=0 (spread across PD range)
    4. Pick n_fail from TARGET=1 (spread across PD range)
    This gives a diverse, representative demo set.
    """
    import sys
    import pickle
    import numpy as np

    print("\nSelecting customers with extreme model scores...")

    # Count records per customer in each table
    bureau_cnt = bureau.groupby("SK_ID_CURR").size().rename("bureau_cnt")
    prev_cnt = prev.groupby("SK_ID_CURR").size().rename("prev_cnt")
    pos_cnt = pos.groupby("SK_ID_CURR").size().rename("pos_cnt")
    inst_cnt = inst.groupby("SK_ID_CURR").size().rename("inst_cnt")
    cc_cnt = cc.groupby("SK_ID_CURR").size().rename("cc_cnt")

    # Merge counts into app
    counts = app[["SK_ID_CURR", "TARGET"]].copy()
    for cnt_series in [bureau_cnt, prev_cnt, pos_cnt, inst_cnt, cc_cnt]:
        counts = counts.merge(cnt_series, on="SK_ID_CURR", how="left")
    counts = counts.fillna(0)

    # Filter: rich data across all tables + all EXT_SOURCE present
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    app_ext = app[["SK_ID_CURR"] + ext_cols]
    counts = counts.merge(app_ext, on="SK_ID_CURR")
    counts["has_all_ext"] = counts[ext_cols].notna().all(axis=1)

    rich = counts[
        (counts["bureau_cnt"] >= MIN_BUREAU) &
        (counts["prev_cnt"] >= MIN_PREV_APP) &
        (counts["pos_cnt"] >= MIN_POS) &
        (counts["inst_cnt"] >= MIN_INST) &
        (counts["has_all_ext"] == True)
    ]

    rich_ids = rich["SK_ID_CURR"].values
    print(f"  {len(rich_ids)} candidates with rich data + all EXT_SOURCE")

    # --- Quick model scoring using only EXT_SOURCE features (fast proxy) ---
    # EXT_SOURCE_2 is the single most predictive feature (SHAP rank #1).
    # We use a composite proxy score as a fast stand-in for the full model.
    rich_app = app[app["SK_ID_CURR"].isin(rich_ids)][
        ["SK_ID_CURR", "TARGET"] + ext_cols
    ].copy()

    # Composite EXT score: lower = higher default risk (proxy for PD)
    # Weighted same as EXT_SOURCE SHAP importance: EXT2 > EXT3 > EXT1
    rich_app["proxy_pd"] = (
        1.0
        - (rich_app["EXT_SOURCE_2"] * 0.5
           + rich_app["EXT_SOURCE_3"] * 0.3
           + rich_app["EXT_SOURCE_1"] * 0.2)
    )

    # Select n_pass from TARGET=0 (evenly spread across PD range)
    pass_pool = rich_app[rich_app["TARGET"] == 0].sort_values("proxy_pd")
    if len(pass_pool) >= n_pass:
        # Sample evenly across the PD range for diversity
        indices = np.linspace(0, len(pass_pool) - 1, n_pass, dtype=int)
        pass_df = pass_pool.iloc[indices]
    else:
        pass_df = pass_pool.head(n_pass)

    # Select n_fail from TARGET=1 (evenly spread across PD range)
    fail_pool = rich_app[rich_app["TARGET"] == 1].sort_values("proxy_pd", ascending=False)
    if len(fail_pool) >= n_fail:
        indices = np.linspace(0, len(fail_pool) - 1, n_fail, dtype=int)
        fail_df = fail_pool.iloc[indices]
    else:
        fail_df = fail_pool.head(n_fail)

    selected = pd.concat([pass_df, fail_df])
    print(f"\nSelected {len(selected)} customers:")
    for _, row in selected.iterrows():
        label = "PASS" if row["TARGET"] == 0 else "FAIL"
        print(f"  SK_ID_CURR={int(row['SK_ID_CURR'])}, TARGET={int(row['TARGET'])} ({label}), "
              f"proxy_pd={row['proxy_pd']:.3f}, "
              f"EXT=[{row['EXT_SOURCE_1']:.3f}, {row['EXT_SOURCE_2']:.3f}, {row['EXT_SOURCE_3']:.3f}]")

    return selected["SK_ID_CURR"].tolist()


def extract_customer_data(sk_id, app, bureau, bb, prev, pos, inst, cc):
    """Extract all data for a single customer."""
    # Application row
    app_row = app[app["SK_ID_CURR"] == sk_id].iloc[0].to_dict()

    # Replace NaN with None for JSON serialization
    for k, v in app_row.items():
        if isinstance(v, float) and np.isnan(v):
            app_row[k] = None

    # Bureau records + balance
    cust_bureau = bureau[bureau["SK_ID_CURR"] == sk_id]
    bureau_ids = cust_bureau["SK_ID_BUREAU"].unique()
    cust_bb = bb[bb["SK_ID_BUREAU"].isin(bureau_ids)]

    # Previous applications + related
    cust_prev = prev[prev["SK_ID_CURR"] == sk_id]
    prev_ids = cust_prev["SK_ID_PREV"].unique()
    cust_pos = pos[pos["SK_ID_CURR"] == sk_id]
    cust_inst = inst[inst["SK_ID_CURR"] == sk_id]
    cust_cc = cc[cc["SK_ID_CURR"] == sk_id]

    return {
        "application_row": app_row,
        "bureau": df_to_records(cust_bureau),
        "bureau_balance": df_to_records(cust_bb),
        "previous_application": df_to_records(cust_prev),
        "pos_cash_balance": df_to_records(cust_pos),
        "installments_payments": df_to_records(cust_inst),
        "credit_card_balance": df_to_records(cust_cc),
    }


def df_to_records(df):
    """Convert DataFrame to list of dicts, replacing NaN with None."""
    records = df.to_dict(orient="records")
    for rec in records:
        for k, v in rec.items():
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                rec[k] = None
    return records


def build_cic_json(data):
    """Build CIC API response JSON from extracted data."""
    app = data["application_row"]
    bureau_records = []

    for rec in data["bureau"]:
        # Get monthly statuses for this bureau record
        sk_bureau = rec.get("SK_ID_BUREAU")
        statuses = [
            bb for bb in data["bureau_balance"]
            if bb.get("SK_ID_BUREAU") == sk_bureau
        ]
        # Sort by MONTHS_BALANCE descending (0 = current, -1 = last month)
        statuses.sort(key=lambda x: x.get("MONTHS_BALANCE", 0), reverse=True)

        bureau_records.append({
            "SK_ID_BUREAU": sk_bureau,
            "CREDIT_ACTIVE": rec.get("CREDIT_ACTIVE"),
            "CREDIT_CURRENCY": rec.get("CREDIT_CURRENCY"),
            "DAYS_CREDIT": rec.get("DAYS_CREDIT"),
            "CREDIT_DAY_OVERDUE": rec.get("CREDIT_DAY_OVERDUE", 0),
            "DAYS_CREDIT_ENDDATE": rec.get("DAYS_CREDIT_ENDDATE"),
            "DAYS_ENDDATE_FACT": rec.get("DAYS_ENDDATE_FACT"),
            "AMT_CREDIT_MAX_OVERDUE": rec.get("AMT_CREDIT_MAX_OVERDUE", 0),
            "CNT_CREDIT_PROLONG": rec.get("CNT_CREDIT_PROLONG", 0),
            "AMT_CREDIT_SUM": rec.get("AMT_CREDIT_SUM"),
            "AMT_CREDIT_SUM_DEBT": rec.get("AMT_CREDIT_SUM_DEBT", 0),
            "AMT_CREDIT_SUM_LIMIT": rec.get("AMT_CREDIT_SUM_LIMIT", 0),
            "AMT_CREDIT_SUM_OVERDUE": rec.get("AMT_CREDIT_SUM_OVERDUE", 0),
            "CREDIT_TYPE": rec.get("CREDIT_TYPE"),
            "DAYS_CREDIT_UPDATE": rec.get("DAYS_CREDIT_UPDATE"),
            "AMT_ANNUITY": rec.get("AMT_ANNUITY"),
            "monthly_status": [
                {"MONTHS_BALANCE": s.get("MONTHS_BALANCE"), "STATUS": s.get("STATUS")}
                for s in statuses
            ]
        })

    # Determine thin_file flag
    has_bureau = len(bureau_records) > 0
    has_ext = all(app.get(f"EXT_SOURCE_{i}") is not None for i in [1, 2, 3])

    cic = {
        "api_version": "CIC-VN-v2.1",
        "query_timestamp": "2026-03-15T10:23:41+07:00",
        "customer_id": str(int(app["SK_ID_CURR"])),
        "ext_source_scores": {
            "EXT_SOURCE_1": app.get("EXT_SOURCE_1"),
            "EXT_SOURCE_2": app.get("EXT_SOURCE_2"),
            "EXT_SOURCE_3": app.get("EXT_SOURCE_3"),
        },
        "credit_inquiry_counts": {
            "AMT_REQ_CREDIT_BUREAU_HOUR": app.get("AMT_REQ_CREDIT_BUREAU_HOUR", 0),
            "AMT_REQ_CREDIT_BUREAU_DAY": app.get("AMT_REQ_CREDIT_BUREAU_DAY", 0),
            "AMT_REQ_CREDIT_BUREAU_WEEK": app.get("AMT_REQ_CREDIT_BUREAU_WEEK", 0),
            "AMT_REQ_CREDIT_BUREAU_MON": app.get("AMT_REQ_CREDIT_BUREAU_MON", 0),
            "AMT_REQ_CREDIT_BUREAU_QRT": app.get("AMT_REQ_CREDIT_BUREAU_QRT", 0),
            "AMT_REQ_CREDIT_BUREAU_YEAR": app.get("AMT_REQ_CREDIT_BUREAU_YEAR", 0),
        },
        "social_circle": {
            "OBS_30_CNT_SOCIAL_CIRCLE": app.get("OBS_30_CNT_SOCIAL_CIRCLE", 0),
            "DEF_30_CNT_SOCIAL_CIRCLE": app.get("DEF_30_CNT_SOCIAL_CIRCLE", 0),
            "OBS_60_CNT_SOCIAL_CIRCLE": app.get("OBS_60_CNT_SOCIAL_CIRCLE", 0),
            "DEF_60_CNT_SOCIAL_CIRCLE": app.get("DEF_60_CNT_SOCIAL_CIRCLE", 0),
        },
        "bureau_records": bureau_records,
        "thin_file_flag": not (has_bureau and has_ext),
    }

    return cic


def build_internal_db_json(data):
    """Build Internal DB JSON from extracted data."""
    sk_id = int(data["application_row"]["SK_ID_CURR"])

    return {
        "SK_ID_CURR": sk_id,
        "previous_applications": data["previous_application"],
        "pos_cash_balance": data["pos_cash_balance"],
        "installments_payments": data["installments_payments"],
        "credit_card_balance": data["credit_card_balance"],
    }


def generate_pdfs_for_customer(data, customer_dir):
    """Generate 5 PDF documents from real dataset values.

    Maps Home Credit columns back to Vietnamese document fields.
    """
    from datetime import date, timedelta

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.pdfgen import canvas
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        print("  WARNING: reportlab not installed. Skipping PDF generation.")
        return

    # Register Vietnamese font
    FONT = "Helvetica"
    for font_path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont("VNFont", font_path))
                FONT = "VNFont"
            except Exception:
                pass
            break

    app = data["application_row"]
    sk_id = int(app["SK_ID_CURR"])
    today = date.today()

    # ── Helper functions ──
    def days_to_date(days_val):
        """Convert DAYS_* (negative days from today) to date string."""
        if days_val is None:
            return "N/A"
        return (today + timedelta(days=int(days_val))).strftime("%Y-%m-%d")

    def draw_header(c, title, subtitle=""):
        c.setFont(FONT, 14)
        c.drawCentredString(A4[0] / 2, A4[1] - 2 * cm, title)
        if subtitle:
            c.setFont(FONT, 10)
            c.drawCentredString(A4[0] / 2, A4[1] - 2.8 * cm, subtitle)
        c.line(2 * cm, A4[1] - 3.2 * cm, A4[0] - 2 * cm, A4[1] - 3.2 * cm)
        return A4[1] - 4 * cm

    def draw_field(c, y, label, value, x=2.5 * cm, label_width=6 * cm):
        c.setFont(FONT, 10)
        c.drawString(x, y, f"{label}:")
        c.drawString(x + label_width, y, str(value if value is not None else "N/A"))
        return y - 0.6 * cm

    def draw_section(c, y, title):
        y -= 0.3 * cm
        c.setFont(FONT, 11)
        c.setFillColor(colors.HexColor("#1a5276"))
        c.drawString(2.2 * cm, y, title)
        c.setFillColor(colors.black)
        c.line(2.2 * cm, y - 0.15 * cm, A4[0] - 2 * cm, y - 0.15 * cm)
        return y - 0.7 * cm

    # ── Use original English enum values directly for round-trip fidelity ──
    gender_display = str(app.get("CODE_GENDER", "M"))
    family_display = str(app.get("NAME_FAMILY_STATUS", "Married"))
    education_display = str(app.get("NAME_EDUCATION_TYPE", ""))

    # ── 1. CCCD PDF ──
    path = os.path.join(customer_dir, "01_cccd.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    y = draw_header(c, "CAN CUOC CONG DAN", "Doc lap - Tu do - Hanh phuc")
    y = draw_section(c, y, "THONG TIN CA NHAN")
    y = draw_field(c, y, "So CCCD", f"0{sk_id}")
    y = draw_field(c, y, "Ho va ten", f"Customer_{sk_id}")
    y = draw_field(c, y, "Ngay sinh", days_to_date(app.get("DAYS_BIRTH")))
    y = draw_field(c, y, "Gioi tinh", gender_display)  # English enum: M/F/XNA
    y = draw_field(c, y, "Quoc tich", "Viet Nam")
    y = draw_field(c, y, "Noi thuong tru", f"Address of customer {sk_id}")
    y = draw_section(c, y, "THONG TIN CAP VA HIEU LUC")
    y = draw_field(c, y, "Ngay cap", days_to_date(app.get("DAYS_ID_PUBLISH")))
    y = draw_field(c, y, "Noi cap", "Cuc Canh sat DKQL cu tru va DLQG ve dan cu")
    # Page 2 for registration date
    c.showPage()
    y = draw_header(c, "CAN CUOC CONG DAN - MAT SAU")
    y = draw_section(c, y, "XAC NHAN DANG KY CU TRU")
    y = draw_field(c, y, "Ngay dang ky", days_to_date(app.get("DAYS_REGISTRATION")))
    y = draw_field(c, y, "Dia chi hien tai", f"Address of customer {sk_id}")
    y = draw_field(c, y, "Dia chi lam viec", f"Work address of customer {sk_id}")
    c.save()
    print(f"  Created: {path}")

    # ── 2. Labor contract PDF ──
    path = os.path.join(customer_dir, "02_hop_dong_lao_dong.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    y = draw_header(c, "HOP DONG LAO DONG")
    y = draw_section(c, y, "BEN SU DUNG LAO DONG (BEN A)")
    y = draw_field(c, y, "Ten doanh nghiep", str(app.get("ORGANIZATION_TYPE", "N/A")))  # English enum
    y = draw_field(c, y, "Dien thoai", "028 1234 5678" if app.get("FLAG_EMP_PHONE") == 1 else "Khong co")
    y = draw_section(c, y, "NGUOI LAO DONG (BEN B)")
    y = draw_field(c, y, "Ho va ten", f"Customer_{sk_id}")
    y = draw_field(c, y, "Chuc danh", str(app.get("OCCUPATION_TYPE", "N/A")))  # English enum
    y = draw_field(c, y, "Ngay bat dau", days_to_date(app.get("DAYS_EMPLOYED")))
    y = draw_field(c, y, "Loai hop dong", str(app.get("NAME_INCOME_TYPE", "Working")))  # English enum directly
    # Page 2 - salary
    c.showPage()
    y = draw_header(c, "HOP DONG LAO DONG (tiep theo)")
    y = draw_section(c, y, "DIEU 2: LUONG VA CHE DO")
    income = app.get("AMT_INCOME_TOTAL")
    monthly = income / 12 if income else None
    y = draw_field(c, y, "Luong co ban", f"{monthly:,.0f} VND/thang" if monthly else "N/A")
    y = draw_field(c, y, "Thu nhap nam", f"{income:,.0f} VND/nam" if income else "N/A")
    c.save()
    print(f"  Created: {path}")

    # ── 3. Household PDF ──
    path = os.path.join(customer_dir, "03_so_ho_khau.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    y = draw_header(c, "SO HO KHAU")
    y = draw_section(c, y, "THONG TIN CHU HO")
    y = draw_field(c, y, "Ho va ten chu ho", f"Customer_{sk_id}")
    y = draw_section(c, y, "TONG HOP")
    cnt_children = app.get("CNT_CHILDREN", 0)
    cnt_fam = app.get("CNT_FAM_MEMBERS", 1)
    y = draw_field(c, y, "Tong so nhan khau", f"{int(cnt_fam) if cnt_fam else 1} nguoi")
    y = draw_field(c, y, "Tinh trang hon nhan chu ho", family_display)
    y = draw_field(c, y, "So con", str(int(cnt_children) if cnt_children else 0))
    c.save()
    print(f"  Created: {path}")

    # ── 4. Housing survey PDF ──
    path = os.path.join(customer_dir, "04_tham_dinh_nha_o.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    y = draw_header(c, "PHIEU THAM DINH TAI SAN / NHA O")
    y = draw_section(c, y, "THONG TIN BAT DONG SAN")
    y = draw_field(c, y, "Loai hinh nha o", str(app.get("NAME_HOUSING_TYPE", "N/A")))
    y = draw_field(c, y, "Loai toa nha", str(app.get("HOUSETYPE_MODE", "block of flats")))  # Separate from NAME_HOUSING_TYPE

    # Housing features: use LIVINGAREA_AVG as proxy for living_area
    living_area_norm = app.get("LIVINGAREA_AVG")
    living_area_m2 = f"{living_area_norm * 200:.1f} m2" if living_area_norm else "N/A"
    y = draw_field(c, y, "Dien tich su dung", living_area_m2)

    # Year built
    years_build = app.get("YEARS_BUILD_AVG")
    year_built = int(1950 + years_build * 80) if years_build else None
    y = draw_field(c, y, "Nam xay dung", str(year_built) if year_built else "N/A")

    # Floors
    floors_norm = app.get("FLOORSMAX_AVG")
    max_floors = int(floors_norm * 50) if floors_norm else None
    y = draw_field(c, y, "So tang toa nha", str(max_floors) if max_floors else "N/A")

    # Elevator
    elev = app.get("ELEVATORS_AVG")
    y = draw_field(c, y, "Co thang may", "Co" if elev and elev > 0.5 else "Khong")

    y = draw_field(c, y, "Vat lieu tuong", str(app.get("WALLSMATERIAL_MODE", "N/A")))
    y = draw_field(c, y, "Tinh trang khan cap", "Binh thuong (khong khan cap)" if app.get("EMERGENCYSTATE_MODE") == "No" else "Co tinh trang khan cap")

    # Quality based on APARTMENTS_AVG as proxy
    quality = app.get("APARTMENTS_AVG")
    quality_score = f"{quality * 10:.1f} / 10" if quality else "N/A"
    y = draw_field(c, y, "Chat luong can ho (1-10)", quality_score)

    # Page 2 - Normalized housing detail fields
    c.showPage()
    y = draw_header(c, "PHIEU THAM DINH (tiep theo)")
    y = draw_section(c, y, "CHI TIET BAT DONG SAN (NORMALIZED 0-1)")

    # Helper to format normalized value or N/A
    def _fmt_norm(val):
        if val is None:
            return "N/A"
        return str(val)

    y = draw_field(c, y, "Dien tich can ho (norm)", _fmt_norm(app.get("APARTMENTS_AVG")))
    y = draw_field(c, y, "Dien tich tang ham (norm)", _fmt_norm(app.get("BASEMENTAREA_AVG")))
    y = draw_field(c, y, "Nam bat dau su dung (norm)", _fmt_norm(app.get("YEARS_BEGINEXPLUATATION_AVG")))
    y = draw_field(c, y, "Nam xay dung (norm)", _fmt_norm(app.get("YEARS_BUILD_AVG")))
    y = draw_field(c, y, "Dien tich chung (norm)", _fmt_norm(app.get("COMMONAREA_AVG")))
    y = draw_field(c, y, "Thang may (norm)", _fmt_norm(app.get("ELEVATORS_AVG")))
    y = draw_field(c, y, "Loi vao (norm)", _fmt_norm(app.get("ENTRANCES_AVG")))
    y = draw_field(c, y, "So tang max (norm)", _fmt_norm(app.get("FLOORSMAX_AVG")))
    y = draw_field(c, y, "So tang min (norm)", _fmt_norm(app.get("FLOORSMIN_AVG")))
    y = draw_field(c, y, "Dien tich dat (norm)", _fmt_norm(app.get("LANDAREA_AVG")))
    y = draw_field(c, y, "Dien tich o (can ho, norm)", _fmt_norm(app.get("LIVINGAPARTMENTS_AVG")))
    y = draw_field(c, y, "Dien tich song (norm)", _fmt_norm(app.get("LIVINGAREA_AVG")))
    y = draw_field(c, y, "Dien tich phi o (phong, norm)", _fmt_norm(app.get("NONLIVINGAPARTMENTS_AVG")))
    y = draw_field(c, y, "Dien tich phi o (norm)", _fmt_norm(app.get("NONLIVINGAREA_AVG")))
    y = draw_field(c, y, "Tong dien tich (norm)", _fmt_norm(app.get("TOTALAREA_MODE")))
    y = draw_field(c, y, "Quy sua chua", _fmt_norm(app.get("FONDKAPREMONT_MODE")))

    # Page 2b - _MODE variant fields
    c.showPage()
    y = draw_header(c, "PHIEU THAM DINH (tiep theo)")
    y = draw_section(c, y, "CHI TIET BAT DONG SAN (MODE)")
    y = draw_field(c, y, "Dien tich can ho (mode)", _fmt_norm(app.get("APARTMENTS_MODE")))
    y = draw_field(c, y, "Dien tich tang ham (mode)", _fmt_norm(app.get("BASEMENTAREA_MODE")))
    y = draw_field(c, y, "Nam bat dau su dung (mode)", _fmt_norm(app.get("YEARS_BEGINEXPLUATATION_MODE")))
    y = draw_field(c, y, "Nam xay dung (mode)", _fmt_norm(app.get("YEARS_BUILD_MODE")))
    y = draw_field(c, y, "Dien tich chung (mode)", _fmt_norm(app.get("COMMONAREA_MODE")))
    y = draw_field(c, y, "Thang may (mode)", _fmt_norm(app.get("ELEVATORS_MODE")))
    y = draw_field(c, y, "Loi vao (mode)", _fmt_norm(app.get("ENTRANCES_MODE")))
    y = draw_field(c, y, "So tang max (mode)", _fmt_norm(app.get("FLOORSMAX_MODE")))
    y = draw_field(c, y, "So tang min (mode)", _fmt_norm(app.get("FLOORSMIN_MODE")))
    y = draw_field(c, y, "Dien tich dat (mode)", _fmt_norm(app.get("LANDAREA_MODE")))
    y = draw_field(c, y, "Dien tich o (can ho, mode)", _fmt_norm(app.get("LIVINGAPARTMENTS_MODE")))
    y = draw_field(c, y, "Dien tich song (mode)", _fmt_norm(app.get("LIVINGAREA_MODE")))
    y = draw_field(c, y, "Dien tich phi o (phong, mode)", _fmt_norm(app.get("NONLIVINGAPARTMENTS_MODE")))
    y = draw_field(c, y, "Dien tich phi o (mode)", _fmt_norm(app.get("NONLIVINGAREA_MODE")))

    # Page 2c - _MEDI variant fields
    c.showPage()
    y = draw_header(c, "PHIEU THAM DINH (tiep theo)")
    y = draw_section(c, y, "CHI TIET BAT DONG SAN (MEDI)")
    y = draw_field(c, y, "Dien tich can ho (medi)", _fmt_norm(app.get("APARTMENTS_MEDI")))
    y = draw_field(c, y, "Dien tich tang ham (medi)", _fmt_norm(app.get("BASEMENTAREA_MEDI")))
    y = draw_field(c, y, "Nam bat dau su dung (medi)", _fmt_norm(app.get("YEARS_BEGINEXPLUATATION_MEDI")))
    y = draw_field(c, y, "Nam xay dung (medi)", _fmt_norm(app.get("YEARS_BUILD_MEDI")))
    y = draw_field(c, y, "Dien tich chung (medi)", _fmt_norm(app.get("COMMONAREA_MEDI")))
    y = draw_field(c, y, "Thang may (medi)", _fmt_norm(app.get("ELEVATORS_MEDI")))
    y = draw_field(c, y, "Loi vao (medi)", _fmt_norm(app.get("ENTRANCES_MEDI")))
    y = draw_field(c, y, "So tang max (medi)", _fmt_norm(app.get("FLOORSMAX_MEDI")))
    y = draw_field(c, y, "So tang min (medi)", _fmt_norm(app.get("FLOORSMIN_MEDI")))
    y = draw_field(c, y, "Dien tich dat (medi)", _fmt_norm(app.get("LANDAREA_MEDI")))
    y = draw_field(c, y, "Dien tich o (can ho, medi)", _fmt_norm(app.get("LIVINGAPARTMENTS_MEDI")))
    y = draw_field(c, y, "Dien tich song (medi)", _fmt_norm(app.get("LIVINGAREA_MEDI")))
    y = draw_field(c, y, "Dien tich phi o (phong, medi)", _fmt_norm(app.get("NONLIVINGAPARTMENTS_MEDI")))
    y = draw_field(c, y, "Dien tich phi o (medi)", _fmt_norm(app.get("NONLIVINGAREA_MEDI")))

    # Page 3 - Region info + city cross-checks
    c.showPage()
    y = draw_header(c, "PHIEU THAM DINH (tiep theo)")
    y = draw_section(c, y, "THONG TIN KHU VUC")
    y = draw_field(c, y, "Mat do dan so (tuong doi)", str(app.get("REGION_POPULATION_RELATIVE", "N/A")))
    y = draw_field(c, y, "Xep hang khu vuc", f"{int(app.get('REGION_RATING_CLIENT', 2))}")
    y = draw_field(c, y, "Xep hang khu vuc (TP)", f"{int(app.get('REGION_RATING_CLIENT_W_CITY', 2))}")
    y = draw_field(c, y, "Dang ky cung vung song",
                   "Co" if app.get("REG_REGION_NOT_LIVE_REGION") == 0 else "Khong")
    y = draw_field(c, y, "Dang ky cung TP lam viec",
                   "Co" if app.get("REG_REGION_NOT_WORK_REGION") == 0 else "Khong")
    y = draw_field(c, y, "Song cung vung lam viec",
                   "Co" if app.get("LIVE_REGION_NOT_WORK_REGION") == 0 else "Khong")

    y = draw_section(c, y, "DIA CHI CROSS-CHECK (THANH PHO)")
    y = draw_field(c, y, "Dang ky cung TP song",
                   "Co" if app.get("REG_CITY_NOT_LIVE_CITY") == 0 else "Khong")
    y = draw_field(c, y, "Dang ky cung TP lam viec (TP)",
                   "Co" if app.get("REG_CITY_NOT_WORK_CITY") == 0 else "Khong")
    y = draw_field(c, y, "Song cung TP lam viec (TP)",
                   "Co" if app.get("LIVE_CITY_NOT_WORK_CITY") == 0 else "Khong")
    c.save()
    print(f"  Created: {path}")

    # ── 5. Loan application PDF ──
    path = os.path.join(customer_dir, "05_don_vay.pdf")
    c = canvas.Canvas(path, pagesize=A4)
    y = draw_header(c, "DON DE NGHI VAY VON")
    y = draw_section(c, y, "I. THONG TIN NGUOI VAY")
    y = draw_field(c, y, "Ho va ten", f"Customer_{sk_id}")
    y = draw_field(c, y, "So CCCD", f"0{sk_id}")
    y = draw_field(c, y, "Ngay sinh", days_to_date(app.get("DAYS_BIRTH")))
    y = draw_field(c, y, "Gioi tinh", gender_display)
    y = draw_field(c, y, "Trinh do hoc van", education_display)  # English enum directly
    y = draw_field(c, y, "Tinh trang hon nhan", family_display)  # English enum directly
    y = draw_field(c, y, "Dien thoai noi lam viec", "1" if app.get("FLAG_EMP_PHONE") == 1 else "0")
    y = draw_field(c, y, "Dien thoai ban cong ty", "1" if app.get("FLAG_WORK_PHONE") == 1 else "0")

    # Car/Realty
    y = draw_field(c, y, "Co xe o to", "Co" if app.get("FLAG_OWN_CAR") == "Y" else "Khong")
    car_age = app.get("OWN_CAR_AGE")
    y = draw_field(c, y, "Tuoi xe (nam)", str(int(car_age)) if car_age else "N/A")
    y = draw_field(c, y, "Co bat dong san", "Co" if app.get("FLAG_OWN_REALTY") == "Y" else "Khong")
    y = draw_field(c, y, "Nguoi dong hanh", str(app.get("NAME_TYPE_SUITE", "Unaccompanied")))

    # Contact flags
    y = draw_field(c, y, "So di dong lien lac duoc", "Co" if app.get("FLAG_CONT_MOBILE") == 1 else "Khong")
    y = draw_field(c, y, "So dien thoai ban", "Co" if app.get("FLAG_PHONE") == 1 else "Khong")
    y = draw_field(c, y, "Email", "Co" if app.get("FLAG_EMAIL") == 1 else "Khong")

    phone_change = app.get("DAYS_LAST_PHONE_CHANGE")
    if phone_change is not None:
        y = draw_field(c, y, "Ngay doi SDT gan nhat",
                       f"{days_to_date(phone_change)} (khoang {abs(int(phone_change))} ngay truoc)")

    # Weekday/Hour of application (stored for round-trip)
    weekday = app.get("WEEKDAY_APPR_PROCESS_START", "MONDAY")
    hour = app.get("HOUR_APPR_PROCESS_START", 10)
    y = draw_field(c, y, "Ngay nop don", str(weekday))
    y = draw_field(c, y, "Gio nop don", str(int(hour)))

    y = draw_section(c, y, "II. THONG TIN KHOAN VAY")
    y = draw_field(c, y, "Loai hop dong", str(app.get("NAME_CONTRACT_TYPE", "Cash loans")))
    credit = app.get("AMT_CREDIT")
    y = draw_field(c, y, "So tien vay", f"{credit:,.0f} VND" if credit else "N/A")
    annuity = app.get("AMT_ANNUITY")
    y = draw_field(c, y, "Tra hang thang (du kien)", f"{annuity:,.0f} VND" if annuity else "N/A")
    goods = app.get("AMT_GOODS_PRICE")
    y = draw_field(c, y, "Gia tri hang hoa", f"{goods:,.0f} VND" if goods else "N/A")

    # Page 2 - FLAG_DOCUMENT values
    c.showPage()
    y = draw_header(c, "DON DE NGHI VAY VON (tiep theo)")
    y = draw_section(c, y, "III. TAI LIEU DA NOP")
    for i in range(2, 22):
        flag_key = f"FLAG_DOCUMENT_{i}"
        y = draw_field(c, y, f"Tai lieu {i}", str(int(app.get(flag_key, 0))))

    c.save()
    print(f"  Created: {path}")


def main():
    app, bureau, bb, prev, pos, inst, cc = load_tables()
    selected_ids = select_customers(app, bureau, prev, pos, inst, cc)

    # Label mapping
    labels = {0: "pass", 1: "fail"}
    target_counts = {0: 0, 1: 0}

    customer_map = {}  # sk_id -> customer_num

    for i, sk_id in enumerate(selected_ids):
        target = int(app[app["SK_ID_CURR"] == sk_id]["TARGET"].iloc[0])
        target_counts[target] += 1
        label = f"{labels[target]}_{target_counts[target]}"
        customer_num = f"customer_{i + 1:03d}"
        customer_dir = OUTPUT_DIR / customer_num

        print(f"\n{'=' * 60}")
        print(f"  Extracting {customer_num}: SK_ID_CURR={sk_id}, TARGET={target} ({label})")
        print(f"{'=' * 60}")

        # Create directory
        customer_dir.mkdir(parents=True, exist_ok=True)

        # Extract all data
        data = extract_customer_data(sk_id, app, bureau, bb, prev, pos, inst, cc)

        # Save full application_row.json (for debugging/reference)
        app_path = customer_dir / "application_row.json"
        with open(app_path, "w", encoding="utf-8") as f:
            json.dump(data["application_row"], f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved: {app_path}")

        # Generate CIC API JSON
        cic = build_cic_json(data)
        cic_path = customer_dir / "07_cic_api_response.json"
        with open(cic_path, "w", encoding="utf-8") as f:
            json.dump(cic, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved: {cic_path} ({len(cic['bureau_records'])} bureau records)")

        # Generate Internal DB JSON
        internal = build_internal_db_json(data)
        internal_path = customer_dir / "08_internal_db.json"
        with open(internal_path, "w", encoding="utf-8") as f:
            json.dump(internal, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved: {internal_path}")
        print(f"    prev_apps={len(internal['previous_applications'])}, "
              f"pos={len(internal['pos_cash_balance'])}, "
              f"inst={len(internal['installments_payments'])}, "
              f"cc={len(internal['credit_card_balance'])}")

        # Generate PDFs
        generate_pdfs_for_customer(data, str(customer_dir))

        customer_map[int(sk_id)] = {
            "dir": customer_num,
            "target": target,
            "label": label,
        }

    # Save customer map
    map_path = OUTPUT_DIR / "customer_map.json"
    with open(map_path, "w") as f:
        json.dump(customer_map, f, indent=2)
    print(f"\n\nCustomer map saved: {map_path}")
    print(f"\nDone! All {len(customer_map)} customers extracted successfully.")


if __name__ == "__main__":
    main()
