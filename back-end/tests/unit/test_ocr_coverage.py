"""
Test: OCR pipeline (PyMuPDF) feature extraction coverage.
Compares OCR-extracted application_row vs ground truth (application_row.json).
"""
import json
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
from creditlens.agents.a1_ingestion.agent import IngestionAgent

CUSTOMER_DIR = Path("data/mock/customer_001")
GROUND_TRUTH = CUSTOMER_DIR / "application_row.json"

# 1. Load ground truth
with open(GROUND_TRUTH, encoding="utf-8") as f:
    truth = json.load(f)
truth.pop("TARGET", None)

# 2. Run OCR pipeline
agent = IngestionAgent()
result = agent.ingest(CUSTOMER_DIR)
ocr_row = result["application_row"]

# 3. Compare field coverage
print("=" * 70)
print("  OCR PIPELINE — FEATURE EXTRACTION COVERAGE REPORT")
print("=" * 70)

# Classify each field
matched = []       # OCR == truth (exact or close)
wrong = []         # OCR != truth (has value but wrong)
missing_ocr = []   # OCR is None/default, truth has real value
extra_ocr = []     # OCR has value, truth is None
both_null = []     # Both None

# Fields that are defaults (not actually extracted)
DEFAULT_VALUES = {
    "NAME_CONTRACT_TYPE": "Cash loans",
    "NAME_TYPE_SUITE": "Unaccompanied",
    "NAME_EDUCATION_TYPE": "Higher education",
    "NAME_HOUSING_TYPE": "House / apartment",
    "NAME_INCOME_TYPE": "Working",
    "ORGANIZATION_TYPE": "Business Entity Type 3",
    "OCCUPATION_TYPE": "Laborers",
    "NAME_FAMILY_STATUS": "Married",
    "FLAG_MOBIL": 1,
    "FLAG_CONT_MOBILE": 1,
}

for key in sorted(set(list(truth.keys()) + list(ocr_row.keys()))):
    t_val = truth.get(key)
    o_val = ocr_row.get(key)
    
    if t_val is None and o_val is None:
        both_null.append(key)
    elif t_val is None and o_val is not None:
        extra_ocr.append((key, o_val, t_val))
    elif t_val is not None and o_val is None:
        missing_ocr.append((key, t_val, o_val))
    else:
        # Both have values — check if they match
        # For numeric: allow tolerance
        try:
            t_num = float(t_val)
            o_num = float(o_val)
            if abs(t_num - o_num) < 0.01 * (abs(t_num) + 1):
                matched.append((key, o_val, t_val))
            else:
                # Check if OCR value is just a default
                if key in DEFAULT_VALUES and o_val == DEFAULT_VALUES[key]:
                    wrong.append((key, o_val, t_val, "DEFAULT"))
                else:
                    wrong.append((key, o_val, t_val, "MISMATCH"))
        except (ValueError, TypeError):
            if str(t_val).strip().lower() == str(o_val).strip().lower():
                matched.append((key, o_val, t_val))
            elif key in DEFAULT_VALUES and o_val == DEFAULT_VALUES[key] and t_val != o_val:
                wrong.append((key, o_val, t_val, "DEFAULT"))
            else:
                wrong.append((key, o_val, t_val, "MISMATCH"))

total = len(truth)
n_matched = len(matched)
n_wrong = len(wrong)
n_missing = len(missing_ocr)
n_default = len([w for w in wrong if w[3] == "DEFAULT"])
n_mismatch = len([w for w in wrong if w[3] == "MISMATCH"])

print(f"\n  Ground truth fields: {total}")
print(f"  OCR output fields:  {len(ocr_row)}")
print()

print(f"  ✅ MATCHED (OCR == Truth):        {n_matched:3d} ({n_matched/total*100:5.1f}%)")
print(f"  ⚠️  WRONG VALUE (OCR != Truth):    {n_wrong:3d} ({n_wrong/total*100:5.1f}%)")
print(f"      └─ Used default value:        {n_default:3d}")
print(f"      └─ Actual mismatch:           {n_mismatch:3d}")
print(f"  ❌ MISSING (Truth has, OCR null):  {n_missing:3d} ({n_missing/total*100:5.1f}%)")
print(f"  ⬜ Both null:                      {len(both_null):3d}")

# Detail sections
print(f"\n{'─'*70}")
print("  ⚠️  WRONG VALUES (OCR extracted but incorrect)")
print(f"{'─'*70}")
for item in wrong[:20]:
    key, ocr_v, truth_v, reason = item
    print(f"  {key:40s} OCR={str(ocr_v)[:20]:20s} Truth={str(truth_v)[:20]:20s} [{reason}]")
if len(wrong) > 20:
    print(f"  ... and {len(wrong)-20} more")

print(f"\n{'─'*70}")
print("  ❌ MISSING FIELDS (Truth has value, OCR returned None)")
print(f"{'─'*70}")
for key, truth_v, _ in missing_ocr[:20]:
    print(f"  {key:40s} Truth={str(truth_v)[:30]}")
if len(missing_ocr) > 20:
    print(f"  ... and {len(missing_ocr)-20} more")

print(f"\n{'─'*70}")
print("  ✅ MATCHED FIELDS")
print(f"{'─'*70}")
for key, ocr_v, truth_v in matched[:15]:
    print(f"  {key:40s} = {str(ocr_v)[:30]}")
if len(matched) > 15:
    print(f"  ... and {len(matched)-15} more")

# Source analysis
print(f"\n{'='*70}")
print("  SOURCE ANALYSIS — Where do matched fields come from?")
print(f"{'='*70}")
pdf_keys = set()
cic_keys = {"EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3",
            "AMT_REQ_CREDIT_BUREAU_HOUR","AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK","AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT","AMT_REQ_CREDIT_BUREAU_YEAR",
            "OBS_30_CNT_SOCIAL_CIRCLE","DEF_30_CNT_SOCIAL_CIRCLE",
            "OBS_60_CNT_SOCIAL_CIRCLE","DEF_60_CNT_SOCIAL_CIRCLE"}
matched_keys = {m[0] for m in matched}
from_cic = matched_keys & cic_keys
from_pdf = matched_keys - cic_keys
print(f"  From CIC API (JSON):    {len(from_cic):3d} fields")
print(f"  From PDF (OCR regex):   {len(from_pdf):3d} fields")
print(f"  TOTAL matched:          {len(matched):3d} / {total} ({len(matched)/total*100:.1f}%)")
