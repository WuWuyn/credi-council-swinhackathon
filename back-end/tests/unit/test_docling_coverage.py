"""
Test: Docling + LLM extraction pipeline vs ground truth.
Sets USE_DOCLING=true temporarily to test the new extraction path.
"""
import json, sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Force USE_DOCLING=true for this test
os.environ["USE_DOCLING"] = "true"

# Clear settings cache so new env is picked up
from credicouncil.config.settings import get_settings
get_settings.cache_clear()

from pathlib import Path
from credicouncil.agents.a1_ingestion.agent import IngestionAgent

CUSTOMER_DIR = Path("data/mock/customer_001")
GROUND_TRUTH = CUSTOMER_DIR / "application_row.json"

# Load ground truth
with open(GROUND_TRUTH, encoding="utf-8") as f:
    truth = json.load(f)
truth.pop("TARGET", None)

# Run Docling+LLM pipeline
print("=" * 70)
print("  DOCLING + LLM EXTRACTION PIPELINE TEST")
print("=" * 70)

agent = IngestionAgent()
result = agent.ingest(CUSTOMER_DIR)
ocr_row = result["application_row"]

# Compare
matched = []
wrong = []
missing_ocr = []
both_null = []

for key in sorted(set(list(truth.keys()) + list(ocr_row.keys()))):
    t_val = truth.get(key)
    o_val = ocr_row.get(key)
    
    if t_val is None and o_val is None:
        both_null.append(key)
    elif t_val is None and o_val is not None:
        pass  # extra
    elif t_val is not None and o_val is None:
        missing_ocr.append((key, t_val, o_val))
    else:
        try:
            t_num = float(t_val)
            o_num = float(o_val)
            if abs(t_num - o_num) < 0.01 * (abs(t_num) + 1):
                matched.append((key, o_val, t_val))
            else:
                wrong.append((key, o_val, t_val))
        except (ValueError, TypeError):
            if str(t_val).strip().lower() == str(o_val).strip().lower():
                matched.append((key, o_val, t_val))
            else:
                wrong.append((key, o_val, t_val))

total = len(truth)
print(f"\n  Ground truth fields: {total}")
print(f"  Docling+LLM fields: {len(ocr_row)}")
print()
print(f"  ✅ MATCHED:  {len(matched):3d} ({len(matched)/total*100:5.1f}%)")
print(f"  ⚠️  WRONG:    {len(wrong):3d} ({len(wrong)/total*100:5.1f}%)")
print(f"  ❌ MISSING:  {len(missing_ocr):3d} ({len(missing_ocr)/total*100:5.1f}%)")
print(f"  ⬜ Both null: {len(both_null):3d}")

if wrong:
    print(f"\n{'─'*70}")
    print("  ⚠️  WRONG VALUES")
    print(f"{'─'*70}")
    for key, ocr_v, truth_v in wrong[:20]:
        print(f"  {key:40s} LLM={str(ocr_v)[:20]:20s} Truth={str(truth_v)[:20]:20s}")

if missing_ocr:
    print(f"\n{'─'*70}")
    print("  ❌ MISSING FIELDS")
    print(f"{'─'*70}")
    for key, truth_v, _ in missing_ocr[:20]:
        print(f"  {key:40s} Truth={str(truth_v)[:30]}")
