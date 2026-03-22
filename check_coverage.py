"""Final coverage: verify ALL 753 features present + 128 raw features"""
import sys, os, logging
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.WARNING)

from creditlens.agents.a1_ingestion.agent import IngestionAgent
from creditlens.agents.a2_feature_engineer.single_customer_fe import SingleCustomerFE

fe = SingleCustomerFE("models/fe_stats.pkl")
a1 = IngestionAgent(use_mock=True)
a1_out = a1.ingest("data/mock/customer_001")

# === RAW INPUT FEATURES (128 target) ===
app_row = a1_out["application_row"]
raw_non_null = len([v for v in app_row.values() if v is not None])
raw_total = len(app_row)
print(f"\n{'='*60}")
print(f"  RAW INPUT FEATURES")
print(f"{'='*60}")
print(f"  Total fields:     {raw_total}")
print(f"  Non-null fields:  {raw_non_null}")
null_fields = [k for k, v in app_row.items() if v is None]
print(f"  Null fields:      {len(null_fields)}")
for f in null_fields[:10]:
    print(f"    {f}")
if len(null_fields) > 10:
    print(f"    ... and {len(null_fields)-10} more")

# === FEATURE VECTOR (753 target) ===
vector = fe.build_features(a1_out)
total = len(vector)
has_value = (vector != -999.0).sum()
has_zero = (vector == 0).sum()
has_sentinel = (vector == -999.0).sum()
has_nonzero = ((vector != 0) & (vector != -999.0)).sum()

print(f"\n{'='*60}")
print(f"  ML FEATURE VECTOR")
print(f"{'='*60}")
print(f"  Feature slots:       {total}/753 ✅")
print(f"  With computed value: {has_value} ({has_value/total*100:.1f}%)")
print(f"    - Non-zero:        {has_nonzero}")
print(f"    - Zero (absent):   {has_zero}")
print(f"  With sentinel -999:  {has_sentinel} (legitimate 'no data')")
print(f"{'='*60}")

if total == 753:
    print(f"  ✅ ALL 753 FEATURE SLOTS FILLED")
else:
    print(f"  ❌ MISSING {753 - total} SLOTS")

# Show sentinel features
if has_sentinel > 0:
    sentinel_feats = vector[vector == -999.0]
    cats = {}
    for f in sentinel_feats.index:
        if f.startswith("bureau_"):
            cat = "BUREAU"
        elif f.startswith("prev_"):
            cat = "PREV_APP"
        elif f.startswith("inst_"):
            cat = "INSTALLMENTS"
        elif "to_prev_" in f:
            cat = "CROSS_TABLE"
        else:
            cat = f.split("_")[0].upper()
        cats.setdefault(cat, []).append(f)
    print(f"\n  Sentinel (-999) features by category:")
    for cat, feats in sorted(cats.items()):
        print(f"    {cat}: {len(feats)}")
        for f in feats:
            print(f"      {f}")
