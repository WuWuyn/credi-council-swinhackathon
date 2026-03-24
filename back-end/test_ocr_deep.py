"""Trace full pipeline: what doc_type does auto-detection assign?"""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
from creditlens.agents.a1_ingestion.document_parser import LocalDocumentParser

parser = LocalDocumentParser()
customer_dir = Path("data/mock/customer_001")

for pdf in sorted(customer_dir.glob("*.pdf")):
    result = parser.extract_document(pdf)  # doc_type="auto"
    dt = result["doc_type"]
    n_fields = len(result["fields"])
    norm_count = len([k for k in result["fields"] if "_norm" in k and result["fields"][k] is not None])
    cross_count = len([k for k in result["fields"] if "reg_" in k or "live_" in k])
    print(f"  {pdf.name:35s} → doc_type={dt:20s} fields={n_fields:3d}  norms={norm_count:2d}  cross={cross_count:2d}")
    
    if dt == "housing":
        print(f"    has apartments_norm: {result['fields'].get('apartments_norm')}")
        print(f"    has reg_live_same_region: {result['fields'].get('reg_live_same_region')}")
