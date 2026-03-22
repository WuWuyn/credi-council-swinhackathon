"""
CreditLens -- End-to-End Pipeline Test.

Runs the full pipeline: A1 -> A2 -> A3 -> A4
on mock customer data to verify all agents work together.

Upgraded: 5C framework + 6-section report + SHAP label_vi + bank statement.
"""

import sys, os, json, logging, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline():
    """Run the full credit scoring pipeline."""
    print("\n" + "=" * 70)
    print("  CREDITLENS — END-TO-END PIPELINE TEST")
    print("=" * 70)

    # Auto-detect mock mode: use real API if GEMINI_API_KEY is set
    use_mock = not bool(os.environ.get("GEMINI_API_KEY"))
    if os.environ.get("USE_MOCK", "").lower() == "true":
        use_mock = True
    elif os.environ.get("USE_MOCK", "").lower() == "false":
        use_mock = False

    mode_label = "MOCK" if use_mock else "REAL (Gemini API)"
    print(f"  Mode: {mode_label}")

    # ── A1: Data Ingestion ──
    print("\n[1/4] A1: Data Ingestion...")
    from creditlens.agents.a1_ingestion.agent import IngestionAgent
    a1 = IngestionAgent(use_mock=True)  # Always mock for data source
    a1_output = a1.ingest(customer_dir="data/mock/customer_001")

    app = a1_output["application_row"]
    print(f"  ✅ {len(app)} application fields")
    print(f"  ✅ Bureau: {len(a1_output['bureau_df'])} records")
    print(f"  ✅ Previous apps: {len(a1_output['previous_application_df'])} records")

    # ── A2: Feature Engineering ──
    print("\n[2/4] A2: LLM Feature Engineering...")
    from creditlens.agents.a2_feature_engineer.agent import FeatureEngineerAgent
    a2 = FeatureEngineerAgent(use_mock=use_mock)
    a2_output = a2.process(a1_output)

    fv = a2_output.get("feature_vector")
    if fv is not None:
        print(f"  ✅ Feature vector: {len(fv)} features")
    else:
        print(f"  ⚠️ Feature vector: None (using raw features)")

    llm = a2_output.get("llm_feats", {})
    print(f"  ✅ LLM features: purpose={llm.get('loan_purpose_category')}")
    print(f"  ✅ Positive signals: {llm.get('positive_signals')}")

    # ── A3: ML Scoring ──
    print("\n[3/4] A3: ML Scoring Engine...")
    from creditlens.agents.a3_scoring.agent import ScoringAgent
    a3 = ScoringAgent(model_path="models/lgbm_ref_v1.pkl")
    a3_output = a3.score(a2_output)

    print(f"  ✅ Credit Score: {a3_output['credit_score']}")
    print(f"  ✅ PD%: {a3_output['pd_pct']:.2f}%")
    print(f"  ✅ Risk Band: {a3_output['risk_band']}")
    print(f"  ✅ Decision: {a3_output['routing']}")

    # SHAP
    shap = a3_output.get("shap_values", {})
    top_pos = shap.get("top_positive_factors", [])[:3]
    top_neg = shap.get("top_negative_factors", [])[:3]
    print(f"  ✅ SHAP top positive (reduce default risk):")
    for f in top_neg[:3]:
        print(f"       {f.get('label_vi', f['feature'])} (SHAP={f['shap_value']:.4f}, val={f.get('value')})")
    print(f"  ✅ SHAP top negative (increase default risk):")
    for f in top_pos[:3]:
        print(f"       {f.get('label_vi', f['feature'])} (SHAP={f['shap_value']:.4f}, val={f.get('value')})")

    # 5C SHAP allocation
    alloc = shap.get("five_c_shap_allocation", {})
    if alloc:
        print(f"  ✅ 5C SHAP allocation:")
        for dim, info in alloc.items():
            print(f"       {dim}: {info.get('pct', 0)}%")

    # ── A4: Report Generator ──
    print("\n[4/4] A4: Report Generator (5C + 6 sections)...")
    from creditlens.agents.a4_report_generator.agent import ReportGeneratorAgent
    a4 = ReportGeneratorAgent(use_mock=use_mock)
    a4_output = a4.generate(a3_output, a2_output, a1_output)

    report = a4_output.get("final_report", {})
    five_c = a4_output.get("five_c_scores", {})
    print(f"  ✅ 5C Scores: {five_c}")
    print(f"  ✅ 5C Total: {sum(five_c.values())}/120")
    print(f"  ✅ Report sections: {list(report.keys())}")
    print(f"  ✅ Recommendation: {report.get('executive_summary', {}).get('recommendation')}")
    print(f"  ✅ Consistency: {a4_output.get('consistency_check', {}).get('passed')}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  PIPELINE SUMMARY")
    print("=" * 70)
    audit = a4_output.get("audit_trail", [])
    print(f"  Audit trail: {len(audit)} entries")
    for entry in audit:
        print(f"    [{entry.get('agent')}] {entry.get('action')}")
    print(f"\n  Credit Score: {a3_output['credit_score']}")
    print(f"  Risk Band: {a3_output['risk_band']}")
    print(f"  PD%: {a3_output['pd_pct']:.2f}%")
    print(f"  Decision: {a3_output['routing']}")
    print(f"  5C Scores: {five_c}")
    print(f"  5C Total: {sum(five_c.values())}/120")
    print(f"  Warnings: {len(a4_output.get('warnings', []))}")
    # Save report
    report_path = "data/mock/customer_001/credit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Report saved: {report_path}")
    print("=" * 70)

    return a4_output


if __name__ == "__main__":
    result = run_pipeline()
