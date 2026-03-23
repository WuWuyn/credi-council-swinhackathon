"""
CreditLens — Test All 4 Demo Customers.

Runs the full pipeline (A1→A2→A3→A4) on all 4 customers
and produces a comparison summary.

Usage:
    python test_all_customers.py
"""
import sys, os, json, logging, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.WARNING,  # Reduce noise for multi-customer run
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CUSTOMERS = [
    {"dir": "data/mock/customer_001", "name": "Customer_418735",  "profile": "TARGET=0 (pass), EXT_high=0.78-0.89", "expected_target": 0},
    {"dir": "data/mock/customer_002", "name": "Customer_394570",  "profile": "TARGET=0 (pass), EXT_high=0.79-0.88", "expected_target": 0},
    {"dir": "data/mock/customer_003", "name": "Customer_272483",  "profile": "TARGET=1 (fail), EXT_low=0.001-0.10", "expected_target": 1},
    {"dir": "data/mock/customer_004", "name": "Customer_169206",  "profile": "TARGET=1 (fail), EXT_low=0.001-0.17", "expected_target": 1},
]



def run_single_customer(customer_dir, customer_name):
    """Run pipeline for a single customer, return results dict."""
    result = {
        "customer_dir": customer_dir,
        "customer_name": customer_name,
        "status": "FAILED",
        "error": None,
    }

    try:
        # ── A1 ──
        from creditlens.agents.a1_ingestion.agent import IngestionAgent
        a1 = IngestionAgent()
        a1_output = a1.ingest(customer_dir=customer_dir)

        app = a1_output["application_row"]
        result["a1_fields"] = len(app)
        result["a1_bureau"] = len(a1_output["bureau_df"])
        result["a1_prev_apps"] = len(a1_output["previous_application_df"])

        # ── A2 ──
        from creditlens.agents.a2_feature_engineer.agent import FeatureEngineerAgent
        a2 = FeatureEngineerAgent()
        a2_output = a2.process(a1_output)

        fv = a2_output.get("feature_vector")
        result["a2_features"] = len(fv) if fv is not None else 0
        result["a2_llm_purpose"] = a2_output.get("llm_feats", {}).get("loan_purpose_category")

        # ── A3 ──
        from creditlens.agents.a3_scoring.agent import ScoringAgent
        a3 = ScoringAgent(model_path="models/lgbm_ref_v1.pkl")
        a3_output = a3.score(a2_output)

        result["credit_score"] = a3_output["credit_score"]
        result["pd_pct"] = a3_output["pd_pct"]
        result["risk_band"] = a3_output["risk_band"]
        result["routing"] = a3_output["routing"]

        shap = a3_output.get("shap_values", {})
        result["shap_top_positive"] = [
            f.get("label_vi", f["feature"]) for f in shap.get("top_positive_factors", [])[:3]
        ]
        result["shap_top_negative"] = [
            f.get("label_vi", f["feature"]) for f in shap.get("top_negative_factors", [])[:3]
        ]

        # ── A4 ──
        from creditlens.agents.a4_report_generator.agent import ReportGeneratorAgent
        a4 = ReportGeneratorAgent()
        a4_output = a4.generate(a3_output, a2_output, a1_output)

        report = a4_output.get("final_report", {})
        five_c = a4_output.get("five_c_scores", {})
        result["five_c"] = five_c
        result["five_c_total"] = sum(five_c.values())
        result["recommendation"] = report.get("executive_summary", {}).get("recommendation")
        result["consistency_passed"] = a4_output.get("consistency_check", {}).get("passed")
        result["report_sections"] = list(report.keys())
        result["warnings_count"] = len(a4_output.get("warnings", []))

        # Save report
        report_path = os.path.join(customer_dir, "credit_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        result["report_path"] = report_path

        # Save SHAP values for PDF report generation
        shap_path = os.path.join(customer_dir, "shap_values.json")
        shap_full = a3_output.get("shap_values", {})
        with open(shap_path, "w", encoding="utf-8") as f:
            json.dump(shap_full, f, ensure_ascii=False, indent=2, default=str)

        # Generate PDF report
        try:
            from creditlens.agents.a4_report_generator.pdf_generator import generate_credit_pdf
            pdf_bytes = generate_credit_pdf(
                report_data=report,
                shap_data=shap_full,
                customer_name=customer_name,
            )
            pdf_path = os.path.join(customer_dir, "credit_report.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            result["pdf_path"] = pdf_path
        except Exception as e:
            result["pdf_error"] = f"{type(e).__name__}: {str(e)}"

        result["status"] = "OK"

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)}"
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


def main():
    # Detect mode
    print("  Mode: REAL (Gemini API)")
    print("=" * 80)

    results = []
    for i, cust in enumerate(CUSTOMERS):
        idx = i + 1
        print(f"\n{'─'*80}")
        print(f"  [{idx}/4] {cust['name']} — {cust['profile']}")
        print(f"  Directory: {cust['dir']}")
        print(f"{'─'*80}")

        r = run_single_customer(cust["dir"], cust["name"])
        results.append(r)

        if r["status"] == "OK":
            print(f"  ✅ Score: {r['credit_score']} | Band: {r['risk_band']} | PD: {r['pd_pct']:.2f}%")
            print(f"  ✅ Decision: {r['routing']}")
            print(f"  ✅ 5C: {r['five_c']} = {r['five_c_total']}/120")
            print(f"  ✅ Consistency: {r['consistency_passed']}")
            print(f"  ✅ Report saved: {r['report_path']}")
            if r.get("pdf_path"):
                print(f"  ✅ PDF saved: {r['pdf_path']}")
            elif r.get("pdf_error"):
                print(f"  ⚠️ PDF failed: {r['pdf_error']}")
        else:
            print(f"  ❌ FAILED: {r['error']}")
            if r.get("traceback"):
                for line in r["traceback"].split("\n")[-5:]:
                    print(f"     {line}")

    # ── Comparison Table ──
    print("\n\n" + "=" * 80)
    print("  COMPARISON SUMMARY")
    print("=" * 80)

    header = f"{'Customer':<20} {'Score':>6} {'Band':>4} {'PD%':>7} {'Decision':>12} {'5C':>5} {'Consistency':>12} {'Status':>8}"
    print(header)
    print("-" * len(header))

    for i, r in enumerate(results):
        if r["status"] == "OK":
            line = f"{CUSTOMERS[i]['name']:<20} {r['credit_score']:>6} {r['risk_band']:>4} {r['pd_pct']:>6.2f}% {r['routing']:>12} {r['five_c_total']:>4}/120 {str(r['consistency_passed']):>12} {'✅':>8}"
        else:
            line = f"{CUSTOMERS[i]['name']:<20} {'—':>6} {'—':>4} {'—':>7} {'—':>12} {'—':>5} {'—':>12} {'❌':>8}"
        print(line)

    # Anomaly checks
    print("\n" + "=" * 80)
    print("  ANOMALY CHECKS")
    print("=" * 80)

    ok_results = [r for r in results if r["status"] == "OK"]

    # Check 1: Scores should have reasonable spread
    if len(ok_results) >= 2:
        scores = [r["credit_score"] for r in ok_results]
        score_range = max(scores) - min(scores)
        if score_range < 50:
            print(f"  ⚠️  Score range too narrow: {score_range} points (all scores: {scores})")
        else:
            print(f"  ✅ Score spread OK: range = {score_range} points ({min(scores)} — {max(scores)})")

    # Check 2: High-risk case should have lower score than standard case
    if len(ok_results) >= 4:
        c1_score = results[0].get("credit_score", 0)
        c4_score = results[3].get("credit_score", 0)
        if c4_score >= c1_score:
            print(f"  ⚠️  High-risk ({c4_score}) >= Standard ({c1_score}). Expected high-risk to score lower!")
        else:
            print(f"  ✅ High-risk ({c4_score}) < Standard ({c1_score}). Scoring logic correct.")

    # Check 3: 5C totals should vary
    if len(ok_results) >= 2:
        totals = [r["five_c_total"] for r in ok_results]
        if len(set(totals)) == 1:
            print(f"  ⚠️  All 5C totals identical: {totals[0]}/120. LLM may not be differentiating.")
        else:
            print(f"  ✅ 5C totals vary: {totals}")

    # Check 4: Consistency should pass for all
    failed_consistency = [CUSTOMERS[i]["name"] for i, r in enumerate(results) if r["status"] == "OK" and not r.get("consistency_passed")]
    if failed_consistency:
        print(f"  ⚠️  Consistency failed for: {', '.join(failed_consistency)}")
    else:
        print(f"  ✅ Consistency passed for all customers")

    # Check 5: Any pipeline failures
    failures = [CUSTOMERS[i]["name"] for i, r in enumerate(results) if r["status"] != "OK"]
    if failures:
        print(f"  ❌ Pipeline FAILED for: {', '.join(failures)}")
    else:
        print(f"  ✅ Pipeline completed for all 4 customers")

    print("\n" + "=" * 80)

    # Save summary JSON
    summary_path = "data/mock/pipeline_test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Summary saved: {summary_path}")

    return results


if __name__ == "__main__":
    main()
