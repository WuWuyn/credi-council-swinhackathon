"""
CREDICOUNCIL — Batch Pipeline Test (4 requests, staggered 3s apart).

Tests parallel processing of 4 customer pipelines with:
- 3s stagger delay between each pipeline start
- Retry + exponential backoff on rate limit errors
- Detailed timing and success/failure reporting
"""

import sys, os, io, json, logging, time, threading
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env
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

# ── Configuration ────────────────────────────────────────────────────
CUSTOMERS = [
    "data/mock/customer_001",  
    "data/mock/customer_005",  
    "data/mock/customer_017",  
    "data/mock/customer_025",  
]
STAGGER_DELAY = 3.0  # seconds between each pipeline start


def run_single_pipeline(customer_dir: str, index: int) -> dict:
    """Run full A1→A2→A3→A4 pipeline for one customer.

    Returns dict with timing, results, and any errors.
    """
    customer_id = os.path.basename(customer_dir)
    result = {
        "customer_id": customer_id,
        "index": index,
        "status": "PENDING",
        "start_time": None,
        "end_time": None,
        "duration_s": None,
        "credit_score": None,
        "risk_band": None,
        "recommendation": None,
        "error": None,
    }

    start = time.time()
    result["start_time"] = datetime.now().strftime("%H:%M:%S")
    logger.info(f"[Pipeline #{index+1}] START — {customer_id}")

    try:
        # A1: Ingestion
        from credicouncil.agents.a1_ingestion.agent import IngestionAgent
        a1 = IngestionAgent()
        a1_output = a1.ingest(customer_dir=customer_dir)
        a1_time = time.time() - start
        logger.info(f"[Pipeline #{index+1}] A1 done in {a1_time:.1f}s — {len(a1_output['application_row'])} fields")

        # A2: Feature Engineering
        from credicouncil.agents.a2_feature_engineer.agent import FeatureEngineerAgent
        a2 = FeatureEngineerAgent()
        a2_output = a2.process(a1_output)
        a2_time = time.time() - start
        logger.info(f"[Pipeline #{index+1}] A2 done in {a2_time:.1f}s")

        # A3: Scoring
        from credicouncil.agents.a3_scoring.agent import ScoringAgent
        a3 = ScoringAgent(model_path="models/lgbm_ref_v1.pkl")
        a3_output = a3.score(a2_output)
        a3_time = time.time() - start
        logger.info(f"[Pipeline #{index+1}] A3 done in {a3_time:.1f}s — Score={a3_output['credit_score']}")

        # A4: Report Generation
        from credicouncil.agents.a4_report_generator.agent import ReportGeneratorAgent
        a4 = ReportGeneratorAgent()
        a4_output = a4.generate(a3_output, a2_output, a1_output)
        a4_time = time.time() - start
        logger.info(f"[Pipeline #{index+1}] A4 done in {a4_time:.1f}s")

        # Save report
        report = a4_output.get("final_report", {})
        report_path = os.path.join(customer_dir, "credit_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)

        # Generate PDF
        try:
            from credicouncil.agents.a4_report_generator.pdf_generator import generate_credit_pdf
            shap_data = a3_output.get("shap_values", {})
            pdf_bytes = generate_credit_pdf(
                report_data=report,
                shap_data=shap_data,
                customer_name=f"Customer_{customer_id}",
            )
            pdf_path = os.path.join(customer_dir, "credit_report.pdf")
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
        except Exception as e:
            logger.warning(f"[Pipeline #{index+1}] PDF generation failed: {e}")

        duration = time.time() - start
        result.update({
            "status": "SUCCESS",
            "end_time": datetime.now().strftime("%H:%M:%S"),
            "duration_s": round(duration, 1),
            "credit_score": a3_output["credit_score"],
            "risk_band": a3_output["risk_band"],
            "pd_pct": round(a3_output["pd_pct"], 2),
            "recommendation": report.get("executive_summary", {}).get("recommendation"),
            "five_c_total": sum(a4_output.get("five_c_scores", {}).values()),
        })
        logger.info(
            f"[Pipeline #{index+1}] ✅ DONE in {duration:.1f}s — "
            f"Score={result['credit_score']}, Band={result['risk_band']}, "
            f"Decision={result['recommendation']}"
        )

    except Exception as e:
        duration = time.time() - start
        result.update({
            "status": "FAILED",
            "end_time": datetime.now().strftime("%H:%M:%S"),
            "duration_s": round(duration, 1),
            "error": str(e),
        })
        logger.error(f"[Pipeline #{index+1}] ❌ FAILED after {duration:.1f}s — {e}")

    return result


def run_batch():
    """Run 4 pipelines with staggered starts (3s apart)."""
    batch_start = time.time()

    print("\n" + "=" * 70)
    print("  CREDICOUNCIL — BATCH PIPELINE TEST")
    print(f"  {len(CUSTOMERS)} customers, stagger={STAGGER_DELAY}s")
    print("=" * 70)

    # Launch threads with stagger delay
    threads = []
    results = [None] * len(CUSTOMERS)

    def _run_thread(idx, cust_dir):
        results[idx] = run_single_pipeline(cust_dir, idx)

    for i, cust_dir in enumerate(CUSTOMERS):
        t = threading.Thread(target=_run_thread, args=(i, cust_dir))
        threads.append(t)
        t.start()
        logger.info(f"  Launched pipeline #{i+1} for {os.path.basename(cust_dir)}")
        if i < len(CUSTOMERS) - 1:
            logger.info(f"  Waiting {STAGGER_DELAY}s before next launch...")
            time.sleep(STAGGER_DELAY)

    # Wait for all to finish
    logger.info("  All pipelines launched. Waiting for completion...")
    for t in threads:
        t.join()

    batch_duration = time.time() - batch_start

    # ── Results Summary ──
    print("\n" + "=" * 70)
    print("  BATCH RESULTS")
    print("=" * 70)
    print(f"  {'#':<3} {'Customer':<15} {'Status':<10} {'Time':<8} {'Score':<7} {'Band':<6} {'Decision':<15} {'5C':<5}")
    print("  " + "-" * 68)

    success_count = 0
    for r in results:
        if r is None:
            continue
        status_icon = "✅" if r["status"] == "SUCCESS" else "❌"
        if r["status"] == "SUCCESS":
            success_count += 1
            print(
                f"  {r['index']+1:<3} {r['customer_id']:<15} {status_icon} "
                f"{r['duration_s']:<8} {r.get('credit_score', 'N/A'):<7} "
                f"{r.get('risk_band', 'N/A'):<6} {r.get('recommendation', 'N/A'):<15} "
                f"{r.get('five_c_total', 'N/A')}"
            )
        else:
            print(
                f"  {r['index']+1:<3} {r['customer_id']:<15} {status_icon} "
                f"{r['duration_s']:<8} {'FAILED':<7} {'—':<6} {'—':<15} "
                f"Error: {r.get('error', 'unknown')[:40]}"
            )

    print("  " + "-" * 68)
    print(f"  Total batch time: {batch_duration:.1f}s")
    print(f"  Success: {success_count}/{len(CUSTOMERS)}")
    print(f"  Avg time per pipeline: {batch_duration/len(CUSTOMERS):.1f}s")
    if success_count == len(CUSTOMERS):
        sequential_est = sum(r["duration_s"] for r in results if r)
        speedup = sequential_est / batch_duration if batch_duration > 0 else 0
        print(f"  Estimated sequential time: {sequential_est:.1f}s")
        print(f"  Speedup: {speedup:.1f}x")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_batch()
