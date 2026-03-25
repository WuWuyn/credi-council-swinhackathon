"""
CREDICOUNCIL — E2E Pipeline Benchmark.

Runs the full A1→A2→A3→A4 pipeline for EVERY customer in data/mock/,
measuring per-customer:
  - Wall-clock time (seconds)
  - Prompt tokens (input)
  - Candidates tokens (output)
  - Total tokens
  - Credit score, PD%, risk band, 5C total

Results are saved to evaluation/e2e/benchmark_results.json
and a human-readable summary is printed to console.

Usage:
    cd back-end
    python -m evaluation.e2e.run_benchmark                 # all customers
    python -m evaluation.e2e.run_benchmark --ids 001 002   # specific customers
    python -m evaluation.e2e.run_benchmark --limit 5       # first N customers
"""

import sys
import os
import io
import json
import logging
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

# ── Ensure back-end/ is on sys.path ─────────────────────────────────────────
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent  # back-end/
sys.path.insert(0, str(BACKEND_ROOT))

# Load .env
try:
    from dotenv import load_dotenv
    env_path = BACKEND_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
MOCK_DIR = BACKEND_ROOT / "data" / "mock"
RESULTS_DIR = BACKEND_ROOT / "evaluation" / "e2e"


def discover_customers(limit: int | None = None, ids: list[str] | None = None) -> list[str]:
    """Discover customer folders in data/mock/.

    Returns list of folder names like ['customer_001', 'customer_002', ...].
    """
    all_folders = sorted([
        d.name for d in MOCK_DIR.iterdir()
        if d.is_dir() and d.name.startswith("customer_")
    ])

    if ids:
        # Filter to specific customer IDs (e.g. "001" → "customer_001")
        targets = set()
        for cid in ids:
            if cid.startswith("customer_"):
                targets.add(cid)
            else:
                targets.add(f"customer_{cid.zfill(3)}")
        all_folders = [f for f in all_folders if f in targets]

    if limit:
        all_folders = all_folders[:limit]

    return all_folders


def run_single_pipeline(customer_dir: str) -> dict:
    """Run the full A1→A2→A3→A4 pipeline for one customer.

    Mirrors test_pipeline.py exactly:
      1. IngestionAgent.ingest(customer_dir)
      2. FeatureEngineerAgent.process(a1_output)
      3. ScoringAgent.score(a2_output)
      4. ReportGeneratorAgent.generate(a3_output, a2_output, a1_output)

    Returns:
        dict with a1_output, a2_output, a3_output, a4_output
    """
    from credicouncil.agents.a1_ingestion.agent import IngestionAgent
    from credicouncil.agents.a2_feature_engineer.agent import FeatureEngineerAgent
    from credicouncil.agents.a3_scoring.agent import ScoringAgent
    from credicouncil.agents.a4_report_generator.agent import ReportGeneratorAgent

    # A1: Data Ingestion
    a1 = IngestionAgent()
    a1_output = a1.ingest(customer_dir=customer_dir)

    # A2: Feature Engineering
    a2 = FeatureEngineerAgent()
    a2_output = a2.process(a1_output)

    # A3: ML Scoring
    a3 = ScoringAgent(model_path=str(BACKEND_ROOT / "models" / "lgbm_ref_v1.pkl"))
    a3_output = a3.score(a2_output)

    # A4: Report Generation
    a4 = ReportGeneratorAgent()
    a4_output = a4.generate(a3_output, a2_output, a1_output)

    return {
        "a1_output": a1_output,
        "a2_output": a2_output,
        "a3_output": a3_output,
        "a4_output": a4_output,
    }


def benchmark_customer(folder_name: str) -> dict:
    """Benchmark a single customer's pipeline run.

    Measures wall-clock time and LLM token usage.

    Returns:
        dict with timing, token counts, and scoring results.
    """
    from credicouncil.services.llm_service import reset_token_counter, get_token_counts

    customer_dir = str(MOCK_DIR / folder_name)

    # Reset token counter before this customer
    reset_token_counter()

    start_time = time.time()
    error = None
    result_data = {}

    try:
        result = run_single_pipeline(customer_dir)

        a3 = result["a3_output"]
        a4 = result["a4_output"]
        report = a4.get("final_report", {})
        executive = report.get("executive_summary", {})
        five_c_scores = a4.get("five_c_scores", {})
        if not five_c_scores:
            five_c_scores = executive.get("five_c_scores", {})

        result_data = {
            "credit_score": a3.get("credit_score", 0),
            "pd_pct": a3.get("pd_pct", 0.0),
            "risk_band": a3.get("risk_band", "N/A"),
            "routing": a3.get("routing", "N/A"),
            "five_c_scores": five_c_scores,
            "five_c_total": sum(five_c_scores.values()) if five_c_scores else 0,
            "recommendation": executive.get("recommendation", a3.get("routing", "N/A")),
        }
    except Exception as e:
        logger.error(f"Pipeline FAILED for {folder_name}: {e}")
        error = str(e)

    elapsed = round(time.time() - start_time, 2)
    tokens = get_token_counts()

    return {
        "customer_id": folder_name,
        "status": "SUCCESS" if error is None else "FAILED",
        "error": error,
        "elapsed_seconds": elapsed,
        "prompt_tokens": tokens["prompt_tokens"],
        "candidates_tokens": tokens["candidates_tokens"],
        "total_tokens": tokens["total_tokens"],
        **result_data,
    }


def run_benchmark(
    limit: int | None = None,
    ids: list[str] | None = None,
    output_file: str | None = None,
    delay: float = 5.0,
) -> dict:
    """Run the full benchmark across all (or selected) customers.

    Args:
        limit: Max number of customers to process.
        ids: Specific customer IDs to process.
        output_file: Path to save results JSON. Defaults to benchmark_results.json.
        delay: Seconds to wait between each customer run (default 5s).

    Returns:
        Full benchmark summary dict.
    """
    customers = discover_customers(limit=limit, ids=ids)
    total = len(customers)

    if total == 0:
        print("❌ No customers found in data/mock/")
        return {}

    print("\n" + "=" * 70)
    print("  CREDICOUNCIL — E2E PIPELINE BENCHMARK")
    print("=" * 70)
    print(f"  Customers: {total}")
    print(f"  Mock dir:  {MOCK_DIR}")
    print(f"  Delay:     {delay}s between customers")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    benchmark_start = time.time()
    results = []

    for idx, folder_name in enumerate(customers, 1):
        print(f"\n{'─' * 60}")
        print(f"  [{idx}/{total}] Processing {folder_name}...")
        print(f"{'─' * 60}")

        customer_result = benchmark_customer(folder_name)
        results.append(customer_result)

        # Print per-customer summary
        status = customer_result["status"]
        elapsed = customer_result["elapsed_seconds"]
        prompt_tk = customer_result["prompt_tokens"]
        cand_tk = customer_result["candidates_tokens"]
        total_tk = customer_result["total_tokens"]

        if status == "SUCCESS":
            score = customer_result.get("credit_score", "—")
            band = customer_result.get("risk_band", "—")
            five_c = customer_result.get("five_c_total", "—")
            print(f"  ✅ {folder_name}: Score={score}, Band={band}, 5C={five_c}/120")
        else:
            print(f"  ❌ {folder_name}: FAILED — {customer_result.get('error', 'Unknown')}")

        print(f"     ⏱ {elapsed}s | Tokens: {prompt_tk} in + {cand_tk} out = {total_tk} total")

        # Delay between customers (skip after last one)
        if idx < total and delay > 0:
            print(f"     💤 Waiting {delay}s before next customer...")
            time.sleep(delay)

    total_elapsed = round(time.time() - benchmark_start, 2)

    # ── Aggregate statistics ──────────────────────────────────────────────
    success_results = [r for r in results if r["status"] == "SUCCESS"]
    failed_results = [r for r in results if r["status"] == "FAILED"]

    agg_prompt = sum(r["prompt_tokens"] for r in results)
    agg_cand = sum(r["candidates_tokens"] for r in results)
    agg_total = sum(r["total_tokens"] for r in results)
    agg_time = sum(r["elapsed_seconds"] for r in results)

    avg_time = round(agg_time / total, 2) if total > 0 else 0
    avg_prompt = round(agg_prompt / total, 0) if total > 0 else 0
    avg_cand = round(agg_cand / total, 0) if total > 0 else 0
    avg_total = round(agg_total / total, 0) if total > 0 else 0

    summary = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_customers": total,
        "success_count": len(success_results),
        "failed_count": len(failed_results),
        "total_wall_clock_seconds": total_elapsed,
        "aggregated": {
            "sum_elapsed_seconds": agg_time,
            "avg_elapsed_seconds": avg_time,
            "sum_prompt_tokens": agg_prompt,
            "sum_candidates_tokens": agg_cand,
            "sum_total_tokens": agg_total,
            "avg_prompt_tokens": int(avg_prompt),
            "avg_candidates_tokens": int(avg_cand),
            "avg_total_tokens": int(avg_total),
        },
        "per_customer": results,
    }

    # ── Save results ──────────────────────────────────────────────────────
    if output_file is None:
        output_file = str(RESULTS_DIR / "benchmark_results.json")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Customers processed:  {total}")
    print(f"  Succeeded:            {len(success_results)}")
    print(f"  Failed:               {len(failed_results)}")
    print(f"  Total wall-clock:     {total_elapsed}s")
    print(f"  Avg time/customer:    {avg_time}s")
    print()
    print(f"  ── Token Usage ──")
    print(f"  Total prompt (input):     {agg_prompt:>10,}")
    print(f"  Total candidates (output):{agg_cand:>10,}")
    print(f"  Total tokens:             {agg_total:>10,}")
    print(f"  Avg prompt/customer:      {int(avg_prompt):>10,}")
    print(f"  Avg candidates/customer:  {int(avg_cand):>10,}")
    print(f"  Avg total/customer:       {int(avg_total):>10,}")
    print()

    # Per-customer table
    print(f"  {'Customer':<16} {'Status':<8} {'Time(s)':<10} {'Input Tk':<12} {'Output Tk':<12} {'Total Tk':<12} {'Score':<8} {'Band':<6}")
    print(f"  {'─'*16} {'─'*8} {'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*8} {'─'*6}")
    for r in results:
        cid = r["customer_id"].replace("customer_", "")
        status = "✅" if r["status"] == "SUCCESS" else "❌"
        t = f"{r['elapsed_seconds']:.1f}"
        p = f"{r['prompt_tokens']:,}"
        c = f"{r['candidates_tokens']:,}"
        tot = f"{r['total_tokens']:,}"
        score = str(r.get("credit_score", "—"))
        band = r.get("risk_band", "—")
        print(f"  {cid:<16} {status:<8} {t:<10} {p:<12} {c:<12} {tot:<12} {score:<8} {band:<6}")

    # Average row
    print(f"  {'─'*16} {'─'*8} {'─'*10} {'─'*12} {'─'*12} {'─'*12} {'─'*8} {'─'*6}")
    avg_t = f"{avg_time:.1f}"
    avg_p = f"{int(avg_prompt):,}"
    avg_c = f"{int(avg_cand):,}"
    avg_tot = f"{int(avg_total):,}"
    print(f"  {'AVERAGE':<16} {'—':<8} {avg_t:<10} {avg_p:<12} {avg_c:<12} {avg_tot:<12} {'—':<8} {'—':<6}")

    print(f"\n  Results saved: {output_file}")
    print("=" * 70)

    return summary


def main():
    # Force UTF-8 stdout (Windows compatibility) — only when run as script
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="CrediCouncil E2E Pipeline Benchmark")
    parser.add_argument(
        "--ids", nargs="+", default=None,
        help="Specific customer IDs to benchmark (e.g. 001 002 010)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of customers to process"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: evaluation/e2e/benchmark_results.json)"
    )
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="Seconds to wait between each customer run (default: 3.0)"
    )
    args = parser.parse_args()

    run_benchmark(limit=args.limit, ids=args.ids, output_file=args.output, delay=args.delay)


if __name__ == "__main__":
    main()
