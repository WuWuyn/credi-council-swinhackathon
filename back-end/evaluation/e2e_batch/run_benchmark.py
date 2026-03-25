"""
CREDICOUNCIL — E2E Batch Pipeline Benchmark.

Runs the FULL A1→A2→A3→A4 pipeline in parallel threads (same as test_batch_pipeline.py),
with staggered starts. Groups of 5 customers run simultaneously, 50 customers = 10 batches.

Additional tracking vs test_batch_pipeline.py:
  - LLM token usage (prompt, candidates, total) per customer
  - Per-agent timing (A1, A2, A3, A4 individually)
  - Batch-level and overall aggregated statistics
  - JSON report export

Usage:
    cd back-end
    python -m evaluation.e2e_batch.run_benchmark                  # all 50 customers
    python -m evaluation.e2e_batch.run_benchmark --limit 10       # first 10
    python -m evaluation.e2e_batch.run_benchmark --batch-size 3   # 3 per batch
    python -m evaluation.e2e_batch.run_benchmark --stagger 5      # 5s stagger
    python -m evaluation.e2e_batch.run_benchmark --delay 10       # 10s between batches
"""

import sys
import os
import io
import json
import logging
import time
import argparse
import threading
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
RESULTS_DIR = BACKEND_ROOT / "evaluation" / "e2e_batch"


def discover_customers(limit: int | None = None, ids: list[str] | None = None) -> list[str]:
    """Discover customer folders in data/mock/."""
    all_folders = sorted([
        d.name for d in MOCK_DIR.iterdir()
        if d.is_dir() and d.name.startswith("customer_")
    ])

    if ids:
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


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split a list into chunks of given size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLE PIPELINE — mirrors test_batch_pipeline.py exactly + token tracking
# ═══════════════════════════════════════════════════════════════════════════════


def run_single_pipeline(customer_dir: str, index: int, total: int) -> dict:
    """Run full A1→A2→A3→A4 pipeline for one customer in current thread.

    Same as test_batch_pipeline.py's run_single_pipeline() but with
    per-agent timing and LLM token tracking.
    """
    from credicouncil.agents.a1_ingestion.agent import IngestionAgent
    from credicouncil.agents.a2_feature_engineer.agent import FeatureEngineerAgent
    from credicouncil.agents.a3_scoring.agent import ScoringAgent
    from credicouncil.agents.a4_report_generator.agent import ReportGeneratorAgent
    from credicouncil.services.llm_service import reset_token_counter, get_token_counts

    folder_name = os.path.basename(customer_dir)

    # Reset token counter for this customer
    reset_token_counter()

    start = time.time()
    logger.info(f"[Pipeline #{index+1}/{total}] START — {folder_name}")

    try:
        # A1: Data Ingestion
        a1_start = time.time()
        a1 = IngestionAgent()
        a1_output = a1.ingest(customer_dir=customer_dir)
        a1_time = round(time.time() - a1_start, 2)
        logger.info(
            f"[Pipeline #{index+1}/{total}] A1 done in {a1_time:.1f}s — "
            f"{len(a1_output['application_row'])} fields"
        )

        # A2: Feature Engineering
        a2_start = time.time()
        a2 = FeatureEngineerAgent()
        a2_output = a2.process(a1_output)
        a2_time = round(time.time() - a2_start, 2)
        logger.info(f"[Pipeline #{index+1}/{total}] A2 done in {a2_time:.1f}s")

        # A3: ML Scoring
        a3_start = time.time()
        a3 = ScoringAgent(model_path=str(BACKEND_ROOT / "models" / "lgbm_ref_v1.pkl"))
        a3_output = a3.score(a2_output)
        a3_time = round(time.time() - a3_start, 2)
        logger.info(
            f"[Pipeline #{index+1}/{total}] A3 done in {a3_time:.1f}s — "
            f"Score={a3_output['credit_score']}"
        )

        # A4: Report Generation
        a4_start = time.time()
        a4 = ReportGeneratorAgent()
        a4_output = a4.generate(a3_output, a2_output, a1_output)
        a4_time = round(time.time() - a4_start, 2)
        logger.info(f"[Pipeline #{index+1}/{total}] A4 done in {a4_time:.1f}s")

        elapsed = round(time.time() - start, 2)
        tokens = get_token_counts()

        # Extract 5C scores
        five_c_scores = a4_output.get("five_c_scores", {})
        report = a4_output.get("final_report", {})
        executive = report.get("executive_summary", {})
        if not five_c_scores:
            five_c_scores = executive.get("five_c_scores", {})

        logger.info(
            f"[Pipeline #{index+1}/{total}] ✅ DONE in {elapsed:.1f}s — "
            f"Score={a3_output['credit_score']}, Band={a3_output['risk_band']}, "
            f"Tokens={tokens['total_tokens']:,}"
        )

        return {
            "customer_id": folder_name,
            "status": "SUCCESS",
            "elapsed_seconds": elapsed,
            "a1_seconds": a1_time,
            "a2_seconds": a2_time,
            "a3_seconds": a3_time,
            "a4_seconds": a4_time,
            "prompt_tokens": tokens["prompt_tokens"],
            "candidates_tokens": tokens["candidates_tokens"],
            "total_tokens": tokens["total_tokens"],
            "credit_score": a3_output.get("credit_score", 0),
            "pd_pct": a3_output.get("pd_pct", 0.0),
            "risk_band": a3_output.get("risk_band", "N/A"),
            "routing": a3_output.get("routing", "N/A"),
            "five_c_total": sum(five_c_scores.values()) if five_c_scores else 0,
            "recommendation": executive.get("recommendation", a3_output.get("routing", "N/A")),
        }

    except Exception as e:
        elapsed = round(time.time() - start, 2)
        tokens = get_token_counts()
        logger.error(f"[Pipeline #{index+1}/{total}] ❌ FAILED after {elapsed:.1f}s — {e}")
        return {
            "customer_id": folder_name,
            "status": "FAILED",
            "elapsed_seconds": elapsed,
            "a1_seconds": 0,
            "a2_seconds": 0,
            "a3_seconds": 0,
            "a4_seconds": 0,
            "prompt_tokens": tokens["prompt_tokens"],
            "candidates_tokens": tokens["candidates_tokens"],
            "total_tokens": tokens["total_tokens"],
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH RUNNER — threading.Thread + stagger (same as test_batch_pipeline.py)
# ═══════════════════════════════════════════════════════════════════════════════


def run_one_batch(
    customer_folders: list[str],
    batch_index: int,
    stagger_delay: float = 3.0,
) -> list[dict]:
    """Run a batch of customers in parallel threads with staggered starts.

    Same mechanism as test_batch_pipeline.py:
    - One threading.Thread per customer
    - Stagger launches by stagger_delay seconds
    - Full A1→A2→A3→A4 runs in each thread
    """
    total = len(customer_folders)
    threads: list[threading.Thread] = []
    results: list[dict | None] = [None] * total

    def _run_thread(idx: int, folder: str):
        customer_dir = str(MOCK_DIR / folder)
        results[idx] = run_single_pipeline(customer_dir, idx, total)

    for i, folder in enumerate(customer_folders):
        t = threading.Thread(target=_run_thread, args=(i, folder))
        threads.append(t)
        t.start()
        logger.info(f"  [Batch {batch_index}] Launched #{i+1}/{total}: {folder}")

        if i < total - 1 and stagger_delay > 0:
            time.sleep(stagger_delay)

    # Wait for all threads to finish
    for t in threads:
        t.join()

    return [r for r in results if r is not None]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════


def run_batch_benchmark(
    limit: int | None = None,
    ids: list[str] | None = None,
    batch_size: int = 5,
    stagger_delay: float = 3.0,
    delay_between_batches: float = 3.0,
    output_file: str | None = None,
) -> dict:
    """Run the full batch benchmark.

    50 customers → 10 batches × 5 customers.
    Each batch uses threading.Thread with staggered starts (same as test_batch_pipeline.py).
    """
    customers = discover_customers(limit=limit, ids=ids)
    total = len(customers)

    if total == 0:
        print("❌ No customers found in data/mock/")
        return {}

    batches = chunk_list(customers, batch_size)
    num_batches = len(batches)

    print("\n" + "=" * 80)
    print("  CREDICOUNCIL — BATCH PIPELINE BENCHMARK")
    print("=" * 80)
    print(f"  Total customers:    {total}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Num batches:        {num_batches}")
    print(f"  Stagger delay:      {stagger_delay}s (between threads in batch)")
    print(f"  Delay between:      {delay_between_batches}s (between batches)")
    print(f"  Pipeline mode:      Full A1→A2→A3→A4 per thread (test_batch_pipeline style)")
    print(f"  Started:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    overall_start = time.time()
    all_customer_results: list[dict] = []
    batch_summaries: list[dict] = []

    for batch_idx, batch_customers in enumerate(batches, 1):
        batch_ids_str = ", ".join(c.replace("customer_", "") for c in batch_customers)

        print(f"\n{'━' * 80}")
        print(f"  BATCH {batch_idx}/{num_batches} — [{batch_ids_str}] ({len(batch_customers)} customers)")
        print(f"{'━' * 80}")

        batch_start = time.time()
        batch_results = run_one_batch(batch_customers, batch_idx, stagger_delay)
        batch_elapsed = round(time.time() - batch_start, 2)

        # Batch metrics
        success = [r for r in batch_results if r.get("status") == "SUCCESS"]
        batch_prompt = sum(r.get("prompt_tokens", 0) for r in batch_results)
        batch_cand = sum(r.get("candidates_tokens", 0) for r in batch_results)
        batch_tok = sum(r.get("total_tokens", 0) for r in batch_results)

        # Sequential estimate (sum of individual times)
        seq_est = sum(r.get("elapsed_seconds", 0) for r in batch_results)
        speedup = round(seq_est / batch_elapsed, 2) if batch_elapsed > 0 else 0

        batch_summaries.append({
            "batch_index": batch_idx,
            "customer_ids": [c.replace("customer_", "") for c in batch_customers],
            "batch_size": len(batch_customers),
            "wall_clock_seconds": batch_elapsed,
            "sequential_estimate_seconds": round(seq_est, 2),
            "speedup": speedup,
            "success_count": len(success),
            "failed_count": len(batch_results) - len(success),
            "prompt_tokens": batch_prompt,
            "candidates_tokens": batch_cand,
            "total_tokens": batch_tok,
        })
        all_customer_results.extend(batch_results)

        # Print batch summary
        print(f"\n  ── Batch {batch_idx} Summary ──")
        print(f"  Wall-clock:       {batch_elapsed:.1f}s")
        print(f"  Sequential est:   {seq_est:.1f}s")
        print(f"  Speedup:          {speedup}x")
        print(f"  Success:          {len(success)}/{len(batch_customers)}")
        print(f"  Tokens:           {batch_tok:,}")

        for r in batch_results:
            cid = r["customer_id"].replace("customer_", "")
            st = "✅" if r.get("status") == "SUCCESS" else "❌"
            t = f"{r.get('elapsed_seconds', 0):.1f}s"
            a1 = f"{r.get('a1_seconds', 0):.1f}"
            a2 = f"{r.get('a2_seconds', 0):.1f}"
            a3 = f"{r.get('a3_seconds', 0):.1f}"
            a4 = f"{r.get('a4_seconds', 0):.1f}"
            tk = f"{r.get('total_tokens', 0):,}"
            score = str(r.get("credit_score", "—"))
            band = r.get("risk_band", "—")
            print(
                f"    {st} {cid:<10} Total={t:<7} "
                f"A1={a1}s A2={a2}s A3={a3}s A4={a4}s  "
                f"Tokens={tk:<10} Score={score:<5} Band={band}"
            )

        # Delay between batches
        if batch_idx < num_batches and delay_between_batches > 0:
            print(f"\n  💤 Waiting {delay_between_batches}s before next batch...")
            time.sleep(delay_between_batches)

    overall_elapsed = round(time.time() - overall_start, 2)

    # ═══════════════════════════════════════════════════════════════════════
    # AGGREGATE STATISTICS
    # ═══════════════════════════════════════════════════════════════════════

    all_success = [r for r in all_customer_results if r.get("status") == "SUCCESS"]
    all_failed = [r for r in all_customer_results if r.get("status") != "SUCCESS"]
    n = max(len(all_customer_results), 1)

    agg_elapsed = sum(r.get("elapsed_seconds", 0) for r in all_customer_results)
    agg_a1 = sum(r.get("a1_seconds", 0) for r in all_customer_results)
    agg_a2 = sum(r.get("a2_seconds", 0) for r in all_customer_results)
    agg_a3 = sum(r.get("a3_seconds", 0) for r in all_customer_results)
    agg_a4 = sum(r.get("a4_seconds", 0) for r in all_customer_results)
    agg_prompt = sum(r.get("prompt_tokens", 0) for r in all_customer_results)
    agg_cand = sum(r.get("candidates_tokens", 0) for r in all_customer_results)
    agg_tok = sum(r.get("total_tokens", 0) for r in all_customer_results)

    avg_elapsed = round(agg_elapsed / n, 2)
    avg_a1 = round(agg_a1 / n, 2)
    avg_a2 = round(agg_a2 / n, 2)
    avg_a3 = round(agg_a3 / n, 2)
    avg_a4 = round(agg_a4 / n, 2)
    avg_prompt = int(round(agg_prompt / n))
    avg_cand = int(round(agg_cand / n))
    avg_tok = int(round(agg_tok / n))

    nb = max(num_batches, 1)
    avg_batch_wall = round(sum(b["wall_clock_seconds"] for b in batch_summaries) / nb, 2)
    avg_batch_speedup = round(sum(b["speedup"] for b in batch_summaries) / nb, 2)

    # Overall speedup: sequential total vs actual wall-clock
    overall_speedup = round(agg_elapsed / overall_elapsed, 2) if overall_elapsed > 0 else 0

    summary = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "total_customers": total,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "stagger_delay": stagger_delay,
            "delay_between_batches": delay_between_batches,
            "pipeline_mode": "full_parallel_A1_A2_A3_A4",
        },
        "overall": {
            "wall_clock_seconds": overall_elapsed,
            "sequential_estimate_seconds": round(agg_elapsed, 2),
            "speedup": overall_speedup,
            "success_count": len(all_success),
            "failed_count": len(all_failed),
        },
        "aggregated_customer": {
            "avg_elapsed_seconds": avg_elapsed,
            "avg_a1_seconds": avg_a1,
            "avg_a2_seconds": avg_a2,
            "avg_a3_seconds": avg_a3,
            "avg_a4_seconds": avg_a4,
            "sum_prompt_tokens": agg_prompt,
            "sum_candidates_tokens": agg_cand,
            "sum_total_tokens": agg_tok,
            "avg_prompt_tokens": avg_prompt,
            "avg_candidates_tokens": avg_cand,
            "avg_total_tokens": avg_tok,
        },
        "aggregated_batch": {
            "avg_wall_clock_seconds": avg_batch_wall,
            "avg_speedup": avg_batch_speedup,
        },
        "batches": batch_summaries,
        "per_customer": all_customer_results,
    }

    # ── Save results ──
    if output_file is None:
        output_file = str(RESULTS_DIR / "batch_benchmark_results.json")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

    # ═══════════════════════════════════════════════════════════════════════
    # PRINT FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 80)
    print("  BATCH BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"  Total customers:      {total}")
    print(f"  Batches:              {num_batches} × {batch_size}")
    print(f"  Succeeded:            {len(all_success)}")
    print(f"  Failed:               {len(all_failed)}")
    print(f"  Total wall-clock:     {overall_elapsed}s")
    print(f"  Sequential estimate:  {round(agg_elapsed, 1)}s")
    print(f"  Overall speedup:      {overall_speedup}x")
    print()
    print(f"  ── Batch Averages ──")
    print(f"  Avg batch wall-clock: {avg_batch_wall}s")
    print(f"  Avg batch speedup:    {avg_batch_speedup}x")
    print()
    print(f"  ── Customer Averages ──")
    print(f"  Avg total time:       {avg_elapsed}s")
    print(f"  Avg A1 time:          {avg_a1}s")
    print(f"  Avg A2 time:          {avg_a2}s")
    print(f"  Avg A3 time:          {avg_a3}s")
    print(f"  Avg A4 time:          {avg_a4}s")
    print(f"  Avg prompt tokens:    {avg_prompt:,}")
    print(f"  Avg output tokens:    {avg_cand:,}")
    print(f"  Avg total tokens:     {avg_tok:,}")
    print()
    print(f"  ── Token Usage (Total) ──")
    print(f"  Total prompt:         {agg_prompt:>12,}")
    print(f"  Total candidates:     {agg_cand:>12,}")
    print(f"  Total tokens:         {agg_tok:>12,}")

    # Per-customer table
    print()
    print(
        f"  {'Customer':<12} {'St':<3} {'Total':<7} {'A1':<6} {'A2':<6} "
        f"{'A3':<6} {'A4':<6} {'In Tk':<9} {'Out Tk':<9} {'All Tk':<9} "
        f"{'Score':<6} {'Band':<5}"
    )
    sep = "─" * 95
    print(f"  {sep}")

    for r in all_customer_results:
        cid = r["customer_id"].replace("customer_", "")
        st = "✅" if r.get("status") == "SUCCESS" else "❌"
        tot = f"{r.get('elapsed_seconds', 0):.1f}"
        a1 = f"{r.get('a1_seconds', 0):.1f}"
        a2 = f"{r.get('a2_seconds', 0):.1f}"
        a3 = f"{r.get('a3_seconds', 0):.1f}"
        a4 = f"{r.get('a4_seconds', 0):.1f}"
        pt = f"{r.get('prompt_tokens', 0):,}"
        ct = f"{r.get('candidates_tokens', 0):,}"
        tt = f"{r.get('total_tokens', 0):,}"
        sc = str(r.get("credit_score", "—"))
        bd = r.get("risk_band", "—")
        print(
            f"  {cid:<12} {st:<3} {tot:<7} {a1:<6} {a2:<6} "
            f"{a3:<6} {a4:<6} {pt:<9} {ct:<9} {tt:<9} "
            f"{sc:<6} {bd:<5}"
        )

    # Average row
    print(f"  {sep}")
    print(
        f"  {'AVERAGE':<12} {'—':<3} {avg_elapsed:<7} {avg_a1:<6} {avg_a2:<6} "
        f"{avg_a3:<6} {avg_a4:<6} {avg_prompt:<9,} {avg_cand:<9,} {avg_tok:<9,} "
        f"{'—':<6} {'—':<5}"
    )

    print(f"\n  Results saved: {output_file}")
    print("=" * 80)

    return summary


def main():
    # Force UTF-8 stdout (Windows compatibility)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="CrediCouncil Batch Pipeline Benchmark (test_batch_pipeline style)"
    )
    parser.add_argument(
        "--ids", nargs="+", default=None,
        help="Specific customer IDs to benchmark (e.g. 001 002 010)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of customers to process"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5,
        help="Number of customers per batch (default: 5)"
    )
    parser.add_argument(
        "--stagger", type=float, default=3.0,
        help="Stagger delay between thread launches in a batch (default: 3.0s)"
    )
    parser.add_argument(
        "--delay", type=float, default=3.0,
        help="Delay between batches (default: 3.0s)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path"
    )
    args = parser.parse_args()

    run_batch_benchmark(
        limit=args.limit,
        ids=args.ids,
        batch_size=args.batch_size,
        stagger_delay=args.stagger,
        delay_between_batches=args.delay,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
