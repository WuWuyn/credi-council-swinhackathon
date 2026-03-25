"""
CREDICOUNCIL — E2E Batch Pipeline Benchmark.

Measures performance of the BATCH pipeline (groups of 5 customers in parallel).
With 50 customers, runs 10 batches × 5 customers each.

Matches the frontend /ws/batch flow:
  Phase 1: A1 ingestion (parallel, 5 workers, staggered)
  Phase 2: Skip human review (auto-approve)
  Phase 3: A2→A3→A4 processing (parallel, 5 workers, staggered)

Metrics per batch:
  - Wall-clock time for Phase 1 (A1 parallel)
  - Wall-clock time for Phase 3 (A2→A4 parallel)
  - Total wall-clock time (Phase 1 + Phase 3)
  - LLM tokens (prompt, candidates, total)
  - Per-customer: score, risk band, 5C total, time, tokens

Results saved to evaluation/e2e_batch/batch_benchmark_results.json

Usage:
    cd back-end
    python -m evaluation.e2e_batch.run_benchmark                  # all 50 customers
    python -m evaluation.e2e_batch.run_benchmark --limit 10       # first 10 customers
    python -m evaluation.e2e_batch.run_benchmark --batch-size 3   # 3 per batch
    python -m evaluation.e2e_batch.run_benchmark --delay 5        # 5s between batches
"""

import sys
import os
import io
import json
import logging
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    """Discover customer folders in data/mock/.

    Returns list of folder names like ['customer_001', 'customer_002', ...].
    """
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
# PHASE 1: Parallel A1 Ingestion (mirrors execute_batch_ingestion)
# ═══════════════════════════════════════════════════════════════════════════════


def run_phase1_parallel(
    customer_folders: list[str],
    max_workers: int = 5,
    stagger_delay: float = 1.0,
) -> list[dict]:
    """Run A1 ingestion in parallel with staggered starts.

    Mirrors the frontend /ws/batch Phase 1 behavior exactly:
    - Uses ThreadPoolExecutor with max_workers
    - Staggers launches by stagger_delay seconds
    - Each thread runs IngestionAgent.ingest()

    Returns list of per-customer results with a1_output and timing.
    """
    from credicouncil.agents.a1_ingestion.agent import IngestionAgent
    from credicouncil.services.llm_service import reset_token_counter, get_token_counts

    total = len(customer_folders)
    results_map: dict[str, dict] = {}

    def _ingest_one(folder_name: str, idx: int) -> dict:
        customer_dir = str(MOCK_DIR / folder_name)
        start = time.time()
        try:
            a1 = IngestionAgent()
            a1_output = a1.ingest(customer_dir=customer_dir)
            elapsed = round(time.time() - start, 2)
            field_count = len(a1_output.get("application_row", {}))
            logger.info(
                f"  [A1 #{idx+1}/{total}] ✅ {folder_name} "
                f"({field_count} fields) in {elapsed:.1f}s"
            )
            return {
                "customer_id": folder_name,
                "status": "OK",
                "a1_elapsed": elapsed,
                "a1_output": a1_output,
            }
        except Exception as e:
            elapsed = round(time.time() - start, 2)
            logger.error(f"  [A1 #{idx+1}/{total}] ❌ {folder_name} FAILED: {e}")
            return {
                "customer_id": folder_name,
                "status": "FAILED",
                "a1_elapsed": elapsed,
                "a1_output": None,
                "error": str(e),
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, folder in enumerate(customer_folders):
            future = executor.submit(_ingest_one, folder, idx)
            futures[future] = folder

            # Stagger
            if idx < total - 1 and stagger_delay > 0:
                time.sleep(stagger_delay)

        for future in as_completed(futures):
            result = future.result()
            results_map[result["customer_id"]] = result

    # Preserve original order
    return [results_map[f] for f in customer_folders if f in results_map]


# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Parallel A2→A3→A4 Processing (mirrors execute_batch_processing_parallel)
# ═══════════════════════════════════════════════════════════════════════════════


def run_phase3_parallel(
    phase1_results: list[dict],
    max_workers: int = 5,
    stagger_delay: float = 2.0,
) -> list[dict]:
    """Run A2→A3→A4 in parallel for all successfully-ingested customers.

    Mirrors the frontend /ws/batch Phase 3 behavior:
    - Auto-approves all A1 outputs (skip human review)
    - Uses ThreadPoolExecutor with max_workers
    - Staggers launches by stagger_delay seconds
    - Tracks per-customer tokens and timing

    Returns list of per-customer results.
    """
    from credicouncil.agents.a2_feature_engineer.agent import FeatureEngineerAgent
    from credicouncil.agents.a3_scoring.agent import ScoringAgent
    from credicouncil.agents.a4_report_generator.agent import ReportGeneratorAgent
    from credicouncil.services.llm_service import reset_token_counter, get_token_counts

    ok_results = [r for r in phase1_results if r["status"] == "OK"]
    total = len(ok_results)
    results_map: dict[str, dict] = {}

    def _process_one(r: dict, idx: int) -> dict:
        folder_name = r["customer_id"]
        a1_output = r["a1_output"]

        # Reset token counter for this customer
        reset_token_counter()
        start = time.time()

        try:
            # A2: Feature Engineering
            a2 = FeatureEngineerAgent()
            a2_output = a2.process(a1_output)

            # A3: ML Scoring
            a3 = ScoringAgent(model_path=str(BACKEND_ROOT / "models" / "lgbm_ref_v1.pkl"))
            a3_output = a3.score(a2_output)

            # A4: Report Generation
            a4 = ReportGeneratorAgent()
            a4_output = a4.generate(a3_output, a2_output, a1_output)

            elapsed = round(time.time() - start, 2)
            tokens = get_token_counts()

            # Extract results
            five_c_scores = a4_output.get("five_c_scores", {})
            report = a4_output.get("final_report", {})
            executive = report.get("executive_summary", {})
            if not five_c_scores:
                five_c_scores = executive.get("five_c_scores", {})

            logger.info(
                f"  [A2→A4 #{idx+1}/{total}] ✅ {folder_name} "
                f"Score={a3_output.get('credit_score')}, "
                f"Band={a3_output.get('risk_band')} in {elapsed:.1f}s"
            )

            return {
                "customer_id": folder_name,
                "status": "SUCCESS",
                "a234_elapsed": elapsed,
                "prompt_tokens": tokens["prompt_tokens"],
                "candidates_tokens": tokens["candidates_tokens"],
                "total_tokens": tokens["total_tokens"],
                "credit_score": a3_output.get("credit_score", 0),
                "pd_pct": a3_output.get("pd_pct", 0.0),
                "risk_band": a3_output.get("risk_band", "N/A"),
                "five_c_total": sum(five_c_scores.values()) if five_c_scores else 0,
                "recommendation": executive.get("recommendation", a3_output.get("routing", "N/A")),
            }

        except Exception as e:
            elapsed = round(time.time() - start, 2)
            tokens = get_token_counts()
            logger.error(f"  [A2→A4 #{idx+1}/{total}] ❌ {folder_name} FAILED: {e}")
            return {
                "customer_id": folder_name,
                "status": "FAILED",
                "a234_elapsed": elapsed,
                "prompt_tokens": tokens["prompt_tokens"],
                "candidates_tokens": tokens["candidates_tokens"],
                "total_tokens": tokens["total_tokens"],
                "error": str(e),
            }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, r in enumerate(ok_results):
            future = executor.submit(_process_one, r, idx)
            futures[future] = r["customer_id"]

            # Stagger
            if idx < total - 1 and stagger_delay > 0:
                time.sleep(stagger_delay)

        for future in as_completed(futures):
            result = future.result()
            results_map[result["customer_id"]] = result

    # Preserve original order + add failed A1 results
    ordered = []
    for r in phase1_results:
        fid = r["customer_id"]
        if fid in results_map:
            # Merge a1_elapsed into final result
            results_map[fid]["a1_elapsed"] = r["a1_elapsed"]
            results_map[fid]["total_elapsed"] = round(
                r["a1_elapsed"] + results_map[fid]["a234_elapsed"], 2
            )
            ordered.append(results_map[fid])
        else:
            ordered.append({
                "customer_id": fid,
                "status": "FAILED_A1",
                "a1_elapsed": r.get("a1_elapsed", 0),
                "error": r.get("error", "A1 ingestion failed"),
            })

    return ordered


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def run_batch_benchmark(
    limit: int | None = None,
    ids: list[str] | None = None,
    batch_size: int = 5,
    delay_between_batches: float = 3.0,
    a1_workers: int = 5,
    a1_stagger: float = 1.0,
    a234_workers: int = 5,
    a234_stagger: float = 2.0,
    output_file: str | None = None,
) -> dict:
    """Run the full batch benchmark.

    Splits customers into batches of `batch_size`, runs each batch with
    parallel A1 → auto-approve → parallel A2→A4, matching the production
    /ws/batch WebSocket flow.

    Args:
        limit: Max total customers to process.
        ids: Specific customer IDs.
        batch_size: Customers per batch (default 5).
        delay_between_batches: Seconds between batches (default 3).
        a1_workers: Parallel workers for A1 phase.
        a1_stagger: Stagger delay for A1 launches.
        a234_workers: Parallel workers for A2→A4 phase.
        a234_stagger: Stagger delay for A2→A4 launches.
        output_file: Path to save JSON results.
    """
    customers = discover_customers(limit=limit, ids=ids)
    total = len(customers)

    if total == 0:
        print("❌ No customers found in data/mock/")
        return {}

    batches = chunk_list(customers, batch_size)
    num_batches = len(batches)

    print("\n" + "=" * 80)
    print("  CREDICOUNCIL — E2E BATCH PIPELINE BENCHMARK")
    print("=" * 80)
    print(f"  Customers:     {total}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Num batches:   {num_batches}")
    print(f"  A1 workers:    {a1_workers} (stagger {a1_stagger}s)")
    print(f"  A2→A4 workers: {a234_workers} (stagger {a234_stagger}s)")
    print(f"  Delay between: {delay_between_batches}s")
    print(f"  Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    overall_start = time.time()
    all_customer_results = []
    batch_summaries = []

    for batch_idx, batch_customers in enumerate(batches, 1):
        batch_start = time.time()
        batch_ids_str = ", ".join(c.replace("customer_", "") for c in batch_customers)

        print(f"\n{'━' * 80}")
        print(f"  BATCH {batch_idx}/{num_batches} — [{batch_ids_str}] ({len(batch_customers)} customers)")
        print(f"{'━' * 80}")

        # ── Phase 1: Parallel A1 ──
        print(f"\n  📥 Phase 1: A1 Ingestion (parallel, {a1_workers} workers)...")
        p1_start = time.time()
        phase1_results = run_phase1_parallel(
            batch_customers,
            max_workers=a1_workers,
            stagger_delay=a1_stagger,
        )
        p1_elapsed = round(time.time() - p1_start, 2)
        a1_ok = sum(1 for r in phase1_results if r["status"] == "OK")
        print(f"  ✅ Phase 1 done: {a1_ok}/{len(batch_customers)} OK in {p1_elapsed:.1f}s")

        # ── Phase 2: Auto-approve (skip human review) ──
        print(f"  ⏩ Phase 2: Auto-approve (skipping human review)")

        # ── Phase 3: Parallel A2→A4 ──
        print(f"  ⚙️  Phase 3: A2→A3→A4 Processing (parallel, {a234_workers} workers)...")
        p3_start = time.time()
        customer_results = run_phase3_parallel(
            phase1_results,
            max_workers=a234_workers,
            stagger_delay=a234_stagger,
        )
        p3_elapsed = round(time.time() - p3_start, 2)

        batch_elapsed = round(time.time() - batch_start, 2)

        # Batch metrics
        success = [r for r in customer_results if r.get("status") == "SUCCESS"]
        failed = [r for r in customer_results if r.get("status") != "SUCCESS"]
        batch_tokens = sum(r.get("total_tokens", 0) for r in customer_results)
        batch_prompt = sum(r.get("prompt_tokens", 0) for r in customer_results)
        batch_cand = sum(r.get("candidates_tokens", 0) for r in customer_results)

        batch_summary = {
            "batch_index": batch_idx,
            "customer_ids": [c.replace("customer_", "") for c in batch_customers],
            "batch_size": len(batch_customers),
            "phase1_seconds": p1_elapsed,
            "phase3_seconds": p3_elapsed,
            "total_seconds": batch_elapsed,
            "success_count": len(success),
            "failed_count": len(failed),
            "prompt_tokens": batch_prompt,
            "candidates_tokens": batch_cand,
            "total_tokens": batch_tokens,
        }
        batch_summaries.append(batch_summary)
        all_customer_results.extend(customer_results)

        # Print batch summary
        print(f"\n  ── Batch {batch_idx} Summary ──")
        print(f"  Phase 1 (A1):     {p1_elapsed:.1f}s")
        print(f"  Phase 3 (A2→A4):  {p3_elapsed:.1f}s")
        print(f"  Total:            {batch_elapsed:.1f}s")
        print(f"  Success:          {len(success)}/{len(batch_customers)}")
        print(f"  Tokens:           {batch_tokens:,}")

        for r in customer_results:
            cid = r["customer_id"].replace("customer_", "")
            status = "✅" if r.get("status") == "SUCCESS" else "❌"
            a1_t = f"{r.get('a1_elapsed', 0):.1f}s"
            a234_t = f"{r.get('a234_elapsed', 0):.1f}s"
            total_t = f"{r.get('total_elapsed', 0):.1f}s"
            tk = f"{r.get('total_tokens', 0):,}"
            score = str(r.get("credit_score", "—"))
            band = r.get("risk_band", "—")
            print(
                f"    {status} {cid:<10} A1={a1_t:<7} A2→A4={a234_t:<7} "
                f"Total={total_t:<7} Tokens={tk:<10} Score={score:<5} Band={band}"
            )

        # Delay between batches (skip after last batch)
        if batch_idx < num_batches and delay_between_batches > 0:
            print(f"\n  💤 Waiting {delay_between_batches}s before next batch...")
            time.sleep(delay_between_batches)

    overall_elapsed = round(time.time() - overall_start, 2)

    # ═══════════════════════════════════════════════════════════════════════
    # AGGREGATE STATISTICS
    # ═══════════════════════════════════════════════════════════════════════

    all_success = [r for r in all_customer_results if r.get("status") == "SUCCESS"]
    all_failed = [r for r in all_customer_results if r.get("status") != "SUCCESS"]

    total_prompt = sum(r.get("prompt_tokens", 0) for r in all_customer_results)
    total_cand = sum(r.get("candidates_tokens", 0) for r in all_customer_results)
    total_tok = sum(r.get("total_tokens", 0) for r in all_customer_results)
    total_a1_time = sum(r.get("a1_elapsed", 0) for r in all_customer_results)
    total_a234_time = sum(r.get("a234_elapsed", 0) for r in all_customer_results)
    total_cust_time = sum(r.get("total_elapsed", 0) for r in all_customer_results)

    n = max(len(all_customer_results), 1)
    avg_a1 = round(total_a1_time / n, 2)
    avg_a234 = round(total_a234_time / n, 2)
    avg_total = round(total_cust_time / n, 2)
    avg_prompt = int(round(total_prompt / n))
    avg_cand = int(round(total_cand / n))
    avg_tok = int(round(total_tok / n))

    # Batch-level averages
    nb = max(num_batches, 1)
    avg_batch_time = round(sum(b["total_seconds"] for b in batch_summaries) / nb, 2)
    avg_batch_p1 = round(sum(b["phase1_seconds"] for b in batch_summaries) / nb, 2)
    avg_batch_p3 = round(sum(b["phase3_seconds"] for b in batch_summaries) / nb, 2)

    summary = {
        "benchmark_timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "total_customers": total,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "a1_workers": a1_workers,
            "a1_stagger": a1_stagger,
            "a234_workers": a234_workers,
            "a234_stagger": a234_stagger,
            "delay_between_batches": delay_between_batches,
        },
        "overall": {
            "wall_clock_seconds": overall_elapsed,
            "success_count": len(all_success),
            "failed_count": len(all_failed),
        },
        "aggregated_customer": {
            "avg_a1_seconds": avg_a1,
            "avg_a234_seconds": avg_a234,
            "avg_total_seconds": avg_total,
            "sum_prompt_tokens": total_prompt,
            "sum_candidates_tokens": total_cand,
            "sum_total_tokens": total_tok,
            "avg_prompt_tokens": avg_prompt,
            "avg_candidates_tokens": avg_cand,
            "avg_total_tokens": avg_tok,
        },
        "aggregated_batch": {
            "avg_batch_seconds": avg_batch_time,
            "avg_phase1_seconds": avg_batch_p1,
            "avg_phase3_seconds": avg_batch_p3,
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
    print()
    print(f"  ── Batch Averages ──")
    print(f"  Avg batch time:       {avg_batch_time}s")
    print(f"  Avg Phase 1 (A1):     {avg_batch_p1}s")
    print(f"  Avg Phase 3 (A2→A4):  {avg_batch_p3}s")
    print()
    print(f"  ── Customer Averages ──")
    print(f"  Avg A1 time:          {avg_a1}s")
    print(f"  Avg A2→A4 time:       {avg_a234}s")
    print(f"  Avg total time:       {avg_total}s")
    print(f"  Avg prompt tokens:    {avg_prompt:,}")
    print(f"  Avg output tokens:    {avg_cand:,}")
    print(f"  Avg total tokens:     {avg_tok:,}")
    print()
    print(f"  ── Token Usage (Total) ──")
    print(f"  Total prompt:         {total_prompt:>12,}")
    print(f"  Total candidates:     {total_cand:>12,}")
    print(f"  Total tokens:         {total_tok:>12,}")

    # Per-customer table
    print()
    hdr = (
        f"  {'Customer':<12} {'Status':<7} {'A1(s)':<7} {'A2-4(s)':<8} "
        f"{'Total(s)':<9} {'Input Tk':<10} {'Out Tk':<10} {'All Tk':<10} "
        f"{'Score':<6} {'Band':<5}"
    )
    print(hdr)
    print("  " + "─" * len(hdr.strip()))

    for r in all_customer_results:
        cid = r["customer_id"].replace("customer_", "")
        st = "✅" if r.get("status") == "SUCCESS" else "❌"
        a1 = f"{r.get('a1_elapsed', 0):.1f}"
        a234 = f"{r.get('a234_elapsed', 0):.1f}"
        tot = f"{r.get('total_elapsed', 0):.1f}"
        pt = f"{r.get('prompt_tokens', 0):,}"
        ct = f"{r.get('candidates_tokens', 0):,}"
        tt = f"{r.get('total_tokens', 0):,}"
        sc = str(r.get("credit_score", "—"))
        bd = r.get("risk_band", "—")
        print(
            f"  {cid:<12} {st:<7} {a1:<7} {a234:<8} {tot:<9} "
            f"{pt:<10} {ct:<10} {tt:<10} {sc:<6} {bd:<5}"
        )

    # Average row
    print("  " + "─" * len(hdr.strip()))
    print(
        f"  {'AVERAGE':<12} {'—':<7} {avg_a1:<7} {avg_a234:<8.1f} {avg_total:<9.1f} "
        f"{avg_prompt:<10,} {avg_cand:<10,} {avg_tok:<10,} {'—':<6} {'—':<5}"
    )

    print(f"\n  Results saved: {output_file}")
    print("=" * 80)

    return summary


def main():
    # Force UTF-8 stdout (Windows compatibility)
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="CrediCouncil E2E Batch Pipeline Benchmark"
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
        "--delay", type=float, default=3.0,
        help="Seconds to wait between batches (default: 3.0)"
    )
    parser.add_argument(
        "--a1-workers", type=int, default=5,
        help="Parallel workers for A1 ingestion (default: 5)"
    )
    parser.add_argument(
        "--a1-stagger", type=float, default=1.0,
        help="Stagger delay for A1 launches (default: 1.0)"
    )
    parser.add_argument(
        "--a234-workers", type=int, default=5,
        help="Parallel workers for A2→A4 processing (default: 5)"
    )
    parser.add_argument(
        "--a234-stagger", type=float, default=2.0,
        help="Stagger delay for A2→A4 launches (default: 2.0)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON file path (default: evaluation/e2e_batch/batch_benchmark_results.json)"
    )
    args = parser.parse_args()

    run_batch_benchmark(
        limit=args.limit,
        ids=args.ids,
        batch_size=args.batch_size,
        delay_between_batches=args.delay,
        a1_workers=args.a1_workers,
        a1_stagger=args.a1_stagger,
        a234_workers=args.a234_workers,
        a234_stagger=args.a234_stagger,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
