"""
MASCA - Multi-Agent System for Credit Assessment
Entry point for running the credit assessment pipeline.

Usage:
    python main.py                    # Evaluate sample 0
    python main.py --sample 5         # Evaluate sample at index 5
    python main.py --sample 0 --verbose  # With debug logging
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import get_llm_config, get_app_config
from data.loader import load_sample, format_sample_for_agent
from pipeline.orchestrator import MASCAOrchestrator


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def print_result_summary(results: dict) -> None:
    """Print a formatted summary of the pipeline results."""
    print("\n" + "=" * 70)
    print("  MASCA CREDIT ASSESSMENT RESULTS")
    print("=" * 70)

    # Final Decision
    decision = results.get("final_decision", {})
    
    score = decision.get("credit_score", "N/A")
    rating = decision.get("credit_rating", "N/A")
    risk_desc = decision.get("risk_level_description", "N/A")
    
    print(f"\n🏆 Final Credit Score: {score}/100")
    print(f"⭐ Credit Rating: {rating} ({risk_desc})")
    print(f"📋 Decision: {decision.get('decision', 'N/A')}")
    print(f"📝 Justification: {decision.get('justification', 'N/A')}")

    # Key Factors
    key_factors = decision.get("key_factors", [])
    if key_factors:
        print("\n🔑 Key Factors:")
        for factor in key_factors:
            print(f"   • {factor}")

    # Layer 2 scores
    layer2 = results.get("layer2", {})
    print("\n📈 Assessment Scores:")
    risk = layer2.get("Risk Modeler", {})
    print(f"   • Risk Score: {risk.get('risk_score', 'N/A')}")
    income = layer2.get("Income & Stability Analyst", {})
    print(f"   • Income Stability: {income.get('income_stability_score', 'N/A')}")
    debt = layer2.get("Debt Analyst", {})
    print(f"   • Loan Feasibility: {debt.get('loan_feasibility_score', 'N/A')}")
    reward = layer2.get("Reward Modeler", {})
    print(f"   • Reward Score: {reward.get('overall_reward_score', 'N/A')}")
    ml = layer2.get("ML Modeler", {})
    print(f"   • ML Credit Score: {ml.get('ml_credit_score', 'N/A')}")

    # Risk-Reward
    rr = results.get("risk_reward_optimization", {})
    print(f"   • Risk-Reward Ratio: {rr.get('risk_reward_ratio', 'N/A')}")

    # Timing
    print(f"\n⏱️  Total Pipeline Time: {results.get('pipeline_time_seconds', 'N/A')}s")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="MASCA - Multi-Agent Credit Assessment System"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample index from German Credit Dataset (0-999)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save full results to JSON file",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load config
    try:
        llm_config = get_llm_config()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        print("   Set your GEMINI_API_KEY or OPENROUTER_API_KEY in .env based on the selected LLM_PROVIDER.")
        sys.exit(1)

    app_config = get_app_config()

    provider = llm_config.provider
    if provider == "google":
        model_display = llm_config.gemini_model_name
    elif provider == "openrouter":
        model_display = llm_config.openrouter_model_name
    else:  # cliproxy
        model_display = f"{llm_config.cliproxy_model_name} @ {llm_config.cliproxy_base_url}"
    print(f"\n🔧 Model: [{provider.upper()}] {model_display}")
    print(f"📂 Dataset: {app_config.dataset_path}")
    print(f"🎯 Sample Index: {args.sample}")

    # Load sample
    try:
        record = load_sample(args.sample, app_config.dataset_path)
        formatted_input = format_sample_for_agent(record)
        target = record.get("TARGET", "Unknown")
        label = "Default" if target == 1 else "No Default" if target == 0 else "Unknown"
        print(f"🏷️  Ground Truth: {label} ({target})")
    except Exception as e:
        print(f"\n❌ Data Error: {e}")
        sys.exit(1)

    print(f"\n{'─'*70}")
    print(formatted_input)
    print(f"{'─'*70}")
    print("\n🚀 Starting MASCA pipeline...\n")

    # Run pipeline
    orchestrator = MASCAOrchestrator(
        config=llm_config, max_workers=app_config.max_concurrent_agents
    )
    results = orchestrator.run(formatted_input)

    # Print summary
    print_result_summary(results)

    # Save to file
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Full results saved to: {output_path}")


if __name__ == "__main__":
    main()
