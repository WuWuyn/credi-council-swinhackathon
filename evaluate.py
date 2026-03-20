import argparse
import sys
import time
import logging
from pathlib import Path
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import get_llm_config, get_app_config
from data.loader import load_sample, format_sample_for_agent
from pipeline.orchestrator import MASCAOrchestrator

def setup_logging(verbose: bool = False) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

def evaluate_samples(n_samples: int, start_idx: int = 0, verbose: bool = False, output_file: str = "evaluation_results.csv"):
    setup_logging(verbose)
    
    try:
        llm_config = get_llm_config()
    except ValueError as e:
        print(f"\n❌ Configuration Error: {e}")
        sys.exit(1)

    app_config = get_app_config()
    
    print(f"🚀 Starting Batch Evaluation on {n_samples} samples (Indices {start_idx} to {start_idx + n_samples - 1})")
    provider = llm_config.provider
    if provider == "google":
        model_display = llm_config.gemini_model_name
    elif provider == "openrouter":
        model_display = llm_config.openrouter_model_name
    else:  # cliproxy
        model_display = f"{llm_config.cliproxy_model_name} @ {llm_config.cliproxy_base_url}"
    print(f"🔧 Model: [{provider.upper()}] {model_display}")
    print("⚠️ Note: Large LLMs on free tiers may take a long time per sample due to rate limits.")
    print("-" * 50)

    orchestrator = MASCAOrchestrator(
        config=llm_config, max_workers=app_config.max_concurrent_agents
    )

    results = []
    correct_predictions = 0
    total_valid = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(range(start_idx, start_idx + n_samples), desc="Evaluating")
    except ImportError:
        print("Progress logging (tqdm not installed)...")
        iterator = range(start_idx, start_idx + n_samples)

    for i in iterator:
        sample_result = {
            "index": i,
            "SK_ID_CURR": None,
            "ground_truth_target": None,
            "predicted_decision": None,
            "predicted_target": None,
            "is_correct": None,
            "credit_score": None,
            "ml_credit_score": None,
            "credit_rating": None,
            "execution_time_s": 0,
            "error": None
        }

        try:
            record = load_sample(i, app_config.dataset_path)
            sample_result["SK_ID_CURR"] = record.get("SK_ID_CURR")
            
            target = record.get("TARGET", -1)
            sample_result["ground_truth_target"] = target
            
            formatted_input = format_sample_for_agent(record)
            
            # Run pipeline
            start_time = time.time()
            pipeline_out = orchestrator.run(formatted_input)
            sample_result["execution_time_s"] = round(time.time() - start_time, 2)
                
            # Extract final decision
            decision_dict = pipeline_out.get("final_decision", {})
            decision = decision_dict.get("decision", "Unknown")
            sample_result["predicted_decision"] = decision
            sample_result["credit_score"] = decision_dict.get("credit_score", None)
            
            # Extract ML Credit Score
            layer2_dict = pipeline_out.get("layer2", {})
            ml_dict = layer2_dict.get("ML Modeler", {})
            sample_result["ml_credit_score"] = ml_dict.get("ml_credit_score", None)
            
            sample_result["credit_rating"] = decision_dict.get("credit_rating", "Unknown")
            
            # MAPPING Logic:
            # If the model says "Approve" -> no default / good loan (0)
            # If the model says "Reject" -> default / bad loan (1)
            pred_decision_upper = decision.upper()
            if "APPROVE" in pred_decision_upper:
                predicted_target = 0
            elif "REJECT" in pred_decision_upper:
                predicted_target = 1
            else:
                predicted_target = -1 
                
            sample_result["predicted_target"] = predicted_target

            # Evaluate correctness
            if target in [0, 1] and predicted_target in [0, 1]:
                total_valid += 1
                is_correct = (target == predicted_target)
                sample_result["is_correct"] = is_correct
                if is_correct:
                    correct_predictions += 1
                    
        except Exception as e:
            sample_result["error"] = str(e)
            
        results.append(sample_result)
        
        # Save intermediate results
        try:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
        except Exception:
            pass

    print("\n" + "=" * 50)
    print("📊 EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total Samples Run: {n_samples}")
    if total_valid > 0:
        accuracy = (correct_predictions / total_valid) * 100
        print(f"Valid Comparable Decisions (Approve/Reject vs 0/1): {total_valid}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("No valid comparable decisions made (all 'Review' or errors).")
        
    print(f"💾 Full results saved to: {output_file}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate MASCA Pipeline over N samples.")
    parser.add_argument("--n", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--start", type=int, default=0, help="Starting index in dataset")
    parser.add_argument("--verbose", action="store_true", help="Print debug/info logs")
    parser.add_argument("--output", type=str, default="evaluation_results.csv", help="Output CSV path")
    
    args = parser.parse_args()
    evaluate_samples(args.n, args.start, args.verbose, args.output)
