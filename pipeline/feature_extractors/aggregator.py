"""
aggregator.py

Master feature aggregation module.
Calls all individual dataset extractors and merges them into
a single flat dictionary suitable for LightGBM inference.

This function is exposed as a Gemini Tool to the FeatureEngineerAgent (Layer 1).
"""

import os
import sys
import traceback

# Data directory (relative to project root, or set via environment variable)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'home-credit-default-risk'
)

def generate_comprehensive_features(sk_id_curr: int, data_dir: str = None) -> dict:
    """
    Generate a comprehensive feature vector for a single applicant (SK_ID_CURR).
    Orchestrates all modular feature extractors and merges their outputs.
    
    Args:
        sk_id_curr: The applicant's unique ID from Home Credit dataset.
        data_dir: Path to the directory containing all Home Credit CSV files.
                  Defaults to the bundled 'home-credit-default-risk' folder.
    
    Returns:
        A flat dictionary of feature_name -> float value, ready for ML inference.
    """
    if data_dir is None:
        data_dir = os.environ.get('HOME_CREDIT_DATA_DIR', DEFAULT_DATA_DIR)
    data_dir = os.path.abspath(data_dir)
    
    features = {}
    errors = []
    empty_extractors = []  # returned {} (no data found for this applicant)
    
    extractors = [
        ('application',   _safe_import_and_run, 'application',   'extract_application_features'),
        ('bureau',        _safe_import_and_run, 'bureau',         'extract_bureau_features'),
        ('previous',      _safe_import_and_run, 'previous',       'extract_previous_features'),
        ('installments',  _safe_import_and_run, 'installments',   'extract_installments_features'),
        ('pos_cash',      _safe_import_and_run, 'pos_cash',       'extract_pos_cash_features'),
        ('credit_card',   _safe_import_and_run, 'credit_card',    'extract_credit_card_features'),
    ]
    
    for name, runner, module_name, func_name in extractors:
        try:
            result = runner(module_name, func_name, sk_id_curr, data_dir)
            if result:
                features.update(result)
            else:
                empty_extractors.append(name)
                print(f"[aggregator] Info: extractor '{name}' returned no data for SK_ID_CURR={sk_id_curr}")
        except Exception as e:
            errors.append(f"{name}: {str(e)}")
            print(f"[aggregator] Warning: extractor '{name}' failed: {e}")
    
    # Build human-readable context for agents to understand missing data
    missing_notes = []
    if 'bureau' in empty_extractors:
        missing_notes.append(
            "IMPORTANT: No credit bureau history found for this applicant. "
            "This means the applicant has NO prior credit transactions with external institutions "
            "— they are a first-time or thin-file borrower. This is NOT a data error."
        )
    if 'previous' in empty_extractors:
        missing_notes.append(
            "NOTE: No previous loan applications found at Home Credit for this applicant "
            "— this is their first application at this institution."
        )
    other_empty = [e for e in empty_extractors if e not in ('bureau', 'previous')]
    if other_empty:
        missing_notes.append(
            f"NOTE: No transaction history available for: {', '.join(other_empty)} "
            "— applicant has not used these credit products before."
        )
    
    if missing_notes:
        features['_missing_data_notes'] = ' | '.join(missing_notes)
    if errors:
        features['_extraction_errors'] = '; '.join(errors)
    
    features['SK_ID_CURR'] = sk_id_curr
    return features


def _safe_import_and_run(module_name, func_name, sk_id_curr, data_dir):
    """Dynamically imports 'pipeline.feature_extractors.<module_name>' and runs func_name."""
    # Add project root to sys.path if needed
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(pkg_dir, '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    import importlib
    mod = importlib.import_module(f'pipeline.feature_extractors.{module_name}')
    func = getattr(mod, func_name)
    return func(sk_id_curr, data_dir)


if __name__ == '__main__':
    import json

    sk_id = int(sys.argv[1]) if len(sys.argv) > 1 else 100001
    data_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"[aggregator] Generating features for SK_ID_CURR={sk_id} ...")
    feats = generate_comprehensive_features(sk_id, data_dir)
    print(f"[aggregator] Done. Total features: {len(feats)}")
    # Print only top 20 for brevity
    sample = {k: v for i, (k, v) in enumerate(feats.items()) if i < 20}
    print(json.dumps(sample, indent=2))
