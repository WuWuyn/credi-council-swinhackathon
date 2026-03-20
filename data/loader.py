"""
Data loader for the Home Credit Default Risk Dataset.
Parses application_train.csv and formats samples for LLM agents.
"""

import pandas as pd
from pathlib import Path
from typing import Optional

def load_dataset(filepath: Optional[str] = None) -> pd.DataFrame:
    """
    Load the Home Credit application dataset.

    Args:
        filepath: Path to application_train.csv.

    Returns:
        Pandas DataFrame containing the application data.
    """
    if filepath is None:
        filepath = str(
            Path(__file__).resolve().parent.parent
            / "home-credit-default-risk"
            / "application_train.csv"
        )

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    return pd.read_csv(path)


def load_sample(index: int = 0, filepath: Optional[str] = None) -> dict:
    """
    Load a single sample from the dataset.

    Args:
        index: 0-based index of the sample.
        filepath: Path to application_train.csv.

    Returns:
        A single record dict.
    """
    df = load_dataset(filepath)
    if index < 0 or index >= len(df):
        raise IndexError(f"Index {index} out of range (0-{len(df) - 1})")
    
    # Fill NAs with appropriate values to avoid JSON serialization issues
    sample = df.iloc[index].fillna("Unknown").to_dict()
    return sample


def format_sample_for_agent(sample: dict) -> str:
    """
    Format a raw Home Credit sample dict into a readable text block.
    """
    # Extract key identity and target info first
    sk_id = sample.get("SK_ID_CURR", "Unknown")
    target = sample.get("TARGET", "Unknown")
    target_str = "Default" if target == 1 else "No Default" if target == 0 else "Unknown"
    
    lines = [f"--- HOME CREDIT APPLICATION RECORD ---"]
    lines.append(f"Application ID (SK_ID_CURR): {sk_id}")
    lines.append(f"Historical Default Status (TARGET): {target_str} ({target})")
    lines.append("--- APPLICANT PROFILE ---")
    
    # Categorize fields for better readability
    categories = {
        "Demographics": ["CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CNT_CHILDREN", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "REGION_POPULATION_RELATIVE"],
        "Income & Employment": ["AMT_INCOME_TOTAL", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "DAYS_BIRTH", "DAYS_EMPLOYED", "OCCUPATION_TYPE", "ORGANIZATION_TYPE"],
        "Credit Request": ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE", "NAME_CONTRACT_TYPE"],
        "Contact Info": ["FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"]
    }
    
    used_keys = set(["SK_ID_CURR", "TARGET"])
    
    for category, keys in categories.items():
        lines.append(f"\n[{category}]")
        for key in keys:
            if key in sample:
                val = sample[key]
                # Convert negative days to positive years approx for readability
                if key in ["DAYS_BIRTH", "DAYS_EMPLOYED"] and isinstance(val, (int, float)) and val < 0:
                    years = abs(val) / 365.25
                    lines.append(f"  {key}: {val} (approx {years:.1f} years)")
                else:
                    lines.append(f"  {key}: {val}")
                used_keys.add(key)
                
    lines.append(f"\n[Additional Info]")
    # Add a few more important ones, skip the huge array of FLAG_DOCUMENT and EXT_SOURCE
    for key in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        if key in sample:
            lines.append(f"  {key}: {sample[key]}")
            
    lines.append("\nNote: Hundreds of other raw attributes exist but are omitted for brevity. A specialized feature engineering tool should be used to extract the full numerical vector.")
    return "\n".join(lines)
