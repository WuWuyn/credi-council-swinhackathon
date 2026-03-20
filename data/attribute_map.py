"""
Attribute mapping for the German Credit Dataset.
Source: german.doc from UCI Machine Learning Repository.

20 attributes (13 categorical, 7 numerical) + 1 label.
"""

# Ordered attribute names matching column positions in german.data
ATTRIBUTE_NAMES: list[str] = [
    "checking_account_status",   # X1  - qualitative
    "duration_months",           # X2  - numerical
    "credit_history",            # X3  - qualitative
    "purpose",                   # X4  - qualitative
    "credit_amount",             # X5  - numerical
    "savings_account",           # X6  - qualitative
    "employment_since",          # X7  - qualitative
    "installment_rate",          # X8  - numerical (% of disposable income)
    "personal_status_sex",       # X9  - qualitative
    "other_debtors",             # X10 - qualitative
    "residence_since",           # X11 - numerical
    "property",                  # X12 - qualitative
    "age_years",                 # X13 - numerical
    "other_installment_plans",   # X14 - qualitative
    "housing",                   # X15 - qualitative
    "existing_credits",          # X16 - numerical
    "job",                       # X17 - qualitative
    "num_dependents",            # X18 - numerical
    "telephone",                 # X19 - qualitative
    "foreign_worker",            # X20 - qualitative
]

# Categorical attribute code -> human-readable description
CATEGORICAL_CODES: dict[str, dict[str, str]] = {
    "checking_account_status": {
        "A11": "< 0 DM",
        "A12": "0 <= ... < 200 DM",
        "A13": ">= 200 DM / salary assignments for at least 1 year",
        "A14": "no checking account",
    },
    "credit_history": {
        "A30": "no credits taken / all credits paid back duly",
        "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly till now",
        "A33": "delay in paying off in the past",
        "A34": "critical account / other credits existing (not at this bank)",
    },
    "purpose": {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings_account": {
        "A61": "< 100 DM",
        "A62": "100 <= ... < 500 DM",
        "A63": "500 <= ... < 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "unknown / no savings account",
    },
    "employment_since": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1 <= ... < 4 years",
        "A74": "4 <= ... < 7 years",
        "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "male: divorced/separated",
        "A92": "female: divorced/separated/married",
        "A93": "male: single",
        "A94": "male: married/widowed",
        "A95": "female: single",
    },
    "other_debtors": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property": {
        "A121": "real estate",
        "A122": "building society savings agreement / life insurance",
        "A123": "car or other (not in savings account)",
        "A124": "unknown / no property",
    },
    "other_installment_plans": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "for free",
    },
    "job": {
        "A171": "unemployed / unskilled - non-resident",
        "A172": "unskilled - resident",
        "A173": "skilled employee / official",
        "A174": "management / self-employed / highly qualified employee / officer",
    },
    "telephone": {
        "A191": "none",
        "A192": "yes, registered under the customer's name",
    },
    "foreign_worker": {
        "A201": "yes",
        "A202": "no",
    },
}

# Numerical attribute indices (0-based) for convenience
NUMERICAL_ATTRIBUTES: set[str] = {
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_since",
    "age_years",
    "existing_credits",
    "num_dependents",
}

# Label mapping
LABEL_MAP: dict[int, str] = {
    1: "good",
    2: "bad",
}
