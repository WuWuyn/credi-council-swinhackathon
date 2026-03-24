"""
Training Feature Engineering — Mirror of lgb1.ipynb (Home Credit Kaggle Reference).

Builds feature matrix from all 7 Home Credit tables.
Feature engineering logic is 100% faithful to the reference notebook.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


# ─── Memory optimization ──────────────────────────────────────────────────────

def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """float64→float32, int64→int32 to save RAM."""
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols   = [c for c in df if df[c].dtype == "int64"]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols]   = df[int_cols].astype(np.int32)
    return df


def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace special JSON characters in column names that LightGBM rejects.

    LightGBM >= 4.x rejects feature names containing: [ ] { } " : , space
    Also replaces parentheses and other non-alphanumeric chars with underscores.
    """
    import re
    df.columns = [
        re.sub(r"[^A-Za-z0-9_]", "_", c).strip("_")
        for c in df.columns
    ]
    # Deduplicate column names if sanitization created duplicates
    seen: dict[str, int] = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df



# ─── Mean encoding with KFold regularization ─────────────────────────────────

def mean_encode(train: pd.DataFrame, val: pd.DataFrame,
                features_to_encode: list[str], target: str,
                drop: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """KFold-regularized mean encoding to avoid leakage."""
    train_encode = train.copy(deep=True)
    val_encode   = val.copy(deep=True)
    for feature in features_to_encode:
        train_global_mean = train[target].mean()
        train_encode_map  = pd.DataFrame(index=train[feature].unique())
        train_encode[feature + "_mean_encode"] = np.nan
        kf = KFold(n_splits=5, shuffle=False)
        for rest, this in kf.split(train):
            train_rest_mean = train[target].iloc[rest].mean()
            encode_map = train.iloc[rest].groupby(feature)[target].mean()
            train_encode[feature + "_mean_encode"].iloc[this] = (
                train[feature].iloc[this].map(encode_map).values
            )
            train_encode_map = pd.concat(
                (train_encode_map, encode_map), axis=1, sort=False
            )
            train_encode_map.fillna(train_rest_mean, inplace=True)
            train_encode[feature + "_mean_encode"].fillna(train_rest_mean, inplace=True)
        train_encode_map["avg"] = train_encode_map.mean(axis=1)
        val_encode[feature + "_mean_encode"] = val[feature].map(train_encode_map["avg"])
        val_encode[feature + "_mean_encode"].fillna(train_global_mean, inplace=True)
    if drop:
        train_encode.drop(features_to_encode, axis=1, inplace=True)
        val_encode.drop(features_to_encode, axis=1, inplace=True)
    return train_encode, val_encode


# ─── Application table ────────────────────────────────────────────────────────

# Columns highly correlated / not informative → drop
REJECTED_APP_FEATURES = [
    "AMT_GOODS_PRICE",
    "APARTMENTS_AVG", "APARTMENTS_MEDI",
    "BASEMENTAREA_AVG", "BASEMENTAREA_MODE", "COMMONAREA_AVG", "COMMONAREA_MODE",
    "ELEVATORS_AVG", "ELEVATORS_MEDI", "ENTRANCES_AVG", "ENTRANCES_MEDI",
    "FLOORSMAX_AVG", "FLOORSMAX_MEDI", "FLOORSMIN_AVG", "FLOORSMIN_MEDI",
    "LANDAREA_AVG", "LANDAREA_MODE",
    "LIVINGAPARTMENTS_AVG", "LIVINGAPARTMENTS_MEDI",
    "LIVINGAREA_AVG", "LIVINGAREA_MODE",
    "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAPARTMENTS_MEDI",
    "NONLIVINGAREA_AVG", "NONLIVINGAREA_MODE", "OBS_30_CNT_SOCIAL_CIRCLE",
    "REGION_RATING_CLIENT", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BEGINEXPLUATATION_MEDI",
    "YEARS_BUILD_AVG", "YEARS_BUILD_MEDI",
    "ELEVATORS_MODE", "ENTRANCES_MODE", "NONLIVINGAPARTMENTS_MODE",
    "FLAG_MOBIL", "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_2",
    "WEEKDAY_APPR_PROCESS_START", "HOUR_APPR_PROCESS_START",
    # mean encoding raw values (keep only *_mean_encode)
    "gender_mean_income", "own_car_mean_income", "own_realty_mean_income",
    "cnt_children_mean_income", "family_status_mean_income",
    "gender_family_status_mean_income", "CNT_CHILDREN_CLIPPED",
]


def engineer_application_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Engineer features from application_train/test table.

    Returns:
        (df_engineered, meanenc_feats, cat_feats)
    """
    meanenc_feats: list[str] = []
    cat_feats: list[str] = []

    # ── Clean ──
    df["CODE_GENDER"].replace("XNA", np.nan, inplace=True)
    df["NAME_FAMILY_STATUS"].replace("Unknown", np.nan, inplace=True)
    df["ORGANIZATION_TYPE"].replace("XNA", np.nan, inplace=True)
    df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    # ── New ratio features ──
    docs = [f for f in df.columns if "FLAG_DOC" in f]
    live = [f for f in df.columns if ("FLAG_" in f) and ("FLAG_DOC" not in f) and ("_FLAG_" not in f)]
    live_num = [f for f in live if df[f].dtype != object]
    df["NEW_DOC_IND_KURT"]        = df[docs].kurtosis(axis=1)
    df["NEW_LIVE_IND_SUM"]        = df[live_num].sum(axis=1)
    df["NEW_INC_PER_CHLD"]        = df["AMT_INCOME_TOTAL"] / (1 + df["CNT_CHILDREN"])
    inc_by_org = df[["AMT_INCOME_TOTAL", "ORGANIZATION_TYPE"]].groupby("ORGANIZATION_TYPE").median()["AMT_INCOME_TOTAL"]
    df["NEW_INC_BY_ORG"]          = df["ORGANIZATION_TYPE"].map(inc_by_org)
    df["NEW_EMPLOY_TO_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["NEW_SOURCES_PROD"]        = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["NEW_EXT_SOURCES_MEAN"]    = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["NEW_SCORES_STD"]          = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    df["NEW_SCORES_STD"]          = df["NEW_SCORES_STD"].fillna(df["NEW_SCORES_STD"].mean())
    df["NEW_CAR_TO_BIRTH_RATIO"]  = df["OWN_CAR_AGE"] / df["DAYS_BIRTH"]
    df["NEW_CAR_TO_EMPLOY_RATIO"] = df["OWN_CAR_AGE"] / df["DAYS_EMPLOYED"]
    df["NEW_PHONE_TO_BIRTH_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_BIRTH"]
    df["NEW_PHONE_TO_EMPLOYED_RATIO"] = df["DAYS_LAST_PHONE_CHANGE"] / df["DAYS_EMPLOYED"]
    df["NEW_CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["AMT_PAY_YEAR"]            = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["AGE_PAYOFF"]              = -df["DAYS_BIRTH"] / 365.25 + df["AMT_PAY_YEAR"]
    df["AMT_ANNUITY_INCOME_RATE"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["AMT_DIFF_CREDIT_GOODS"]   = df["AMT_CREDIT"] - df["AMT_GOODS_PRICE"]
    df["AMT_CREDIT_GOODS_PERC"]   = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["DOCUMENT_CNT"]            = df.loc[:, df.columns.str.startswith("FLAG_DOCUMENT")].sum(axis=1)
    df["AGE_EMPLOYED"]            = df["DAYS_EMPLOYED"] - df["DAYS_BIRTH"]
    df["AMT_INCOME_OVER_CHILD"]   = df["AMT_INCOME_TOTAL"] / df["CNT_CHILDREN"].replace(0, np.nan)
    df["CNT_ADULT"]               = df["CNT_FAM_MEMBERS"] - df["CNT_CHILDREN"]
    df["ADULT_RATIO"]             = df["CNT_ADULT"] / df["CNT_FAM_MEMBERS"]
    df["AMT_REQ_CREDIT_BUREAU_MON_CHANGE"]  = df.get("AMT_REQ_CREDIT_BUREAU_QRT", 0) / 2 - df.get("AMT_REQ_CREDIT_BUREAU_MON", 0)
    df["AMT_REQ_CREDIT_BUREAU_QRT_CHANGE"]  = df.get("AMT_REQ_CREDIT_BUREAU_YEAR", 0) / 3 - df.get("AMT_REQ_CREDIT_BUREAU_QRT", 0)

    # ── Region / income mean features ──
    df["CNT_CHILDREN_CLIPPED"] = df["CNT_CHILDREN"].clip(0, 10)
    df["REGION"], _ = pd.factorize(df["REGION_POPULATION_RELATIVE"])
    meanenc_feats.append("REGION")

    df["GENDER_FAMILY_STATUS"] = df["CODE_GENDER"].astype(str) + df["NAME_FAMILY_STATUS"].astype(str)
    for grp_col, col_name in [
        ("CODE_GENDER",          "gender_mean_income"),
        ("FLAG_OWN_CAR",         "own_car_mean_income"),
        ("FLAG_OWN_REALTY",      "own_realty_mean_income"),
        ("CNT_CHILDREN_CLIPPED", "cnt_children_mean_income"),
        ("REGION",               "region_mean_income"),
        ("NAME_FAMILY_STATUS",   "family_status_mean_income"),
        ("GENDER_FAMILY_STATUS", "gender_family_status_mean_income"),
    ]:
        grp_map = df.groupby(grp_col)["AMT_INCOME_TOTAL"].median()
        df[col_name] = df[grp_col].map(grp_map)
        df[col_name + "_rel"] = (df["AMT_INCOME_TOTAL"] - df[col_name]) / df[col_name]

    # ── Drop rejected features ──
    for f in REJECTED_APP_FEATURES:
        if f in df.columns:
            del df[f]

    # ── Label encode categoricals ──
    categorical_feats = [f for f in df.columns if df[f].dtype == "object"]
    for f in categorical_feats:
        nunique = df[f].nunique(dropna=False)
        if nunique < 6:
            cat_feats.append(f)
        else:
            meanenc_feats.append(f)
        df[f], _ = pd.factorize(df[f])

    df = downcast_dtypes(df)
    return df, meanenc_feats, cat_feats


# ─── Bureau balance ───────────────────────────────────────────────────────────

def engineer_bureau_balance_features(bubl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Aggregate bureau_balance by SK_ID_BUREAU."""
    bubl_last_DPD = bubl[bubl.STATUS.isin(["1","2","3","4","5"])].groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].max()
    bubl_last_DPD.rename("MONTH_LAST_DPD", inplace=True)
    bubl_last_C   = bubl[bubl.STATUS == "C"].groupby("SK_ID_BUREAU")["MONTHS_BALANCE"].min()
    bubl_last_C.rename("MONTH_LAST_C", inplace=True)

    STATUS_TCNT = pd.Series(
        bubl.groupby("SK_ID_BUREAU")["STATUS"].value_counts(), name="STATUS_TCNT"
    )
    STATUS_TCNT = pd.pivot_table(
        STATUS_TCNT.reset_index(), index="SK_ID_BUREAU", columns="STATUS",
        values="STATUS_TCNT", fill_value=0
    )
    STATUS_TCNT["DPD_SUM"] = np.zeros(len(STATUS_TCNT))
    count = np.zeros(len(STATUS_TCNT))
    for i in range(6):
        if str(i) in STATUS_TCNT.columns:
            STATUS_TCNT["DPD_SUM"] += STATUS_TCNT[str(i)] * i
            count += STATUS_TCNT[str(i)]
            del STATUS_TCNT[str(i)]
    STATUS_TCNT["DPD_MEAN"] = STATUS_TCNT["DPD_SUM"] / (count + 0.0001)
    STATUS_TCNT.columns = ["STATUS_TCNT_" + f for f in STATUS_TCNT.columns]

    STATUS_12CNT = pd.Series(
        bubl[bubl["MONTHS_BALANCE"] >= -12].groupby("SK_ID_BUREAU")["STATUS"].value_counts(),
        name="STATUS_6CNT"
    )
    STATUS_12CNT = pd.pivot_table(
        STATUS_12CNT.reset_index(), index="SK_ID_BUREAU", columns="STATUS",
        values="STATUS_6CNT", fill_value=0
    )
    STATUS_12CNT["DPD_SUM"] = np.zeros(len(STATUS_12CNT))
    count12 = np.zeros(len(STATUS_12CNT))
    for i in range(6):
        if str(i) in STATUS_12CNT.columns:
            STATUS_12CNT["DPD_SUM"] += STATUS_12CNT[str(i)] * i
            count12 += STATUS_12CNT[str(i)]
            del STATUS_12CNT[str(i)]
    STATUS_12CNT["DPD_MEAN"] = STATUS_12CNT["DPD_SUM"] / (count12 + 0.0001)
    STATUS_12CNT.columns = ["STATUS_12CNT_" + f for f in STATUS_12CNT.columns]

    return STATUS_TCNT, STATUS_12CNT, bubl_last_DPD, bubl_last_C


def engineer_bureau_features(
    buro: pd.DataFrame,
    STATUS_TCNT: pd.DataFrame,
    STATUS_12CNT: pd.DataFrame,
    bubl_last_DPD: pd.Series,
    bubl_last_C: pd.Series,
) -> pd.DataFrame:
    """Aggregate bureau.csv → ~120 features per SK_ID_CURR."""
    # Clean extreme values
    for col in ["DAYS_CREDIT_ENDDATE", "DAYS_CREDIT_UPDATE", "DAYS_ENDDATE_FACT"]:
        buro.loc[buro[col] < -40000, col] = np.nan

    # Ratio columns
    buro["AMT_DEBT_RATIO"]        = buro["AMT_CREDIT_SUM_DEBT"] / (1 + buro["AMT_CREDIT_SUM"])
    buro["AMT_LIMIT_RATIO"]       = buro["AMT_CREDIT_SUM_LIMIT"] / (1 + buro["AMT_CREDIT_SUM"])
    buro["AMT_SUM_OVERDUE_RATIO"] = buro["AMT_CREDIT_SUM_OVERDUE"] / (1 + buro["AMT_CREDIT_SUM"])
    buro["AMT_MAX_OVERDUE_RATIO"] = buro["AMT_CREDIT_MAX_OVERDUE"] / (1 + buro["AMT_CREDIT_SUM"])
    buro["DAYS_END_DIFF"]         = buro["DAYS_ENDDATE_FACT"] - buro["DAYS_CREDIT_ENDDATE"]

    # Most recent bureau record per customer
    idx = buro.groupby("SK_ID_CURR")["DAYS_CREDIT"].idxmax()
    buro_recent = buro.loc[idx.values].copy()
    buro_recent.columns = ["recent_" + f for f in buro_recent.columns]
    cat_recent = [f for f in buro_recent.columns if buro_recent[f].dtype == "object"]
    for f in cat_recent:
        buro_recent[f], _ = pd.factorize(buro_recent[f])
    del buro_recent["recent_SK_ID_BUREAU"]
    buro_recent.rename(columns={"recent_SK_ID_CURR": "SK_ID_CURR"}, inplace=True)
    buro_recent.set_index("SK_ID_CURR", inplace=True)

    # Merge bureau_balance features
    for f in STATUS_TCNT.columns:
        buro[f] = buro["SK_ID_BUREAU"].map(STATUS_TCNT[f])
    for f in STATUS_12CNT.columns:
        buro[f] = buro["SK_ID_BUREAU"].map(STATUS_12CNT[f])
    buro["MONTH_LAST_DPD"] = buro["SK_ID_BUREAU"].map(bubl_last_DPD)
    buro["MONTH_LAST_C"]   = buro["SK_ID_BUREAU"].map(bubl_last_C)

    # One-hot encode categoricals
    buro_cat_features = [f for f in buro.columns if buro[f].dtype == "object"]
    for f in buro_cat_features:
        if buro[f].nunique(dropna=False) <= 2:
            buro[f], _ = pd.factorize(buro[f])
        else:
            buro = pd.concat([buro, pd.get_dummies(buro[f], prefix=f)], axis=1)
            del buro[f]

    # Aggregations
    max_feats = ["MONTH_LAST_DPD", "MONTH_LAST_C", "DAYS_CREDIT", "DAYS_CREDIT_ENDDATE"]
    min_feats = ["MONTH_LAST_DPD", "MONTH_LAST_C", "DAYS_CREDIT", "DAYS_CREDIT_ENDDATE"]
    avg_feats = [f for f in buro.columns if "DAY" in f]

    max_buro = buro[[*max_feats, "SK_ID_CURR"]].groupby("SK_ID_CURR").max()
    max_buro.columns = ["max_" + f for f in max_buro.columns]
    min_buro = buro[[*min_feats, "SK_ID_CURR"]].groupby("SK_ID_CURR").max()
    min_buro.columns = ["min_" + f for f in min_buro.columns]
    avg_buro = buro[[*avg_feats, "SK_ID_CURR"]].groupby("SK_ID_CURR").mean()
    avg_buro.columns = ["avg_" + f for f in avg_buro.columns]

    sum_feats = [f for f in buro.columns if f not in ("SK_ID_CURR", "SK_ID_BUREAU")]
    sum_buro = buro[[*sum_feats, "SK_ID_CURR"]].groupby("SK_ID_CURR").sum()
    for cat in buro_cat_features:
        cols = [f for f in sum_buro.columns if cat in f]
        sum_buro[cat + "_mode"] = sum_buro[cols].idxmax(axis=1)
        if len(cols) >= 10:
            for c in cols:
                del sum_buro[c]
    sum_buro.columns = ["sum_" + f for f in sum_buro.columns]

    # Active bureau loans
    if "CREDIT_ACTIVE_Active" in buro.columns:
        active_buro = buro.loc[buro["CREDIT_ACTIVE_Active"] == 1].copy()
    else:
        active_buro = buro.loc[buro.get("CREDIT_ACTIVE", "") == "Active"].copy() if len(buro) > 0 else buro.iloc[0:0].copy()

    if len(active_buro) > 0:
        active_buro["DAYS_LEFT_RATIO"] = (
            active_buro["DAYS_CREDIT_ENDDATE"] /
            (active_buro["DAYS_CREDIT_ENDDATE"] - active_buro["DAYS_CREDIT"] + 0.001)
        )
        active_buro["AMT_CREDIT_LEFT"] = active_buro["AMT_CREDIT_SUM"] * active_buro["DAYS_LEFT_RATIO"]
        active_buro["AMT_CREDIT_LEFT_OVER_ANNUITY"] = (
            active_buro["AMT_CREDIT_LEFT"] / (active_buro["AMT_ANNUITY"] + 0.001)
        )
        active_sum_cols = [f for f in sum_feats if not any(x in f for x in
            ["CREDIT_CURRENCY", "CREDIT_ACTIVE", "STATUS_", "MONTH_", "CREDIT_TYPE"]
        )] + ["AMT_CREDIT_LEFT", "AMT_CREDIT_LEFT_OVER_ANNUITY"]
        active_sum_cols = [f for f in active_sum_cols if f in active_buro.columns]

        active_sum_buro  = active_buro[[*active_sum_cols, "SK_ID_CURR"]].groupby("SK_ID_CURR").sum()
        active_sum_buro.columns = ["active_sum_" + f for f in active_sum_buro.columns]
        active_sum_buro["active_count"] = buro.loc[
            buro.get("CREDIT_ACTIVE_Active", pd.Series(dtype=int)) == 1
        ].groupby("SK_ID_CURR")["SK_ID_BUREAU"].nunique()

        active_avg_cols = active_sum_cols + ["DAYS_LEFT_RATIO"]
        active_avg_cols = [f for f in active_avg_cols if f in active_buro.columns]
        active_avg_buro = active_buro[[*active_avg_cols, "SK_ID_CURR"]].groupby("SK_ID_CURR").mean()
        active_avg_buro.columns = ["active_avg_" + f for f in active_avg_buro.columns]
        for ratio_name, num_col, den_col in [
            ("active_AMT_DEBT_TOTAL_RATIO",        "AMT_CREDIT_SUM_DEBT",    "AMT_CREDIT_SUM"),
            ("active_AMT_LIMIT_TOTAL_RATIO",       "AMT_CREDIT_SUM_LIMIT",   "AMT_CREDIT_SUM"),
            ("active_AMT_SUM_OVERDUE_TOTAL_RATIO", "AMT_CREDIT_SUM_OVERDUE", "AMT_CREDIT_SUM"),
            ("active_AMT_MAX_OVERDUE_TOTAL_RATIO", "AMT_CREDIT_MAX_OVERDUE", "AMT_CREDIT_SUM"),
        ]:
            active_avg_buro[ratio_name] = (
                active_buro.groupby("SK_ID_CURR")[num_col].sum() /
                active_buro.groupby("SK_ID_CURR")[den_col].sum()
            )
    else:
        active_sum_buro = pd.DataFrame(index=avg_buro.index)
        active_avg_buro = pd.DataFrame(index=avg_buro.index)

    # Merge all
    result = avg_buro.merge(min_buro, how="outer", on="SK_ID_CURR")
    result = result.merge(max_buro, how="outer", on="SK_ID_CURR")
    result = result.merge(sum_buro, how="outer", on="SK_ID_CURR")
    result = result.merge(active_sum_buro, how="outer", on="SK_ID_CURR")
    result = result.merge(active_avg_buro, how="outer", on="SK_ID_CURR")
    result = result.merge(buro_recent, how="outer", on="SK_ID_CURR")

    for cur_cols in [["sum_CREDIT_CURRENCY_currency 2", "sum_CREDIT_CURRENCY_currency 3", "sum_CREDIT_CURRENCY_currency 4"]]:
        existing = [c for c in cur_cols if c in result.columns]
        if existing:
            result["used_other_currency"] = (result[existing].sum(axis=1) > 0).astype(int)
    result["count"] = buro.groupby("SK_ID_CURR")["SK_ID_BUREAU"].nunique()
    result["AMT_DEBT_TOTAL_RATIO"]        = buro.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].sum() / buro.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].sum()
    result["AMT_LIMIT_TOTAL_RATIO"]       = buro.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_LIMIT"].sum() / buro.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].sum()
    result["AMT_SUM_OVERDUE_TOTAL_RATIO"] = buro.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_OVERDUE"].sum() / buro.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].sum()
    result["AMT_MAX_OVERDUE_TOTAL_RATIO"] = buro.groupby("SK_ID_CURR")["AMT_CREDIT_MAX_OVERDUE"].sum() / buro.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].sum()

    result.columns = ["bureau_" + f for f in result.columns]
    result = downcast_dtypes(result)
    return result


# ─── Credit card balance ──────────────────────────────────────────────────────

def engineer_credit_card_features(ccbl: pd.DataFrame) -> pd.DataFrame:
    """Aggregate credit_card_balance → ~216 features per SK_ID_CURR."""
    sum_feats = [f for f in ccbl.columns if (
        ("AMT" in f) or ("SK_DPD" in f) or ("CNT" in f and "CUM" not in f)
    )]
    sum_ccbl_mon = ccbl.groupby(["SK_ID_CURR", "MONTHS_BALANCE"])[sum_feats].sum()
    sum_ccbl_mon["CNT_ACCOUNT_W_MONTH"] = ccbl.groupby(["SK_ID_CURR", "MONTHS_BALANCE"])["SK_ID_PREV"].count()
    sum_ccbl_mon = sum_ccbl_mon.reset_index()

    # Ratio features
    sum_ccbl_mon["AMT_BALANCE_CREDIT_RATIO"]   = (sum_ccbl_mon["AMT_BALANCE"] / (sum_ccbl_mon["AMT_CREDIT_LIMIT_ACTUAL"] + 0.001)).clip(-100, 100)
    sum_ccbl_mon["AMT_CREDIT_USE_RATIO"]       = (sum_ccbl_mon["AMT_DRAWINGS_CURRENT"] / (sum_ccbl_mon["AMT_CREDIT_LIMIT_ACTUAL"] + 0.001)).clip(-100, 100)
    sum_ccbl_mon["AMT_DRAWING_ATM_RATIO"]      = sum_ccbl_mon["AMT_DRAWINGS_ATM_CURRENT"] / (sum_ccbl_mon["AMT_DRAWINGS_CURRENT"] + 0.001)
    sum_ccbl_mon["AMT_DRAWINGS_OTHER_RATIO"]   = sum_ccbl_mon.get("AMT_DRAWINGS_OTHER_CURRENT", 0) / (sum_ccbl_mon["AMT_DRAWINGS_CURRENT"] + 0.001)
    sum_ccbl_mon["AMT_DRAWINGS_POS_RATIO"]     = sum_ccbl_mon.get("AMT_DRAWINGS_POS_CURRENT", 0) / (sum_ccbl_mon["AMT_DRAWINGS_CURRENT"] + 0.001)
    sum_ccbl_mon["AMT_PAY_USE_RATIO"]          = ((sum_ccbl_mon["AMT_PAYMENT_TOTAL_CURRENT"] + 0.001) / (sum_ccbl_mon["AMT_DRAWINGS_CURRENT"] + 0.001)).clip(-100, 100)
    sum_ccbl_mon["AMT_BALANCE_RECIVABLE_RATIO"] = sum_ccbl_mon["AMT_BALANCE"] / (sum_ccbl_mon["AMT_TOTAL_RECEIVABLE"] + 0.001)
    sum_ccbl_mon["AMT_DRAWING_BALANCE_RATIO"]  = sum_ccbl_mon["AMT_DRAWINGS_CURRENT"] / (sum_ccbl_mon["AMT_BALANCE"] + 0.001)
    sum_ccbl_mon["AMT_RECEIVABLE_PRINCIPAL_DIFF"] = sum_ccbl_mon["AMT_TOTAL_RECEIVABLE"] - sum_ccbl_mon.get("AMT_RECEIVABLE_PRINCIPAL", 0)
    sum_ccbl_mon["AMT_PAY_INST_DIFF"] = sum_ccbl_mon["AMT_PAYMENT_CURRENT"] - sum_ccbl_mon.get("AMT_INST_MIN_REGULARITY", 0)

    # Drop redundant
    for f in ["AMT_RECIVABLE", "AMT_RECEIVABLE_PRINCIPAL", "AMT_DRAWINGS_OTHER_CURRENT", "AMT_DRAWINGS_POS_CURRENT"]:
        if f in sum_ccbl_mon.columns:
            del sum_ccbl_mon[f]

    sum_feats2 = [f for f in sum_ccbl_mon.columns if (("AMT" in f) or ("SK_DPD" in f) or ("CNT" in f and "CUM" not in f))]

    # Time window means
    mean4_ccbl  = sum_ccbl_mon.loc[sum_ccbl_mon.MONTHS_BALANCE >= -4].groupby("SK_ID_CURR").mean()
    del mean4_ccbl["MONTHS_BALANCE"]
    mean4_ccbl.columns = ["mean4_" + f for f in mean4_ccbl.columns]

    mean12_ccbl = sum_ccbl_mon.loc[sum_ccbl_mon.MONTHS_BALANCE >= -12].groupby("SK_ID_CURR").mean()
    del mean12_ccbl["MONTHS_BALANCE"]
    mean12_ccbl.columns = ["mean12_" + f for f in mean12_ccbl.columns]

    mean36_ccbl = sum_ccbl_mon.loc[sum_ccbl_mon.MONTHS_BALANCE >= -36].groupby("SK_ID_CURR").mean()
    del mean36_ccbl["MONTHS_BALANCE"]
    mean36_ccbl.columns = ["mean36_" + f for f in mean36_ccbl.columns]

    # Scaled sum/mean (exponential time weighting)
    sum_ccbl_mon2 = sum_ccbl_mon.copy(deep=True)
    sum_ccbl_mon2["YEAR_SCALE"] = (sum_ccbl_mon2["MONTHS_BALANCE"] / 12.0).apply(np.exp)
    for f in sum_feats2:
        if f in sum_ccbl_mon2.columns:
            sum_ccbl_mon2[f] = sum_ccbl_mon2[f] * sum_ccbl_mon2["YEAR_SCALE"]
    scale_sum_ccbl = sum_ccbl_mon2.groupby("SK_ID_CURR").sum()
    for c in ["MONTHS_BALANCE", "YEAR_SCALE"]:
        if c in scale_sum_ccbl.columns:
            del scale_sum_ccbl[c]
    scale_sum_ccbl.columns = ["scale_sum_" + f for f in scale_sum_ccbl.columns]
    year_scale_sum = sum_ccbl_mon2.groupby("SK_ID_CURR")["YEAR_SCALE"].sum()
    scale_mean_ccbl = pd.DataFrame()
    for f in scale_sum_ccbl.columns:
        scale_mean_ccbl[f] = scale_sum_ccbl[f] / year_scale_sum
    scale_mean_ccbl.columns = ["scale_mean_" + f for f in scale_mean_ccbl.columns]

    # Overall mean, var, max, min
    if "MONTHS_BALANCE" in sum_ccbl_mon.columns:
        del sum_ccbl_mon["MONTHS_BALANCE"]
    mean_ccbl = sum_ccbl_mon.groupby("SK_ID_CURR").mean()
    mean_ccbl.columns = ["mean_" + f for f in mean_ccbl.columns]
    var_ccbl  = sum_ccbl_mon.groupby("SK_ID_CURR").var()
    var_ccbl.columns  = ["var_" + f for f in var_ccbl.columns]
    max_ccbl  = sum_ccbl_mon.groupby("SK_ID_CURR").max()
    max_ccbl.columns  = ["max_" + f for f in max_ccbl.columns]
    min_cols = [c for c in ["AMT_TOTAL_RECEIVABLE", "AMT_RECEIVABLE_PRINCIPAL_DIFF"] if c in sum_ccbl_mon.columns]
    min_ccbl  = sum_ccbl_mon.groupby("SK_ID_CURR")[min_cols].min()
    min_ccbl.columns  = ["min_" + f for f in min_ccbl.columns]

    # Last DPD
    ccbl_last_DPD  = ccbl[ccbl.SK_DPD > 0].groupby("SK_ID_CURR")["MONTHS_BALANCE"].max()
    ccbl_last_DPD.rename("MONTH_LAST_DPD", inplace=True)
    ccbl_last_DPD7 = ccbl[ccbl.SK_DPD_DEF > 7].groupby("SK_ID_CURR")["MONTHS_BALANCE"].max()
    ccbl_last_DPD7.rename("MONTH_LAST_DPD7", inplace=True)

    # Most recent
    idx    = ccbl.groupby("SK_ID_CURR")["MONTHS_BALANCE"].idxmax()
    recent = ccbl[["SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT_MATURE_CUM",
                   "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"]].iloc[idx.values].copy()
    recent["NAME_CONTRACT_STATUS"], _ = pd.factorize(recent["NAME_CONTRACT_STATUS"])
    recent.set_index("SK_ID_CURR", inplace=True)

    NAME_STATUS_CNT = pd.Series(
        ccbl.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"].value_counts(),
        name="NAME_CONTRACT_STATUS_COUNT"
    )
    NAME_STATUS_CNT = pd.pivot_table(
        NAME_STATUS_CNT.reset_index(), index="SK_ID_CURR",
        columns="NAME_CONTRACT_STATUS", values="NAME_CONTRACT_STATUS_COUNT", fill_value=0
    )
    recent = recent.merge(NAME_STATUS_CNT, how="outer", on="SK_ID_CURR")

    # Merge all
    ccbl_mon = mean4_ccbl.copy()
    for df_part in [mean12_ccbl, mean36_ccbl, scale_sum_ccbl, scale_mean_ccbl, mean_ccbl, var_ccbl, max_ccbl, min_ccbl]:
        ccbl_mon = ccbl_mon.merge(df_part, how="outer", on="SK_ID_CURR")
    ccbl_mon["MONTH_LAST_DPD"]  = ccbl_last_DPD
    ccbl_mon["MONTH_LAST_DPD7"] = ccbl_last_DPD7
    ccbl_mon = ccbl_mon.merge(recent, how="outer", on="SK_ID_CURR")

    ccbl_mon.fillna(0, inplace=True)
    ccbl_mon.columns = ["cc_" + f for f in ccbl_mon.columns]
    ccbl_mon = downcast_dtypes(ccbl_mon)
    return ccbl_mon


# ─── POS cash balance ─────────────────────────────────────────────────────────

def engineer_pos_cash_features(pos: pd.DataFrame) -> pd.DataFrame:
    """Aggregate POS_CASH_balance → ~27 features per SK_ID_CURR."""
    # Most recent record per SK_ID_PREV (for use with prev_application)
    idx = pos.groupby("SK_ID_PREV")["MONTHS_BALANCE"].idxmax()
    pos_prev_last = pos[["SK_ID_PREV", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE"]].loc[idx.values].copy()
    pos_prev_last["INSTAL_LEFT_RATIO"] = pos_prev_last["CNT_INSTALMENT_FUTURE"] / (pos_prev_last["CNT_INSTALMENT"] + 0.001)
    pos_prev_last.set_index("SK_ID_PREV", inplace=True)

    # Most recent per SK_ID_CURR
    idx = pos.groupby("SK_ID_CURR")["MONTHS_BALANCE"].idxmax()
    pos_recent = pos[["SK_ID_CURR", "MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE",
                       "NAME_CONTRACT_STATUS", "SK_DPD", "SK_DPD_DEF"]].loc[idx.values].copy()
    pos_recent["NAME_CONTRACT_STATUS"], _ = pd.factorize(pos_recent["NAME_CONTRACT_STATUS"])
    pos_recent.set_index("SK_ID_CURR", inplace=True)
    pos_recent.columns = ["recent_" + f for f in pos_recent.columns]

    NAME_STATUS_CNT = pd.Series(
        pos.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS"].value_counts(),
        name="NAME_CONTRACT_STATUS_COUNT"
    )
    NAME_STATUS_CNT = pd.pivot_table(
        NAME_STATUS_CNT.reset_index(), index="SK_ID_CURR",
        columns="NAME_CONTRACT_STATUS", values="NAME_CONTRACT_STATUS_COUNT", fill_value=0
    )
    NAME_STATUS_CNT.columns = ["NAME_CONTRACT_STATUS_CNT_" + f for f in NAME_STATUS_CNT.columns]

    # Time-scaled DPD aggregation
    pos["YEAR_SCALE"]        = (pos["MONTHS_BALANCE"] / 12.0).apply(np.exp)
    pos["SK_DPD_SCALE"]      = pos["SK_DPD"] * pos["YEAR_SCALE"]
    pos["SK_DPD_DEF_SCALE"]  = pos["SK_DPD_DEF"] * pos["YEAR_SCALE"]

    pos_max        = pos.groupby("SK_ID_CURR")[["SK_DPD", "SK_DPD_DEF"]].max()
    pos_max.columns = ["max_" + f for f in pos_max.columns]
    pos_mean       = pos.groupby("SK_ID_CURR")[["SK_DPD", "SK_DPD_DEF"]].mean()
    pos_mean.columns = ["mean_" + f for f in pos_mean.columns]
    pos_sum        = pos.groupby("SK_ID_CURR")[["SK_DPD_SCALE", "SK_DPD_DEF_SCALE"]].sum()
    pos_year_sum   = pos.groupby("SK_ID_CURR")["YEAR_SCALE"].sum()
    pos_mean_scale = pd.DataFrame()
    for f in pos_sum.columns:
        pos_mean_scale[f] = pos_sum[f] / pos_year_sum
    pos_sum.columns        = ["sum_" + f for f in pos_sum.columns]
    pos_mean_scale.columns = ["mean_" + f for f in pos_mean_scale.columns]

    pos_last_DPD = pos[pos.SK_DPD > 0].groupby("SK_ID_CURR")["MONTHS_BALANCE"].max()
    pos_last_DPD.rename("MONTH_LAST_DPD", inplace=True)

    pos_recent = pos_recent.merge(pos_max, how="outer", on="SK_ID_CURR")
    pos_recent = pos_recent.merge(pos_mean, how="outer", on="SK_ID_CURR")
    pos_recent = pos_recent.merge(pos_sum, how="outer", on="SK_ID_CURR")
    pos_recent = pos_recent.merge(pos_mean_scale, how="outer", on="SK_ID_CURR")
    pos_recent["MONTH_LAST_DPD"] = pos_last_DPD
    pos_recent = pos_recent.merge(NAME_STATUS_CNT, how="outer", on="SK_ID_CURR")
    pos_recent["MONTH_CNT"] = pos.groupby("SK_ID_CURR")["MONTHS_BALANCE"].count()
    pos_recent["MONTH_MAX"] = pos.groupby("SK_ID_CURR")["MONTHS_BALANCE"].min()
    pos_recent["count"]     = pos.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique()

    pos_recent.fillna(0, inplace=True)
    pos_recent = downcast_dtypes(pos_recent)
    pos_recent.columns = ["pos_" + f for f in pos_recent.columns]
    # Return both: pos features AND pos_prev_last (needed by engineer_prev_application_features)
    return pos_recent, pos_prev_last


# ─── Installment payments ─────────────────────────────────────────────────────

def engineer_installment_features(inst: pd.DataFrame) -> pd.DataFrame:
    """Aggregate installments_payments → ~27 time-weighted features per SK_ID_CURR."""
    inst_NUM_INSTALMENT_VERSION = inst.groupby("SK_ID_CURR")["NUM_INSTALMENT_VERSION"].nunique()

    # Merge same-month payments (weighted by payment amount)
    inst["DAYS_ENTRY_PAYMENT_weighted"] = inst["DAYS_ENTRY_PAYMENT"] * inst["AMT_PAYMENT"]
    inst = inst.groupby(["SK_ID_PREV", "SK_ID_CURR", "NUM_INSTALMENT_NUMBER"]).agg({
        "DAYS_INSTALMENT": "mean",
        "DAYS_ENTRY_PAYMENT_weighted": "sum",
        "AMT_INSTALMENT": "mean",
        "AMT_PAYMENT": "sum",
    })
    inst["DAYS_ENTRY_PAYMENT"] = inst["DAYS_ENTRY_PAYMENT_weighted"] / inst["AMT_PAYMENT"]
    inst = inst.reset_index()
    del inst["DAYS_ENTRY_PAYMENT_weighted"]

    # Engineered features
    inst["AMT_PAYMENT_PERC"]  = inst["AMT_PAYMENT"] / (inst["AMT_INSTALMENT"] + 0.001)
    inst["DPD"]               = (inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]).clip(lower=0).fillna(30)
    inst["DBD"]               = (inst["DAYS_INSTALMENT"] - inst["DAYS_ENTRY_PAYMENT"]).clip(lower=0).fillna(0)
    inst["AMT_PAYMENT_DIFF"]  = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]

    # Time-weighted scale: exp(DAYS/365.25)
    inst["DAYS_ENTRY_PAYMENT_SCALE"] = (inst["DAYS_ENTRY_PAYMENT"] / 365.25).apply(np.exp)
    inst["DPD_SCALE"]                = inst["DPD"] * inst["DAYS_ENTRY_PAYMENT_SCALE"]
    inst["DBD_SCALE"]                = inst["DBD"] * inst["DAYS_ENTRY_PAYMENT_SCALE"]
    inst["AMT_PAYMENT_DIFF_SCALE"]   = inst["AMT_PAYMENT_DIFF"] * inst["DAYS_ENTRY_PAYMENT_SCALE"]
    inst["AMT_PAYMENT_SCALE"]        = inst["AMT_PAYMENT"] * inst["DAYS_ENTRY_PAYMENT_SCALE"]

    inst_max = inst.groupby("SK_ID_CURR")[["DPD", "DBD", "AMT_PAYMENT_DIFF", "AMT_PAYMENT_PERC"]].max()
    inst_max.columns = ["max_" + f for f in inst_max.columns]
    inst_var = inst.groupby("SK_ID_CURR")[["DPD", "DBD", "AMT_PAYMENT_DIFF", "AMT_PAYMENT_PERC"]].var()
    inst_var.columns = ["var_" + f for f in inst_var.columns]
    inst_sum = inst.groupby("SK_ID_CURR")[["DPD_SCALE", "DBD_SCALE", "AMT_PAYMENT_DIFF_SCALE", "AMT_PAYMENT_SCALE"]].sum()

    inst_day_scale_sum = inst.groupby("SK_ID_CURR")["DAYS_ENTRY_PAYMENT_SCALE"].sum()
    inst_avg_scale = pd.DataFrame()
    for f in inst_sum.columns:
        inst_avg_scale[f] = inst_sum[f] / inst_day_scale_sum
    inst_sum.columns        = ["sum_" + f for f in inst_sum.columns]
    inst_avg_scale.columns  = ["mean_" + f for f in inst_avg_scale.columns]

    inst_avg = inst.groupby("SK_ID_CURR")[["DPD", "DBD", "AMT_PAYMENT_DIFF", "AMT_PAYMENT", "AMT_PAYMENT_PERC"]].mean()
    inst_avg.columns = ["mean_" + f for f in inst_avg.columns]

    inst_last_late     = inst[inst.DAYS_INSTALMENT < inst.DAYS_ENTRY_PAYMENT].groupby("SK_ID_CURR")["DAYS_INSTALMENT"].max()
    inst_last_late.rename("DAYS_LAST_LATE", inplace=True)
    inst_last_underpaid = inst[inst.AMT_INSTALMENT < inst.AMT_PAYMENT].groupby("SK_ID_CURR")["DAYS_INSTALMENT"].max()
    inst_last_underpaid.rename("DAYS_LAST_UNDERPAID", inplace=True)

    inst_avg = inst_avg.merge(inst_max, on="SK_ID_CURR", how="outer")
    inst_avg = inst_avg.merge(inst_var, on="SK_ID_CURR", how="outer")
    inst_avg = inst_avg.merge(inst_sum, on="SK_ID_CURR", how="outer")
    inst_avg = inst_avg.merge(inst_avg_scale, on="SK_ID_CURR", how="outer")
    inst_avg["DAYS_LAST_LATE"]             = inst_last_late
    inst_avg["DAYS_LAST_UNDERPAID"]        = inst_last_underpaid
    inst_avg["N_NUM_INSTALMENT_VERSION"]   = inst_NUM_INSTALMENT_VERSION
    inst_avg["AMT_PAYMENT_TOTAL_RATIO"]    = (
        inst.groupby("SK_ID_CURR")["AMT_PAYMENT"].sum() /
        inst.groupby("SK_ID_CURR")["AMT_INSTALMENT"].sum()
    )
    inst_avg["length"] = inst.groupby("SK_ID_CURR")["SK_ID_PREV"].count()
    inst_avg["count"]  = inst.groupby("SK_ID_CURR")["SK_ID_PREV"].nunique()
    inst_avg.columns   = ["inst_" + f for f in inst_avg.columns]
    inst_avg = downcast_dtypes(inst_avg)
    return inst_avg


def engineer_prev_application_features(
    prev: pd.DataFrame,
    inst_prev_last: pd.Series,
    pos_prev_last: pd.DataFrame,
    inst_target1: np.ndarray,
    pos_target1: np.ndarray,
    cc_target1: np.ndarray,
    app_credit_annuity: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate previous_application -> ~232 features per SK_ID_CURR.

    Mirror of lgb1.ipynb cell 14. All gaps vs reference fixed:
      - FLAG_LAST_APPL_PER_CONTRACT filter
      - APP_CREDIT_PERC feature
      - DAYS_TOTAL2 correct formula
      - >360000 threshold (not exact 365243)
      - DEFAULTED cross-reference with inst/pos/cc targets
      - approved/refused aggregation
      - closest_credit/annuity_defaulted
      - active_sum_prev
    """
    # Gap 1: filter mistake rows
    prev = prev.loc[prev['FLAG_LAST_APPL_PER_CONTRACT'] == 'Y'].copy()
    del prev['FLAG_LAST_APPL_PER_CONTRACT']

    # Gap 4: threshold >360000 (not exact 365243)
    for col in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
                "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        if col in prev.columns:
            prev.loc[prev[col] > 360000, col] = np.nan

    # Feature engineering
    # Gap 2: APP_CREDIT_PERC
    prev["APP_CREDIT_PERC"]       = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]
    prev["AMT_DIFF_CREAPP"]       = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    prev["AMT_DIFF_CREDIT_GOODS"] = prev["AMT_CREDIT"] - prev.get("AMT_GOODS_PRICE", np.nan)
    prev["AMT_CREDIT_GOODS_PERC"] = prev["AMT_CREDIT"] / (prev.get("AMT_GOODS_PRICE", np.nan))
    prev["AMT_PAY_YEAR"]          = prev["AMT_CREDIT"] / (prev["AMT_ANNUITY"] + 0.001)
    prev["DAYS_TOTAL"]            = prev["DAYS_LAST_DUE"] - prev["DAYS_FIRST_DUE"]
    # Gap 3: DAYS_TOTAL2 correct formula
    prev["DAYS_TOTAL2"]           = prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_FIRST_DUE"]
    prev["DAYS_END_DIFF"]         = prev["DAYS_LAST_DUE_1ST_VERSION"] - prev["DAYS_LAST_DUE"]
    prev["CNT_PAYMENT_DIFF"]      = prev["AMT_PAY_YEAR"] - prev["SK_ID_PREV"].map(pos_prev_last["CNT_INSTALMENT"])

    # Gap 5: DEFAULTED cross-reference with inst/pos/cc targets
    prev["DEFAULTED"] = 0
    prev.loc[prev["SK_ID_PREV"].isin(inst_target1), "DEFAULTED"] = 1
    prev.loc[prev["SK_ID_PREV"].isin(pos_target1),  "DEFAULTED"] = 1
    prev.loc[prev["SK_ID_PREV"].isin(cc_target1),   "DEFAULTED"] = 1
    prev.loc[prev["NAME_CONTRACT_STATUS"] != "Approved", "DEFAULTED"] = np.nan

    # Rejected features
    for f in ["AMT_GOODS_PRICE", "WEEKDAY_APPR_PROCESS_START",
               "HOUR_APPR_PROCESS_START", "NFLAG_LAST_APPL_IN_DAY"]:
        if f in prev.columns:
            del prev[f]

    # Most recent application (before one-hot so we can track categorical)
    idx = prev.groupby("SK_ID_CURR")["DAYS_DECISION"].idxmax()
    prev_recent = prev.loc[idx.values].copy()
    prev_recent.columns = ["recent_" + f for f in prev_recent.columns]
    cat_recent = [f for f in prev_recent.columns if prev_recent[f].dtype == "object"]
    for f in cat_recent:
        prev_recent[f], _ = pd.factorize(prev_recent[f])
    del prev_recent["recent_SK_ID_PREV"]
    prev_recent.rename(columns={"recent_SK_ID_CURR": "SK_ID_CURR"}, inplace=True)
    prev_recent.set_index("SK_ID_CURR", inplace=True)

    # One-hot encode categoricals
    prev_cat_features = [f for f in prev.columns if prev[f].dtype == "object"]
    for f in prev_cat_features:
        if prev[f].nunique(dropna=False) <= 2:
            prev[f], _ = pd.factorize(prev[f])
        else:
            prev = pd.concat([prev, pd.get_dummies(prev[f], prefix=f)], axis=1)
            del prev[f]

    avg_feats = [f for f in prev.columns.values
                 if ("DAYS" in f) or ("RATE" in f) or ("AMT" in f)]
    # clean extreme
    for f in avg_feats:
        if f in prev.columns:
            prev.loc[prev[f] > 300000, f] = np.nan
    avg_feats = [f for f in avg_feats if f in prev.columns]
    avg_prev = prev[[*avg_feats, "SK_ID_CURR"]].groupby("SK_ID_CURR").mean()
    avg_prev.columns = ["avg_" + f for f in avg_prev.columns]

    max_feats = [f for f in prev.columns.values if ("DAYS" in f) or ("AMT" in f)]
    max_feats = [f for f in max_feats if f in prev.columns]
    max_prev = prev[[*max_feats, "SK_ID_CURR"]].groupby("SK_ID_CURR").max()
    max_prev.columns = ["max_" + f for f in max_prev.columns]

    min_prev = prev[["DAYS_DECISION", "SK_ID_CURR"]].groupby("SK_ID_CURR").min()
    min_prev.columns = ["min_" + f for f in min_prev.columns]

    nosum_feats = {"SK_ID_CURR", "SK_ID_PREV", "DAYS_TOTAL", "DAYS_TOTAL2",
                  "DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
                  "DAYS_LAST_DUE", "DAYS_TERMINATION", "RATE_DOWN_PAYMENT",
                  "RATE_INTEREST_PRIMARY", "RATE_INTEREST_PRIVILEGED",
                  "AMT_CREDIT_GOODS_PERC", "APP_CREDIT_PERC"}
    sum_feats = [f for f in prev.columns.values if f not in nosum_feats]
    sum_prev = prev[[*sum_feats, "SK_ID_CURR"]].groupby("SK_ID_CURR").sum()
    # mode of categorical features
    for cat_ in prev_cat_features:
        cols = [f for f in sum_prev.columns if cat_ in f]
        if cols:
            sum_prev[cat_ + "_mode"] = sum_prev[cols].idxmax(axis=1)
            if len(cols) >= 10:
                for col in cols:
                    del sum_prev[col]
    sum_prev.columns = ["sum_" + f for f in sum_prev.columns]

    # Gap 8: active loans subset
    prev_active = prev.loc[
        prev["DAYS_LAST_DUE"].isnull() & (prev["DAYS_LAST_DUE_1ST_VERSION"].fillna(0) > 0)
    ].copy()
    if len(prev_active) > 0:
        prev_active["AMT_LEFT"]  = prev_active["AMT_ANNUITY"] * prev_active["DAYS_LAST_DUE_1ST_VERSION"] / 365.25
        prev_active["AMT_PAID"]  = prev_active["SK_ID_PREV"].map(inst_prev_last)
        prev_active["AMT_OWE"]   = (
            (prev_active["AMT_CREDIT"] - prev_active["AMT_DOWN_PAYMENT"].fillna(0)) *
            (1 + prev_active["RATE_INTEREST_PRIVILEGED"].fillna(0))
        )
        prev_active["AMT_LEFT2"] = (prev_active["AMT_OWE"] - prev_active["AMT_PAID"]).clip(lower=0)
        prev_active["LEFT_RATIO"]	= prev_active["SK_ID_PREV"].map(pos_prev_last["INSTAL_LEFT_RATIO"])
        prev_active["AMT_LEFT3"] = prev_active["AMT_CREDIT"] * prev_active["LEFT_RATIO"]
        prev_active["AMT_PAY_YEAR_LEFT"] = prev_active["AMT_LEFT"] / prev_active["AMT_ANNUITY"]
        active_sum_feats = [f for f in prev_active.columns if "AMT" in f]
        active_sum_prev = prev_active[[*active_sum_feats, "SK_ID_CURR"]].groupby("SK_ID_CURR").sum()
        active_sum_prev.columns = ["active_sum_" + f for f in active_sum_prev.columns]
        active_sum_prev["active_count"] = prev_active.groupby("SK_ID_CURR")["SK_ID_PREV"].count()
    else:
        active_sum_prev = pd.DataFrame(index=avg_prev.index)

    # Gap 6: approved / refused aggregation
    num_aggregations = {
        "SK_ID_PREV":            ["count"],
        "AMT_ANNUITY":           ["max", "mean"],
        "AMT_APPLICATION":       ["max", "mean"],
        "AMT_CREDIT":            ["mean", "sum"],
        "APP_CREDIT_PERC":       ["max", "mean"],
        "AMT_DIFF_CREAPP":       ["max", "mean"],
        "AMT_DIFF_CREDIT_GOODS": ["max", "mean"],
        "AMT_CREDIT_GOODS_PERC": ["max", "mean"],
        "AMT_PAY_YEAR":          ["max", "mean"],
        "AMT_DOWN_PAYMENT":      ["max", "mean"],
        "RATE_DOWN_PAYMENT":     ["max", "mean"],
        "DAYS_DECISION":         ["max", "mean", "min"],
        "CNT_PAYMENT":           ["mean", "sum"],
    }
    # Only aggregate columns that actually exist in prev
    num_aggregations = {k: v for k, v in num_aggregations.items() if k in prev.columns}

    approved_col = next((c for c in prev.columns if "NAME_CONTRACT_STATUS" in c and "Approved" in c), None)
    refused_col  = next((c for c in prev.columns if "NAME_CONTRACT_STATUS" in c and "Refused" in c), None)

    if approved_col:
        approved_prev = prev[prev[approved_col] == 1].groupby("SK_ID_CURR").agg(num_aggregations)
        approved_prev.columns = pd.Index(["approved_" + e[0] + "_" + e[1].upper() for e in approved_prev.columns.tolist()])
    else:
        approved_prev = pd.DataFrame(index=avg_prev.index)

    if refused_col:
        refused_prev = prev[prev[refused_col] == 1].groupby("SK_ID_CURR").agg(num_aggregations)
        refused_prev.columns = pd.Index(["refused_" + e[0] + "_" + e[1].upper() for e in refused_prev.columns.tolist()])
    else:
        refused_prev = pd.DataFrame(index=avg_prev.index)

    # Gap 7: closest credit/annuity defaulted
    if "AMT_CREDIT" in prev.columns and "AMT_CREDIT" in app_credit_annuity.columns:
        prev["AMT_CREDIT_DIFF"]  = (prev["AMT_CREDIT"]  - prev["SK_ID_CURR"].map(app_credit_annuity["AMT_CREDIT"])).abs()
        prev["AMT_ANNUITY_DIFF"] = (prev["AMT_ANNUITY"] - prev["SK_ID_CURR"].map(app_credit_annuity["AMT_ANNUITY"])).abs()

        idx_c = prev.groupby("SK_ID_CURR")["AMT_CREDIT_DIFF"].idxmin().dropna()
        prev_closest_credit_defaulted = (
            prev[["SK_ID_CURR", "DEFAULTED"]].loc[idx_c].set_index("SK_ID_CURR")
            .rename(columns={"DEFAULTED": "closest_credit_defaulted"})
        )

        idx_a = prev.groupby("SK_ID_CURR")["AMT_ANNUITY_DIFF"].idxmin().dropna()
        prev_closest_annuity_defaulted = (
            prev[["SK_ID_CURR", "DEFAULTED"]].loc[idx_a].set_index("SK_ID_CURR")
            .rename(columns={"DEFAULTED": "closest_annuity_defaulted"})
        )
    else:
        prev_closest_credit_defaulted  = pd.DataFrame(index=avg_prev.index)
        prev_closest_annuity_defaulted = pd.DataFrame(index=avg_prev.index)

    # Merge all
    result = avg_prev.merge(max_prev, on="SK_ID_CURR", how="outer")
    result = result.merge(sum_prev, on="SK_ID_CURR", how="outer")
    result = result.merge(min_prev, on="SK_ID_CURR", how="outer")
    result = result.merge(active_sum_prev, on="SK_ID_CURR", how="outer")
    result = result.merge(approved_prev, on="SK_ID_CURR", how="outer")
    result = result.merge(refused_prev, on="SK_ID_CURR", how="outer")
    result = result.merge(prev_recent, on="SK_ID_CURR", how="outer")
    result = result.merge(prev_closest_credit_defaulted, on="SK_ID_CURR", how="outer")
    result = result.merge(prev_closest_annuity_defaulted, on="SK_ID_CURR", how="outer")
    result["count"]           = prev.groupby("SK_ID_CURR")["SK_ID_PREV"].count()
    result["DEFALUTED_RATIO"] = prev.groupby("SK_ID_CURR")["DEFAULTED"].mean()

    result.columns = ["prev_" + f for f in result.columns]
    result = downcast_dtypes(result)
    return result


# ─── Main builder ─────────────────────────────────────────────────────────────

def build_all_features(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load all tables, build full feature matrix, return (X, y).

    Mirror of lgb1.ipynb — all 7 gaps vs reference code fixed.
    """
    data_dir = Path(data_dir)

    logger.info("Loading application_train...")
    app = pd.read_csv(data_dir / "application_train.csv",
                      dtype={"SK_ID_CURR": "int32", "TARGET": "int8"})
    y = app["TARGET"].copy()
    # Save AMT_CREDIT / AMT_ANNUITY for prev_app 'closest' feature (Gap 7)
    app_credit_annuity = app[["SK_ID_CURR", "AMT_CREDIT", "AMT_ANNUITY"]].set_index("SK_ID_CURR")

    app, meanenc_feats, cat_feats = engineer_application_features(app)
    logger.info(f"  application: {app.shape}, meanenc_feats={len(meanenc_feats)}")

    logger.info("Loading bureau_balance...")
    bubl = pd.read_csv(data_dir / "bureau_balance.csv",
                       dtype={"SK_ID_BUREAU": "int32"})
    STATUS_TCNT, STATUS_12CNT, bubl_last_DPD, bubl_last_C = engineer_bureau_balance_features(bubl)
    del bubl; gc.collect()

    logger.info("Loading bureau...")
    buro = pd.read_csv(data_dir / "bureau.csv",
                       dtype={"SK_ID_CURR": "int32", "SK_ID_BUREAU": "int32"})
    avg_buro = engineer_bureau_features(buro, STATUS_TCNT, STATUS_12CNT, bubl_last_DPD, bubl_last_C)
    del buro, STATUS_TCNT, STATUS_12CNT; gc.collect()
    logger.info(f"  bureau: {avg_buro.shape}")

    logger.info("Loading credit_card_balance...")
    ccbl = pd.read_csv(data_dir / "credit_card_balance.csv",
                       dtype={"SK_ID_CURR": "int32", "SK_ID_PREV": "int32",
                              "MONTHS_BALANCE": "int16", "SK_DPD": "int16", "SK_DPD_DEF": "int16"})
    # cc_target1: prev loans with any credit card DPD (needed by prev_app DEFAULTED)
    cc_target1 = ccbl[ccbl.SK_DPD > 0]["SK_ID_PREV"].unique()
    ccbl_mon = engineer_credit_card_features(ccbl)
    del ccbl; gc.collect()
    logger.info(f"  credit_card: {ccbl_mon.shape}")

    logger.info("Loading POS_CASH_balance...")
    pos = pd.read_csv(data_dir / "POS_CASH_balance.csv",
                      dtype={"SK_ID_CURR": "int32", "SK_ID_PREV": "int32"})
    # pos_target1: prev loans with any POS DPD
    pos_target1 = pos[pos.SK_DPD > 0]["SK_ID_PREV"].unique()
    pos_feat, pos_prev_last = engineer_pos_cash_features(pos)
    del pos; gc.collect()
    logger.info(f"  pos_cash: {pos_feat.shape}")

    logger.info("Loading installments_payments...")
    inst = pd.read_csv(data_dir / "installments_payments.csv",
                       dtype={"SK_ID_CURR": "int32", "SK_ID_PREV": "int32"})
    # inst_target1: prev loans with late or underpayment
    inst_target1 = inst.loc[
        (inst["DAYS_ENTRY_PAYMENT"] > inst["DAYS_INSTALMENT"] + 1) |
        (inst["AMT_PAYMENT"] < inst["AMT_INSTALMENT"])
    ]["SK_ID_PREV"].unique()
    # inst_prev_last: total payment per SK_ID_PREV for prev_app active AMT_PAID
    inst_prev_last = inst.groupby("SK_ID_PREV")["AMT_PAYMENT"].sum()
    inst_feat = engineer_installment_features(inst)
    del inst; gc.collect()
    logger.info(f"  installments: {inst_feat.shape}")

    logger.info("Loading previous_application...")
    prev = pd.read_csv(data_dir / "previous_application.csv",
                       dtype={"SK_ID_CURR": "int32", "SK_ID_PREV": "int32"})
    prev_feat = engineer_prev_application_features(
        prev,
        inst_prev_last  = inst_prev_last,
        pos_prev_last   = pos_prev_last,
        inst_target1    = inst_target1,
        pos_target1     = pos_target1,
        cc_target1      = cc_target1,
        app_credit_annuity = app_credit_annuity,
    )
    del prev; gc.collect()
    logger.info(f"  prev_app: {prev_feat.shape}")

    # ── Merge all onto application ──
    logger.info("Merging all tables...")
    app.set_index("SK_ID_CURR", inplace=True)
    for feat_df, name in [
        (avg_buro,  "bureau"),
        (ccbl_mon,  "credit_card"),
        (pos_feat,  "pos_cash"),
        (inst_feat, "installments"),
        (prev_feat, "prev_app"),
    ]:
        app = app.merge(feat_df, how="left", on="SK_ID_CURR")
        del feat_df
        gc.collect()
        logger.info(f"  after merging {name}: {app.shape}")

    # ── Cross-table ratio features (lgb1.ipynb cell 20) ──
    # Total annuity across current + bureau active + prev active
    app["Total_AMT_ANNUITY"] = app[[
        "AMT_ANNUITY",
        "bureau_active_sum_AMT_ANNUITY",
        "prev_active_sum_AMT_ANNUITY"
    ]].sum(axis=1)
    app["Total_ANNUITY_INCOME_RATIO"] = app["Total_AMT_ANNUITY"] / app["AMT_INCOME_TOTAL"]

    # Total credit (exclude already paid)
    app["Total_CREDIT"] = app[["AMT_CREDIT", "prev_active_sum_AMT_LEFT"]].sum(axis=1)
    app["Total_CREDIT_INCOME_RATIO"] = app["Total_CREDIT"] / app["AMT_INCOME_TOTAL"]

    # Total account counts
    app["Total_acc"] = app[["prev_count", "bureau_count"]].sum(axis=1)
    app["Total_active_acc"] = app[["prev_active_count", "bureau_active_count"]].sum(axis=1)

    # Total amount left
    app["Total_AMT_LEFT"] = (
        app["AMT_CREDIT"]
        + app["prev_active_sum_AMT_LEFT"]
        + app["bureau_active_sum_AMT_CREDIT_LEFT"]
    )
    app["Total_AMT_LEFT_INCOME_RATIO"] = app["Total_AMT_LEFT"] / app["AMT_INCOME_TOTAL"]

    # Current application vs previous approved/refused
    shared_feats = ["AMT_ANNUITY", "AMT_CREDIT", "AMT_PAY_YEAR",
                    "AMT_DIFF_CREDIT_GOODS", "AMT_CREDIT_GOODS_PERC"]
    for f_ in shared_feats:
        prev_app_mean = "prev_approved_" + f_ + "_MEAN"
        prev_ref_mean = "prev_refused_" + f_ + "_MEAN"
        if prev_app_mean in app.columns:
            app[f_ + "_to_prev_approved"] = (
                (app[f_] - app[prev_app_mean]) / app[prev_app_mean]
            )
        if prev_ref_mean in app.columns:
            app[f_ + "_to_prev_refused"] = (
                (app[f_] - app[prev_ref_mean]) / app[prev_ref_mean]
            )

    logger.info(f"  after cross-table ratios: {app.shape}")

    # Drop TARGET column from X
    if "TARGET" in app.columns:
        del app["TARGET"]

    # Encode any remaining object dtype columns (e.g. _mode aggregations)
    # Must be done BEFORE fillna so -999 doesn't become mixed-type
    for col in app.select_dtypes(include="object").columns:
        app[col], _ = pd.factorize(app[col])

    # Final cleanup
    app.replace([np.inf, -np.inf], np.nan, inplace=True)
    app.fillna(-999, inplace=True)
    app = downcast_dtypes(app)


    # Sanitize column names: LightGBM >= 4.x rejects special JSON chars
    import re
    old_to_new = {
        c: re.sub(r"[^A-Za-z0-9_]", "_", c).strip("_")
        for c in app.columns
    }
    # Deduplicate after sanitization
    seen: dict[str, int] = {}
    dedup: dict[str, str] = {}
    for old, new in old_to_new.items():
        if new in seen:
            seen[new] += 1
            dedup[old] = f"{new}_{seen[new]}"
        else:
            seen[new] = 0
            dedup[old] = new
    app.rename(columns=dedup, inplace=True)

    # Remap meanenc_feats and cat_feats to sanitized names
    meanenc_feats = [dedup.get(f, f) for f in meanenc_feats if dedup.get(f, f) in app.columns]
    cat_feats     = [dedup.get(f, f) for f in cat_feats     if dedup.get(f, f) in app.columns]

    logger.info(f"Final feature matrix: {app.shape}")
    return app, y, meanenc_feats, cat_feats
