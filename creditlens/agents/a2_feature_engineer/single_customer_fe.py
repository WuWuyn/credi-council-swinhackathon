"""
Single-Customer Feature Engineering — builds the full 753-feature vector
for one customer using pre-computed training statistics.

This is the bridge between A1 output (raw data) and A3 input (model features).
Reuses all feature engineering logic from training/feature_engineering.py
but handles batch-dependent operations using saved statistics.
"""

from __future__ import annotations

import gc
import logging
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root for training imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.feature_engineering import (
    REJECTED_APP_FEATURES,
    downcast_dtypes,
    engineer_bureau_balance_features,
    engineer_bureau_features,
    engineer_credit_card_features,
    engineer_installment_features,
    engineer_pos_cash_features,
    engineer_prev_application_features,
)

logger = logging.getLogger(__name__)


class SingleCustomerFE:
    """Feature engineering for a single customer.

    Loads pre-computed statistics from training data and uses them
    to replicate the full feature engineering pipeline for one customer.

    Usage:
        fe = SingleCustomerFE("models/fe_stats.pkl")
        feature_vector = fe.build_features(a1_output)
        # feature_vector is a pd.Series with 753 features
    """

    def __init__(self, stats_path: str = "models/fe_stats.pkl",
                 model_path: str = "models/lgbm_ref_v1.pkl"):
        stats_file = Path(stats_path)
        if not stats_file.exists():
            raise FileNotFoundError(
                f"FE stats not found: {stats_path}. "
                f"Run: python training/precompute_fe_stats.py --data-dir home-credit-default-risk/"
            )
        with open(stats_file, "rb") as f:
            self.stats = pickle.load(f)

        # Use model's feature names (753) for final alignment
        model_file = Path(model_path)
        if model_file.exists():
            try:
                from creditlens.agents.a3_scoring.model import CreditLensModel
                model = CreditLensModel()
                model.load(model_path)
                self.feature_names = model.feature_names
                logger.info(f"SingleCustomerFE: using model's {len(self.feature_names)} features")
            except Exception as e:
                logger.warning(f"Could not load model features: {e}, using FE stats")
                self.feature_names = self.stats["feature_names"]
        else:
            self.feature_names = self.stats["feature_names"]
            logger.info(f"SingleCustomerFE: using FE stats' {len(self.feature_names)} features")

    def build_features(self, a1_output: dict[str, Any]) -> pd.Series:
        """Build full feature vector from A1 output.

        Args:
            a1_output: Output from A1 IngestionAgent.ingest()

        Returns:
            pd.Series with 753 features matching model expectation.
        """
        app_row = a1_output["application_row"]
        sk_id = app_row.get("SK_ID_CURR", 100002)

        # ── 1. Application features ──
        app_df = pd.DataFrame([app_row])
        # Coerce None → NaN for numeric columns (prevents operator.neg(None) crash)
        # Known categorical columns that should NOT be coerced to numeric
        CATEGORICAL_COLS = {
            "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
            "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
            "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE", "HOUSETYPE_MODE",
            "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
            "WEEKDAY_APPR_PROCESS_START",
        }
        for col in app_df.columns:
            if app_df[col].dtype == object and col not in CATEGORICAL_COLS:
                app_df[col] = pd.to_numeric(app_df[col], errors='coerce')
        app_df = app_df.fillna(value=np.nan)
        app_df = self._engineer_application_features(app_df)
        logger.info(f"  App features: {app_df.shape[1]} cols")

        # Save credit/annuity for prev_app cross-reference
        app_credit_annuity = pd.DataFrame({
            "AMT_CREDIT": [app_row.get("AMT_CREDIT", 0)],
            "AMT_ANNUITY": [app_row.get("AMT_ANNUITY", 0)],
        }, index=[sk_id])
        app_credit_annuity.index.name = "SK_ID_CURR"

        # ── 2. Bureau features ──
        bureau_df = a1_output.get("bureau_df", pd.DataFrame())
        bb_df = a1_output.get("bureau_balance_df", pd.DataFrame())

        if not bureau_df.empty and not bb_df.empty:
            bureau_df["SK_ID_CURR"] = sk_id
            try:
                STATUS_TCNT, STATUS_12CNT, bubl_last_DPD, bubl_last_C = \
                    engineer_bureau_balance_features(bb_df)
                bureau_feat = engineer_bureau_features(
                    bureau_df, STATUS_TCNT, STATUS_12CNT, bubl_last_DPD, bubl_last_C
                )
                logger.info(f"  Bureau features: {bureau_feat.shape[1]} cols")
            except Exception as e:
                logger.warning(f"  Bureau features failed: {e}")
                bureau_feat = pd.DataFrame(index=[sk_id])
        else:
            bureau_feat = pd.DataFrame(index=[sk_id])
        bureau_feat.index.name = "SK_ID_CURR"

        # ── 3. Credit card features ──
        cc_df = a1_output.get("credit_card_df", pd.DataFrame())
        cc_target1 = np.array([])  # prev loans with credit card DPD
        if not cc_df.empty:
            cc_df["SK_ID_CURR"] = sk_id
            cc_target1 = cc_df[cc_df.get("SK_DPD", pd.Series(dtype=int)) > 0]["SK_ID_PREV"].unique() \
                if "SK_DPD" in cc_df.columns else np.array([])
            try:
                cc_feat = engineer_credit_card_features(cc_df)
                logger.info(f"  Credit card features: {cc_feat.shape[1]} cols")
            except Exception as e:
                logger.warning(f"  Credit card features failed: {e}")
                cc_feat = pd.DataFrame(index=[sk_id])
        else:
            cc_feat = pd.DataFrame(index=[sk_id])
        cc_feat.index.name = "SK_ID_CURR"

        # ── 4. POS cash features ──
        pos_df = a1_output.get("pos_cash_df", pd.DataFrame())
        pos_target1 = np.array([])
        pos_prev_last = pd.DataFrame()
        if not pos_df.empty:
            pos_df["SK_ID_CURR"] = sk_id
            pos_target1 = pos_df[pos_df.get("SK_DPD", pd.Series(dtype=int)) > 0]["SK_ID_PREV"].unique() \
                if "SK_DPD" in pos_df.columns else np.array([])
            try:
                pos_result = engineer_pos_cash_features(pos_df)
                if isinstance(pos_result, tuple):
                    pos_feat, pos_prev_last = pos_result
                else:
                    pos_feat = pos_result
                logger.info(f"  POS cash features: {pos_feat.shape[1]} cols")
            except Exception as e:
                logger.warning(f"  POS cash features failed: {e}")
                pos_feat = pd.DataFrame(index=[sk_id])
        else:
            pos_feat = pd.DataFrame(index=[sk_id])
        pos_feat.index.name = "SK_ID_CURR"

        # ── 5. Installment features ──
        inst_df = a1_output.get("installments_df", pd.DataFrame())
        inst_target1 = np.array([])
        inst_prev_last = pd.Series(dtype=float)
        if not inst_df.empty:
            inst_df["SK_ID_CURR"] = sk_id
            try:
                inst_target1 = inst_df.loc[
                    (inst_df["DAYS_ENTRY_PAYMENT"] > inst_df["DAYS_INSTALMENT"] + 1) |
                    (inst_df["AMT_PAYMENT"] < inst_df["AMT_INSTALMENT"])
                ]["SK_ID_PREV"].unique() if all(c in inst_df.columns for c in
                    ["DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT", "AMT_PAYMENT", "AMT_INSTALMENT"]) \
                    else np.array([])
                inst_prev_last = inst_df.groupby("SK_ID_PREV")["AMT_PAYMENT"].sum()
                inst_feat = engineer_installment_features(inst_df)
                logger.info(f"  Installment features: {inst_feat.shape[1]} cols")
            except Exception as e:
                logger.warning(f"  Installment features failed: {e}")
                inst_feat = pd.DataFrame(index=[sk_id])
        else:
            inst_feat = pd.DataFrame(index=[sk_id])
        inst_feat.index.name = "SK_ID_CURR"

        # ── 6. Previous application features ──
        prev_df = a1_output.get("previous_application_df", pd.DataFrame())
        if not prev_df.empty:
            prev_df["SK_ID_CURR"] = sk_id
            try:
                # Ensure pos_prev_last is valid
                if pos_prev_last.empty:
                    pos_prev_last = pd.DataFrame(
                        {"CNT_INSTALMENT": [0], "CNT_INSTALMENT_FUTURE": [0], "INSTAL_LEFT_RATIO": [0]},
                        index=prev_df["SK_ID_PREV"].unique()[:1]
                    )
                    pos_prev_last.index.name = "SK_ID_PREV"

                prev_feat = engineer_prev_application_features(
                    prev_df, inst_prev_last, pos_prev_last,
                    inst_target1, pos_target1, cc_target1,
                    app_credit_annuity,
                )
                logger.info(f"  Prev app features: {prev_feat.shape[1]} cols")
            except Exception as e:
                logger.warning(f"  Prev app features failed: {e}")
                prev_feat = pd.DataFrame(index=[sk_id])
        else:
            prev_feat = pd.DataFrame(index=[sk_id])
        prev_feat.index.name = "SK_ID_CURR"

        # ── 7. Merge all ──
        app_df.set_index("SK_ID_CURR", inplace=True)
        for feat_df, name in [
            (bureau_feat, "bureau"),
            (cc_feat, "credit_card"),
            (pos_feat, "pos_cash"),
            (inst_feat, "installments"),
            (prev_feat, "prev_app"),
        ]:
            app_df = app_df.merge(feat_df, how="left", on="SK_ID_CURR")
            logger.info(f"  After {name}: {app_df.shape[1]} cols")

        # ── 7.5 Ensure all model features exist ──
        # Must run BEFORE cross-table ratios so prev_approved_*/prev_refused_*
        # columns exist for ratio computation.
        existing_raw = set(app_df.columns)
        existing_sanitized = set()
        for c in app_df.columns:
            sanitized = re.sub(r"[^A-Za-z0-9_]", "_", c).strip("_")
            existing_sanitized.add(sanitized)
        all_existing = existing_raw | existing_sanitized

        n_added = 0
        for feat in self.feature_names:
            if feat in all_existing:
                continue
            n_added += 1
            if any(feat.startswith(p) for p in [
                "bureau_sum_CREDIT_ACTIVE_", "bureau_sum_CREDIT_CURRENCY_",
                "bureau_sum_CREDIT_TYPE_", "bureau_sum_STATUS_",
                "bureau_active_", "bureau_used_other_currency",
                "prev_sum_NAME_", "prev_sum_CODE_", "prev_sum_CHANNEL_",
                "prev_sum_NAME_CONTRACT_", "prev_sum_NAME_TYPE_SUITE_",
                "prev_sum_NAME_YIELD_GROUP_", "prev_sum_NAME_PORTFOLIO_",
                "prev_sum_NAME_PRODUCT_TYPE_", "prev_sum_NAME_PAYMENT_TYPE_",
                "prev_sum_NAME_CLIENT_TYPE_",
                "prev_approved_", "prev_refused_",
                "cc_Approved", "cc_Completed", "cc_Demand",
                "cc_Refused", "cc_Sent_proposal", "cc_Signed",
                "pos_NAME_CONTRACT_STATUS_CNT_",
            ]):
                app_df[feat] = 0
            else:
                app_df[feat] = np.nan
        if n_added:
            logger.info(f"  Injected {n_added} missing model feature columns")

        # ── 8. Cross-table ratio features ──
        app_df = self._add_cross_table_ratios(app_df)

        # ── 9. Cleanup: encode remaining objects, fill NaN, sanitize ──
        if "TARGET" in app_df.columns:
            del app_df["TARGET"]

        # Remove duplicate columns (keep first occurrence)
        # Merges can create duplicates, e.g. AMT_DIFF_CREDIT_GOODS from both
        # application FE and prev_app FE. Without dedup, sanitization adds _1
        # suffix causing 71+ feature alignment misses.
        if app_df.columns.duplicated().any():
            app_df = app_df.loc[:, ~app_df.columns.duplicated(keep='first')]

        for col in app_df.select_dtypes(include="object").columns:
            fmap = self.stats["factorize_maps"].get(col)
            if fmap:
                app_df[col] = app_df[col].map(fmap).fillna(-1).astype(int)
            else:
                app_df[col], _ = pd.factorize(app_df[col])

        app_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        app_df.fillna(-999, inplace=True)

        # Sanitize column names (same as training)
        old_to_new = {
            c: re.sub(r"[^A-Za-z0-9_]", "_", c).strip("_")
            for c in app_df.columns
        }
        seen: dict[str, int] = {}
        dedup: dict[str, str] = {}
        for old, new in old_to_new.items():
            if new in seen:
                seen[new] += 1
                dedup[old] = f"{new}_{seen[new]}"
            else:
                seen[new] = 0
                dedup[old] = new
        app_df.rename(columns=dedup, inplace=True)

        # ── 10. Align to model's feature names ──
        result = {}
        for feat in self.feature_names:
            if feat in app_df.columns:
                result[feat] = float(app_df[feat].iloc[0])
            else:
                result[feat] = -999.0  # Missing feature default

        n_matched = sum(1 for v in result.values() if v != -999.0)
        logger.info(f"  Final: {n_matched}/{len(self.feature_names)} features matched")

        return pd.Series(result)

    def _engineer_application_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Application features using pre-computed stats for batch-dependent ops."""
        # Clean
        df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", np.nan)
        df["NAME_FAMILY_STATUS"] = df["NAME_FAMILY_STATUS"].replace("Unknown", np.nan)
        df["ORGANIZATION_TYPE"] = df["ORGANIZATION_TYPE"].replace("XNA", np.nan)
        df.loc[df["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

        # Ratio features (same as training)
        docs = [f for f in df.columns if "FLAG_DOC" in f]
        live = [f for f in df.columns if ("FLAG_" in f) and ("FLAG_DOC" not in f) and ("_FLAG_" not in f)]
        live_num = [f for f in live if df[f].dtype != object]

        if docs:
            df["NEW_DOC_IND_KURT"] = df[docs].kurtosis(axis=1)
        else:
            df["NEW_DOC_IND_KURT"] = 0

        if live_num:
            df["NEW_LIVE_IND_SUM"] = df[live_num].sum(axis=1)
        else:
            df["NEW_LIVE_IND_SUM"] = 0

        df["NEW_INC_PER_CHLD"] = df["AMT_INCOME_TOTAL"] / (1 + df["CNT_CHILDREN"])

        # Use pre-computed inc_by_org
        inc_by_org = self.stats.get("inc_by_org", {})
        df["NEW_INC_BY_ORG"] = df["ORGANIZATION_TYPE"].map(inc_by_org)

        df["NEW_EMPLOY_TO_BIRTH_RATIO"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
        df["NEW_SOURCES_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
        df["NEW_EXT_SOURCES_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
        df["NEW_SCORES_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
        df["NEW_SCORES_STD"] = df["NEW_SCORES_STD"].fillna(
            self.stats.get("global_scores_std_mean", 0.15)
        )
        df["NEW_CAR_TO_BIRTH_RATIO"] = df.get("OWN_CAR_AGE", pd.Series([np.nan])) / df["DAYS_BIRTH"]
        df["NEW_CAR_TO_EMPLOY_RATIO"] = df.get("OWN_CAR_AGE", pd.Series([np.nan])) / df["DAYS_EMPLOYED"]
        df["NEW_PHONE_TO_BIRTH_RATIO"] = df.get("DAYS_LAST_PHONE_CHANGE", pd.Series([0])) / df["DAYS_BIRTH"]
        df["NEW_PHONE_TO_EMPLOYED_RATIO"] = df.get("DAYS_LAST_PHONE_CHANGE", pd.Series([0])) / df["DAYS_EMPLOYED"]
        df["NEW_CREDIT_TO_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
        df["AMT_PAY_YEAR"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
        df["AGE_PAYOFF"] = -df["DAYS_BIRTH"] / 365.25 + df["AMT_PAY_YEAR"]
        df["AMT_ANNUITY_INCOME_RATE"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
        df["AMT_DIFF_CREDIT_GOODS"] = df["AMT_CREDIT"] - df.get("AMT_GOODS_PRICE", pd.Series([0]))
        df["AMT_CREDIT_GOODS_PERC"] = df["AMT_CREDIT"] / df.get("AMT_GOODS_PRICE", pd.Series([1]))
        df["DOCUMENT_CNT"] = df.loc[:, df.columns.str.startswith("FLAG_DOCUMENT")].sum(axis=1) \
            if any(c.startswith("FLAG_DOCUMENT") for c in df.columns) else 0
        df["AGE_EMPLOYED"] = df["DAYS_EMPLOYED"] - df["DAYS_BIRTH"]
        df["AMT_INCOME_OVER_CHILD"] = df["AMT_INCOME_TOTAL"] / df["CNT_CHILDREN"].replace(0, np.nan)
        df["CNT_ADULT"] = df.get("CNT_FAM_MEMBERS", pd.Series([1])) - df["CNT_CHILDREN"]
        df["ADULT_RATIO"] = df["CNT_ADULT"] / df.get("CNT_FAM_MEMBERS", pd.Series([1]))
        df["AMT_REQ_CREDIT_BUREAU_MON_CHANGE"] = (
            df.get("AMT_REQ_CREDIT_BUREAU_QRT", pd.Series([0])) / 2 -
            df.get("AMT_REQ_CREDIT_BUREAU_MON", pd.Series([0]))
        )
        df["AMT_REQ_CREDIT_BUREAU_QRT_CHANGE"] = (
            df.get("AMT_REQ_CREDIT_BUREAU_YEAR", pd.Series([0])) / 3 -
            df.get("AMT_REQ_CREDIT_BUREAU_QRT", pd.Series([0]))
        )

        # Region / income mean features (using pre-computed medians)
        df["CNT_CHILDREN_CLIPPED"] = df["CNT_CHILDREN"].clip(0, 10)
        df["REGION"] = 0  # factorized to 0 for single customer

        df["GENDER_FAMILY_STATUS"] = df["CODE_GENDER"].astype(str) + df["NAME_FAMILY_STATUS"].astype(str)

        group_medians = self.stats.get("group_medians", {})
        for grp_col, col_name in [
            ("CODE_GENDER",          "gender_mean_income"),
            ("FLAG_OWN_CAR",         "own_car_mean_income"),
            ("FLAG_OWN_REALTY",      "own_realty_mean_income"),
            ("CNT_CHILDREN_CLIPPED", "cnt_children_mean_income"),
            ("REGION",               "region_mean_income"),
            ("NAME_FAMILY_STATUS",   "family_status_mean_income"),
            ("GENDER_FAMILY_STATUS", "gender_family_status_mean_income"),
        ]:
            median_map = group_medians.get(col_name, {})
            if median_map and grp_col in df.columns:
                val = df[grp_col].iloc[0]
                df[col_name] = median_map.get(val, median_map.get(str(val), 0))
            else:
                df[col_name] = 0
            income = df["AMT_INCOME_TOTAL"].iloc[0]
            med_val = df[col_name].iloc[0] if df[col_name].iloc[0] != 0 else 1
            df[col_name + "_rel"] = (income - med_val) / med_val

        # Drop rejected features
        for f in REJECTED_APP_FEATURES:
            if f in df.columns:
                del df[f]

        # Factorize categoricals using pre-computed maps
        factorize_maps = self.stats.get("factorize_maps", {})
        for col in df.select_dtypes(include="object").columns:
            fmap = factorize_maps.get(col, {})
            if fmap:
                df[col] = df[col].map(fmap).fillna(-1).astype(int)
            else:
                df[col], _ = pd.factorize(df[col])

        # Add mean_encode features (model expects *_mean_encode columns)
        # These are created during training's mean_encode() function
        mean_encode_maps = self.stats.get("mean_encode_maps", {})
        meanenc_feats = self.stats.get("meanenc_feats", [])
        global_target_mean = self.stats.get("global_target_mean", 0.08)

        for feat in meanenc_feats:
            col_name = feat + "_mean_encode"
            if feat in df.columns and feat in mean_encode_maps:
                val = df[feat].iloc[0]
                mean_map = mean_encode_maps[feat]
                df[col_name] = mean_map.get(val, mean_map.get(int(val) if not pd.isna(val) else -1, global_target_mean))
            else:
                df[col_name] = global_target_mean

        return df

    def _add_cross_table_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cross-table ratio features (same as training)."""
        # Safe sum: only sum columns that exist
        def safe_sum(cols):
            existing = [c for c in cols if c in df.columns]
            return df[existing].sum(axis=1) if existing else 0

        df["Total_AMT_ANNUITY"] = safe_sum([
            "AMT_ANNUITY", "bureau_active_sum_AMT_ANNUITY", "prev_active_sum_AMT_ANNUITY"
        ])
        df["Total_ANNUITY_INCOME_RATIO"] = df["Total_AMT_ANNUITY"] / df.get("AMT_INCOME_TOTAL", 1)

        df["Total_CREDIT"] = safe_sum(["AMT_CREDIT", "prev_active_sum_AMT_LEFT"])
        df["Total_CREDIT_INCOME_RATIO"] = df["Total_CREDIT"] / df.get("AMT_INCOME_TOTAL", 1)

        df["Total_acc"] = safe_sum(["prev_count", "bureau_count"])
        df["Total_active_acc"] = safe_sum(["prev_active_count", "bureau_active_count"])

        df["Total_AMT_LEFT"] = safe_sum([
            "AMT_CREDIT", "prev_active_sum_AMT_LEFT", "bureau_active_sum_AMT_CREDIT_LEFT"
        ])
        df["Total_AMT_LEFT_INCOME_RATIO"] = df["Total_AMT_LEFT"] / df.get("AMT_INCOME_TOTAL", 1)

        # Current vs previous approved/refused
        shared_feats = ["AMT_ANNUITY", "AMT_CREDIT", "AMT_PAY_YEAR",
                        "AMT_DIFF_CREDIT_GOODS", "AMT_CREDIT_GOODS_PERC"]
        for f_ in shared_feats:
            prev_app_mean = "prev_approved_" + f_ + "_MEAN"
            prev_ref_mean = "prev_refused_" + f_ + "_MEAN"
            if prev_app_mean in df.columns and f_ in df.columns:
                df[f_ + "_to_prev_approved"] = (
                    (df[f_] - df[prev_app_mean]) / (df[prev_app_mean] + 0.001)
                )
            if prev_ref_mean in df.columns and f_ in df.columns:
                df[f_ + "_to_prev_refused"] = (
                    (df[f_] - df[prev_ref_mean]) / (df[prev_ref_mean] + 0.001)
                )

        return df
