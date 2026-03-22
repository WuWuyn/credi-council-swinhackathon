"""
CreditLens A1 — Data Ingestion Agent (Local Version).

# LOCAL_SUB: Uses PyMuPDF instead of AWS Textract, mock JSON instead of real APIs.
# See LOCAL_SUBSTITUTIONS.md for migration guide.

Orchestrates 4-channel data ingestion:
    1. PDF Documents → LocalDocumentParser → identity/employment/housing fields
    2. CIC API (mock JSON) → CICService → bureau records + ext scores
    3. Bank Statement CSV → parse_bank_statement → alt data features
    4. Internal DB (mock JSON) → InternalDBReader → previous loan DataFrames

Outputs:
    - application_row: dict matching application_train columns (for feature engineering)
    - bureau_df: DataFrame matching bureau.csv
    - bureau_balance_df: DataFrame matching bureau_balance.csv
    - previous_application_df: DataFrame matching previous_application.csv
    - pos_cash_df: DataFrame matching POS_CASH_balance.csv
    - installments_df: DataFrame matching installments_payments.csv
    - credit_card_df: DataFrame matching credit_card_balance.csv
    - confidence_map: confidence per extracted field
    - raw_texts: raw OCR text per document
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from creditlens.agents.a1_ingestion.document_parser import LocalDocumentParser
from creditlens.agents.a1_ingestion.cic_service import CICService
from creditlens.agents.a1_ingestion.bank_statement_parser import parse_bank_statement
from creditlens.agents.a1_ingestion.internal_db_reader import InternalDBReader

logger = logging.getLogger(__name__)


class IngestionAgent:
    """Agent A1 — Data Ingestion & Feature Pipeline (Local Version).

    Takes a customer's document folder and produces structured data
    matching the Home Credit dataset format for feature engineering.
    """

    def __init__(self, use_mock: bool = True):
        self.doc_parser = LocalDocumentParser()
        self.cic = CICService(use_mock=use_mock)
        self.internal_db = InternalDBReader()

    def ingest(
        self,
        customer_dir: str | Path,
        applicant_id: str | None = None,
    ) -> dict[str, Any]:
        """Run full ingestion pipeline from a customer directory.

        Expected directory structure:
            customer_dir/
                01_cccd.pdf
                02_hop_dong_lao_dong.pdf
                03_so_ho_khau.pdf
                04_tham_dinh_nha_o.pdf
                05_don_vay.pdf
                06_sao_ke_ngan_hang.csv
                07_cic_api_response.json
                08_internal_db.json

        Args:
            customer_dir: Path to customer data directory.
            applicant_id: Optional applicant ID (auto-generated if None).

        Returns:
            Dict with application_row, DataFrames, confidence_map, audit_trail.
        """
        customer_dir = Path(customer_dir)
        if not customer_dir.exists():
            raise FileNotFoundError(f"Customer directory not found: {customer_dir}")

        if applicant_id is None:
            applicant_id = hashlib.sha256(
                f"{customer_dir.name}_{datetime.now().isoformat()}".encode()
            ).hexdigest()[:16]

        logger.info(f"{'='*60}")
        logger.info(f"  A1 Ingestion — Customer: {customer_dir.name}")
        logger.info(f"  Applicant ID: {applicant_id}")
        logger.info(f"{'='*60}")

        # ── Channel 1: Parse PDF documents ─────────────────────────────
        doc_fields = {}
        confidence_map = {}
        raw_texts = {}
        identity_names = []

        pdf_files = sorted(customer_dir.glob("*.pdf"))
        for pdf_path in pdf_files:
            logger.info(f"  Parsing: {pdf_path.name}")
            result = self.doc_parser.extract_document(pdf_path)
            doc_type = result.get("doc_type", "unknown")
            fields = result.get("fields", {})
            conf = result.get("confidence", {})
            raw_texts[doc_type] = result.get("raw_text", "")

            doc_fields[doc_type] = fields
            confidence_map.update({f"{doc_type}.{k}": v for k, v in conf.items()})

            # Collect names for cross-validation
            name = fields.get("full_name", "")
            if name:
                identity_names.append(name.strip().upper())

        # Cross-document identity validation
        if len(identity_names) >= 2:
            identity_flag = "OK" if len(set(identity_names)) == 1 else "MISMATCH"
        elif len(identity_names) == 1:
            identity_flag = "OK"
        else:
            identity_flag = "MISSING"

        logger.info(f"  Identity consistency: {identity_flag}")

        # ── Channel 2: CIC API ─────────────────────────────────────────
        cic_path = customer_dir / "07_cic_api_response.json"
        cic_result = self.cic.query(cic_path if cic_path.exists() else None)
        logger.info(f"  CIC: thin_file={cic_result.get('thin_file_flag')}, "
                     f"bureau_records={len(cic_result.get('bureau_records', []))}")

        # ── Channel 3: Bank Statement ──────────────────────────────────
        bank_path = customer_dir / "06_sao_ke_ngan_hang.csv"
        bank_features = {}
        if bank_path.exists():
            try:
                bank_result = parse_bank_statement(bank_path)
                bank_features = bank_result.get("features", {})
                logger.info(f"  Bank statement: {len(bank_features)} features extracted")
            except Exception as e:
                logger.warning(f"  Bank statement error: {e}")

        # ── Channel 4: Internal DB ─────────────────────────────────────
        internal_path = customer_dir / "08_internal_db.json"
        internal_dfs = self.internal_db.read(internal_path if internal_path.exists() else None)

        # ── Build application_row (matching application_train columns) ──
        application_row = self._build_application_row(
            doc_fields, cic_result, bank_features, applicant_id
        )

        # ── Build bureau DataFrames ────────────────────────────────────
        bureau_df, bureau_balance_df = self._build_bureau_dfs(cic_result)

        # ── Audit trail ────────────────────────────────────────────────
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "agent": "A1",
            "action": "ingest",
            "input_summary": {
                "n_pdfs": len(pdf_files),
                "has_cic": not cic_result.get("thin_file_flag", True),
                "has_bank_statement": bank_path.exists(),
                "has_internal_db": internal_path.exists(),
            },
            "output_summary": {
                "n_application_fields": len([v for v in application_row.values() if v is not None]),
                "n_bureau_records": len(bureau_df),
                "identity_consistency": identity_flag,
            },
        }

        logger.info(f"  Application row: {len([v for v in application_row.values() if v is not None])} non-null fields")
        logger.info(f"  Bureau records: {len(bureau_df)}")

        return {
            "application_id": applicant_id,
            "application_row": application_row,
            "bureau_df": bureau_df,
            "bureau_balance_df": bureau_balance_df,
            "previous_application_df": internal_dfs.get("previous_application", pd.DataFrame()),
            "pos_cash_df": internal_dfs.get("POS_CASH_balance", pd.DataFrame()),
            "installments_df": internal_dfs.get("installments_payments", pd.DataFrame()),
            "credit_card_df": internal_dfs.get("credit_card_balance", pd.DataFrame()),
            "bank_features": bank_features,
            "confidence_map": confidence_map,
            "identity_consistency_flag": identity_flag,
            "thin_file_flag": cic_result.get("thin_file_flag", True),
            "raw_texts": raw_texts,
            "audit_trail": [audit_entry],
        }

    def _build_application_row(
        self,
        doc_fields: dict[str, dict],
        cic_result: dict[str, Any],
        bank_features: dict[str, Any],
        applicant_id: str,
    ) -> dict[str, Any]:
        """Build a dict matching application_train columns from parsed data.

        Maps extracted document fields → Home Credit dataset column names.
        """
        cccd = doc_fields.get("cccd", {})
        employment = doc_fields.get("employment", {})
        household = doc_fields.get("household", {})
        housing = doc_fields.get("housing", {})
        loan_app = doc_fields.get("loan_application", {})

        today = date.today()

        # Helper: days from date string to today (negative = past)
        def days_from_today(date_str: str | None) -> int | None:
            if not date_str:
                return None
            try:
                d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
                return (d - today).days
            except (ValueError, TypeError):
                return None

        # ── Build application_train row ──
        row = {
            "SK_ID_CURR": hash(applicant_id) % 1000000 + 100000,

            # Loan details (from Đơn vay)
            "NAME_CONTRACT_TYPE": loan_app.get("contract_type", "Cash loans"),
            "AMT_CREDIT": loan_app.get("loan_amount"),
            "AMT_ANNUITY": loan_app.get("monthly_payment"),
            "AMT_GOODS_PRICE": loan_app.get("goods_price"),
            "NAME_TYPE_SUITE": loan_app.get("type_suite", "Unaccompanied"),

            # Identity (from CCCD)
            "CODE_GENDER": self._map_gender(cccd.get("gender")),
            "DAYS_BIRTH": days_from_today(cccd.get("date_of_birth")),
            "DAYS_ID_PUBLISH": days_from_today(cccd.get("id_issue_date")),
            "DAYS_REGISTRATION": days_from_today(cccd.get("registration_date")),

            # Family (from Sổ hộ khẩu)
            "CNT_CHILDREN": household.get("children_count", 0),
            "CNT_FAM_MEMBERS": household.get("family_members_count", 1),
            "NAME_FAMILY_STATUS": self._map_marital_status(
                household.get("marital_status") or loan_app.get("marital_status")
            ),

            # Employment (from HĐLĐ)
            "AMT_INCOME_TOTAL": employment.get("annual_income") or (
                employment.get("base_salary", 0) * 12 if employment.get("base_salary") else None
            ),
            "NAME_INCOME_TYPE": self._map_income_type(employment.get("contract_type")),
            "DAYS_EMPLOYED": days_from_today(employment.get("employment_start_date")),
            "ORGANIZATION_TYPE": self._map_org_type(employment.get("employer_name")),
            "OCCUPATION_TYPE": self._map_occupation(employment.get("position")),

            # Contact flags (from Đơn vay)
            "FLAG_MOBIL": 1,  # Always 1 for application
            "FLAG_EMP_PHONE": 1 if employment.get("employer_phone") else 0,
            "FLAG_WORK_PHONE": 1 if employment.get("employer_phone") else 0,
            "FLAG_CONT_MOBILE": loan_app.get("flag_cont_mobile", 1),
            "FLAG_PHONE": loan_app.get("flag_phone", 0),
            "FLAG_EMAIL": loan_app.get("flag_email", 1),

            # Assets
            "FLAG_OWN_CAR": "Y" if loan_app.get("has_car") else "N",
            "FLAG_OWN_REALTY": "Y" if loan_app.get("has_realty") else "N",
            "OWN_CAR_AGE": loan_app.get("car_age"),

            # Education
            "NAME_EDUCATION_TYPE": loan_app.get("education_type", "Higher education"),
            "NAME_HOUSING_TYPE": housing.get("housing_type", "House / apartment"),

            # Region info (from Housing)
            "REGION_POPULATION_RELATIVE": housing.get("region_population_relative", 0.035),
            "REGION_RATING_CLIENT": housing.get("region_rating", 2),
            "REGION_RATING_CLIENT_W_CITY": housing.get("region_rating_w_city", 2),

            # Address cross-checks
            "REG_REGION_NOT_LIVE_REGION": 0 if housing.get("reg_live_same_region") else 1,
            "REG_REGION_NOT_WORK_REGION": 0 if housing.get("reg_work_same_city") else 1,
            "LIVE_REGION_NOT_WORK_REGION": 0 if housing.get("live_work_same_region") else 1,
            "REG_CITY_NOT_LIVE_CITY": 0,
            "REG_CITY_NOT_WORK_CITY": 0,
            "LIVE_CITY_NOT_WORK_CITY": 0,

            # Application process (auto-captured)
            "WEEKDAY_APPR_PROCESS_START": today.strftime("%A").upper()[:3]
                                          if today.weekday() < 5 else "MONDAY",
            "HOUR_APPR_PROCESS_START": datetime.now().hour,

            # Phone change
            "DAYS_LAST_PHONE_CHANGE": self._parse_days_phone_change(
                loan_app.get("days_last_phone_change_info")
            ),

            # CIC scores (from CIC API)
            "EXT_SOURCE_1": cic_result.get("EXT_SOURCE_1"),
            "EXT_SOURCE_2": cic_result.get("EXT_SOURCE_2"),
            "EXT_SOURCE_3": cic_result.get("EXT_SOURCE_3"),

            # CIC inquiry counts
            "AMT_REQ_CREDIT_BUREAU_HOUR": cic_result.get("AMT_REQ_CREDIT_BUREAU_HOUR", 0),
            "AMT_REQ_CREDIT_BUREAU_DAY": cic_result.get("AMT_REQ_CREDIT_BUREAU_DAY", 0),
            "AMT_REQ_CREDIT_BUREAU_WEEK": cic_result.get("AMT_REQ_CREDIT_BUREAU_WEEK", 0),
            "AMT_REQ_CREDIT_BUREAU_MON": cic_result.get("AMT_REQ_CREDIT_BUREAU_MON", 0),
            "AMT_REQ_CREDIT_BUREAU_QRT": cic_result.get("AMT_REQ_CREDIT_BUREAU_QRT", 0),
            "AMT_REQ_CREDIT_BUREAU_YEAR": cic_result.get("AMT_REQ_CREDIT_BUREAU_YEAR", 0),

            # Social circle
            "OBS_30_CNT_SOCIAL_CIRCLE": cic_result.get("OBS_30_CNT_SOCIAL_CIRCLE", 0),
            "DEF_30_CNT_SOCIAL_CIRCLE": cic_result.get("DEF_30_CNT_SOCIAL_CIRCLE", 0),
            "OBS_60_CNT_SOCIAL_CIRCLE": cic_result.get("OBS_60_CNT_SOCIAL_CIRCLE", 0),
            "DEF_60_CNT_SOCIAL_CIRCLE": cic_result.get("DEF_60_CNT_SOCIAL_CIRCLE", 0),

            # Housing features (from thẩm định)
            **self._build_housing_features(housing),

            # Document flags — which docs were submitted
            **self._build_document_flags(doc_fields),
        }

        # ── Inject bank statement features ──────────────────────────────
        # Per document_new.md Section 1.1 Kênh 3: bank features map 1-1
        # to Home Credit equivalents. This ensures the ML model can
        # leverage transaction behavioral signals.
        if bank_features:
            row.update(self._inject_bank_features(bank_features, row, cic_result))

        return row

    def _inject_bank_features(
        self,
        bank_features: dict[str, Any],
        row: dict[str, Any],
        cic_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Map bank statement features → Home Credit equivalent signals.

        Design reference (document_new.md lines 42-53):
        - avg_monthly_inflow_vnd  → AMT_INCOME_TOTAL proxy
        - income_stability_index  → 1 - CV(AMT_INSTALMENT)
        - salary_pattern_detected → employment confirmation
        - overdraft_count_6m      → SK_DPD > 0 count proxy
        - debt_service_behavior   → payment status proxy
        - inflow_outflow_ratio    → income/expense health
        """
        injected = {}
        logger.info("  Injecting bank features into application_row...")

        # 1. Income validation: if bank shows real inflow, use it
        #    avg_monthly_inflow is more reliable than stated income for
        #    self-employed/freelancers
        avg_inflow = bank_features.get("avg_monthly_inflow_vnd", 0)
        if avg_inflow and avg_inflow > 0:
            stated_income = row.get("AMT_INCOME_TOTAL")
            if stated_income is None or stated_income == 0:
                # No stated income → Use bank inflow as annual income
                injected["AMT_INCOME_TOTAL"] = avg_inflow * 12
                logger.info(f"    AMT_INCOME_TOTAL set from bank: {avg_inflow * 12:,.0f}")
            elif abs(stated_income - avg_inflow * 12) / max(stated_income, 1) > 0.3:
                # Significant discrepancy → use lower of two (conservative)
                conservative = min(stated_income, avg_inflow * 12)
                injected["AMT_INCOME_TOTAL"] = conservative
                logger.info(
                    f"    AMT_INCOME_TOTAL adjusted: stated={stated_income:,.0f} "
                    f"vs bank={avg_inflow * 12:,.0f} → using {conservative:,.0f}"
                )

        # 2. Synthesize EXT_SOURCE scores from bank behavioral data
        #    EXT_SOURCE_1/2/3 are the most predictive features (SHAP top-3)
        #    For customers with weak/missing CIC, bank data can provide
        #    equivalent behavioral scoring.
        #
        #    This follows document.md lines 192-194:
        #    "Transaction Behavioral ★ Alt. Data → Engineered từ
        #     installments_payments.csv + credit_card_balance.csv"
        stability = bank_features.get("income_stability_index", 0.5)
        bill_ratio = bank_features.get("regular_bill_payment_ratio", 0.5)
        overdraft = bank_features.get("overdraft_count_6m", 0)
        io_ratio = bank_features.get("inflow_outflow_ratio", 1.0)
        salary = bank_features.get("salary_pattern_detected", False)
        debt_service = bank_features.get("debt_service_behavior", "MISSING")

        # Debt service score: ON_TIME → 1.0, LATE_1_30 → 0.4, LATE_31_60 → 0.1, MISSING → 0.5
        debt_score_map = {"ON_TIME": 1.0, "LATE_1_30": 0.4, "LATE_31_60": 0.1, "MISSING": 0.5}
        debt_score = debt_score_map.get(debt_service, 0.5)

        # Build composite bank behavioral scores (0-1 range, like EXT_SOURCE)
        # Score 1: Income reliability = stability × salary_bonus × io_health
        io_health = min(1.0, max(0, (io_ratio - 0.5) / 1.0))  # 0.5→0, 1.5→1
        bank_score_1 = stability * (1.1 if salary else 0.8) * io_health
        bank_score_1 = max(0, min(1.0, bank_score_1))

        # Score 2: Payment discipline = bill_payment × debt_service × overdraft_penalty
        overdraft_penalty = max(0, 1.0 - overdraft * 0.15)  # Each overdraft -15%
        bank_score_2 = bill_ratio * debt_score * overdraft_penalty
        bank_score_2 = max(0, min(1.0, bank_score_2))

        # Score 3: Overall financial health (blend)
        bank_score_3 = (bank_score_1 * 0.4 + bank_score_2 * 0.4 +
                        io_health * 0.2)
        bank_score_3 = max(0, min(1.0, bank_score_3))

        # Injection strategy:
        # - If EXT_SOURCE is already set (from CIC), blend with bank score
        # - If EXT_SOURCE is null (thin-file), use bank score directly
        thin_file = cic_result.get("thin_file_flag", False)

        for ext_key, bank_score in [
            ("EXT_SOURCE_1", bank_score_1),
            ("EXT_SOURCE_2", bank_score_2),
            ("EXT_SOURCE_3", bank_score_3),
        ]:
            existing = row.get(ext_key)
            if existing is None or thin_file:
                # Thin-file or missing: use bank score as sole signal
                injected[ext_key] = round(bank_score, 4)
                logger.info(f"    {ext_key}: bank_score={bank_score:.4f} (replaced null/thin-file)")
            else:
                # Has CIC: blend 70% CIC + 30% bank (CIC still primary)
                blended = existing * 0.7 + bank_score * 0.3
                injected[ext_key] = round(blended, 4)
                logger.info(
                    f"    {ext_key}: CIC={existing:.4f} + bank={bank_score:.4f} → blended={blended:.4f}"
                )

        logger.info(f"  Bank injection complete: {len(injected)} fields updated")
        return injected

    def _build_housing_features(self, housing: dict) -> dict[str, Any]:
        """Build normalized housing features from housing survey data.

        Maps housing survey fields to the 46 housing columns in application_train.
        """
        quality = housing.get("apartment_quality")
        if quality:
            # Normalize quality score from "7.5 / 10" to 0-1
            nums = [float(n) for n in str(quality).split("/")]
            quality_norm = nums[0] / nums[1] if len(nums) == 2 else nums[0] / 10
        else:
            quality_norm = None

        # Parse areas
        living_area = housing.get("living_area")
        if isinstance(living_area, str):
            nums = [float(n) for n in living_area.split() if n.replace(".", "").isdigit()]
            living_area = nums[0] if nums else None

        # Normalized housing values (Home Credit uses 0-1 normalized values)
        # We use the quality score as proxy for all quality metrics
        housing_feats = {}
        housing_norm_cols = [
            "APARTMENTS", "BASEMENTAREA", "YEARS_BEGINEXPLUATATION",
            "YEARS_BUILD", "COMMONAREA", "ELEVATORS", "ENTRANCES",
            "FLOORSMAX", "FLOORSMIN", "LANDAREA", "LIVINGAPARTMENTS",
            "LIVINGAREA", "NONLIVINGAPARTMENTS", "NONLIVINGAREA",
        ]

        for col in housing_norm_cols:
            for suffix in ["_AVG", "_MODE", "_MEDI"]:
                housing_feats[col + suffix] = quality_norm

        # Override specific values where we have actual data
        if housing.get("max_floors"):
            floor_norm = min(1.0, housing["max_floors"] / 50)
            housing_feats["FLOORSMAX_AVG"] = floor_norm
            housing_feats["FLOORSMAX_MODE"] = floor_norm
            housing_feats["FLOORSMAX_MEDI"] = floor_norm

        if living_area:
            area_norm = min(1.0, living_area / 200)
            housing_feats["LIVINGAREA_AVG"] = area_norm
            housing_feats["LIVINGAREA_MODE"] = area_norm
            housing_feats["LIVINGAREA_MEDI"] = area_norm

        if housing.get("year_built"):
            year_norm = min(1.0, max(0, (housing["year_built"] - 1950) / 80))
            housing_feats["YEARS_BUILD_AVG"] = year_norm
            housing_feats["YEARS_BUILD_MODE"] = year_norm
            housing_feats["YEARS_BUILD_MEDI"] = year_norm

        if housing.get("has_elevator"):
            elev = 1.0 if str(housing["has_elevator"]).lower() in ("co", "yes", "true", "1") else 0.0
            housing_feats["ELEVATORS_AVG"] = elev
            housing_feats["ELEVATORS_MODE"] = elev
            housing_feats["ELEVATORS_MEDI"] = elev

        # Categorical housing fields
        housing_feats["FONDKAPREMONT_MODE"] = housing.get("fond_kapremont", "reg oper account")
        housing_feats["HOUSETYPE_MODE"] = housing.get("housing_type", "block of flats")
        housing_feats["TOTALAREA_MODE"] = quality_norm if quality_norm else 0.0
        housing_feats["WALLSMATERIAL_MODE"] = housing.get("wall_material", "Panel")
        housing_feats["EMERGENCYSTATE_MODE"] = (
            "No" if "khong" in str(housing.get("emergency_state", "")).lower()
               or "binh thuong" in str(housing.get("emergency_state", "")).lower()
            else "Yes"
        )

        return housing_feats

    def _build_document_flags(self, doc_fields: dict) -> dict[str, int]:
        """Build FLAG_DOCUMENT_2 through FLAG_DOCUMENT_21."""
        flags = {}
        # Document submission flags — set based on which docs we received
        has_cccd = "cccd" in doc_fields
        has_employment = "employment" in doc_fields
        has_household = "household" in doc_fields
        has_housing = "housing" in doc_fields
        has_loan = "loan_application" in doc_fields

        # Home Credit FLAG_DOCUMENT_* flags:
        # 3 is most common (identity doc), 8 is second common
        flags["FLAG_DOCUMENT_2"] = 0
        flags["FLAG_DOCUMENT_3"] = 1 if has_cccd else 0
        flags["FLAG_DOCUMENT_4"] = 0
        flags["FLAG_DOCUMENT_5"] = 0
        flags["FLAG_DOCUMENT_6"] = 1 if has_employment else 0
        flags["FLAG_DOCUMENT_7"] = 0
        flags["FLAG_DOCUMENT_8"] = 1 if has_housing else 0
        flags["FLAG_DOCUMENT_9"] = 0
        flags["FLAG_DOCUMENT_10"] = 0
        flags["FLAG_DOCUMENT_11"] = 0
        flags["FLAG_DOCUMENT_12"] = 0
        flags["FLAG_DOCUMENT_13"] = 0
        flags["FLAG_DOCUMENT_14"] = 0
        flags["FLAG_DOCUMENT_15"] = 0
        flags["FLAG_DOCUMENT_16"] = 1 if has_household else 0
        flags["FLAG_DOCUMENT_17"] = 0
        flags["FLAG_DOCUMENT_18"] = 1 if has_loan else 0
        flags["FLAG_DOCUMENT_19"] = 0
        flags["FLAG_DOCUMENT_20"] = 0
        flags["FLAG_DOCUMENT_21"] = 0

        return flags

    def _build_bureau_dfs(self, cic_result: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Convert CIC bureau records to DataFrames matching Home Credit format."""
        bureau_records = cic_result.get("bureau_records", [])

        if not bureau_records:
            return pd.DataFrame(), pd.DataFrame()

        bureau_rows = []
        balance_rows = []

        for rec in bureau_records:
            bureau_rows.append({
                "SK_ID_CURR": None,  # Will be set later
                "SK_ID_BUREAU": rec.get("SK_ID_BUREAU"),
                "CREDIT_ACTIVE": rec.get("CREDIT_ACTIVE"),
                "CREDIT_CURRENCY": rec.get("CREDIT_CURRENCY", "currency 1"),
                "DAYS_CREDIT": rec.get("DAYS_CREDIT"),
                "CREDIT_DAY_OVERDUE": rec.get("CREDIT_DAY_OVERDUE", 0),
                "DAYS_CREDIT_ENDDATE": rec.get("DAYS_CREDIT_ENDDATE"),
                "DAYS_ENDDATE_FACT": rec.get("DAYS_ENDDATE_FACT"),
                "AMT_CREDIT_MAX_OVERDUE": rec.get("AMT_CREDIT_MAX_OVERDUE", 0),
                "CNT_CREDIT_PROLONG": rec.get("CNT_CREDIT_PROLONG", 0),
                "AMT_CREDIT_SUM": rec.get("AMT_CREDIT_SUM"),
                "AMT_CREDIT_SUM_DEBT": rec.get("AMT_CREDIT_SUM_DEBT", 0),
                "AMT_CREDIT_SUM_LIMIT": rec.get("AMT_CREDIT_SUM_LIMIT", 0),
                "AMT_CREDIT_SUM_OVERDUE": rec.get("AMT_CREDIT_SUM_OVERDUE", 0),
                "CREDIT_TYPE": rec.get("CREDIT_TYPE"),
                "DAYS_CREDIT_UPDATE": rec.get("DAYS_CREDIT_UPDATE"),
                "AMT_ANNUITY": rec.get("AMT_ANNUITY"),
            })

            for ms in rec.get("monthly_status", []):
                balance_rows.append({
                    "SK_ID_BUREAU": rec.get("SK_ID_BUREAU"),
                    "MONTHS_BALANCE": ms.get("MONTHS_BALANCE"),
                    "STATUS": ms.get("STATUS"),
                })

        bureau_df = pd.DataFrame(bureau_rows)
        balance_df = pd.DataFrame(balance_rows)

        return bureau_df, balance_df

    # ── Mapping helpers ────────────────────────────────────────────────

    @staticmethod
    def _map_gender(gender_str: str | None) -> str:
        if not gender_str:
            return "M"
        g = str(gender_str).lower()
        if g in ("nam", "male", "m"):
            return "M"
        elif g in ("nu", "female", "f"):
            return "F"
        return "XNA"

    @staticmethod
    def _map_marital_status(status: str | None) -> str:
        if not status:
            return "Married"
        s = str(status).lower()
        if "ket hon" in s or "married" in s or "da ket" in s:
            return "Married"
        elif "doc than" in s or "single" in s:
            return "Single / not married"
        elif "ly hon" in s or "divorced" in s:
            return "Separated"
        elif "goa" in s or "widow" in s:
            return "Widow"
        return "Married"

    @staticmethod
    def _map_income_type(contract_type: str | None) -> str:
        if not contract_type:
            return "Working"
        ct = str(contract_type).lower()
        if "huu" in ct or "pension" in ct:
            return "Pensioner"
        elif "kinh doanh" in ct or "business" in ct:
            return "Commercial associate"
        return "Working"

    @staticmethod
    def _map_org_type(employer_name: str | None) -> str:
        if not employer_name:
            return "Business Entity Type 3"
        name = str(employer_name).lower()
        if "cong nghe" in name or "tech" in name or "it" in name:
            return "Business Entity Type 3"
        elif "ngan hang" in name or "bank" in name:
            return "Bank"
        elif "benh vien" in name or "hospital" in name:
            return "Medicine"
        elif "truong" in name or "school" in name or "university" in name:
            return "School"
        return "Business Entity Type 3"

    @staticmethod
    def _map_occupation(position: str | None) -> str:
        if not position:
            return "Laborers"
        p = str(position).lower()
        if "engineer" in p or "developer" in p or "ky su" in p:
            return "Core staff"
        elif "manager" in p or "director" in p or "giam doc" in p:
            return "Managers"
        elif "accountant" in p or "ke toan" in p:
            return "Accountants"
        elif "driver" in p or "lai xe" in p:
            return "Drivers"
        elif "sale" in p:
            return "Sales staff"
        return "Core staff"

    @staticmethod
    def _parse_days_phone_change(info: str | None) -> float:
        """Parse phone change info like '2024-06-15 (khoang 270 ngay truoc)'."""
        if not info:
            return -365.0  # Default: changed ~1 year ago
        import re
        nums = re.findall(r"(\d+)\s*ngay", str(info))
        if nums:
            return -float(nums[0])
        # Try to parse date
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", str(info))
        if dates:
            try:
                d = datetime.strptime(dates[0], "%Y-%m-%d").date()
                return (d - date.today()).days
            except ValueError:
                pass
        return -365.0
