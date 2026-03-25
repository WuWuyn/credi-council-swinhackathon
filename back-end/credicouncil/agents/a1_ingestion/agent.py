"""
CREDICOUNCIL A1 — Data Ingestion Agent (Local Version).

# LOCAL_SUB: Uses Docling+EasyOCR+LLM instead of AWS Textract, mock JSON instead of real APIs.
# See LOCAL_SUBSTITUTIONS.md for migration guide.

Orchestrates 4-channel data ingestion:
    1. PDF Documents → DoclingOCR + LLMFieldExtractor → identity/employment/housing fields
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

import asyncio
import hashlib
import logging
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credicouncil.agents.a1_ingestion.cic_service import CICService
from credicouncil.agents.a1_ingestion.internal_db_reader import InternalDBReader
from credicouncil.agents.a1_ingestion.llm_field_extractor import LLMFieldExtractor
from credicouncil.services.docling_ocr_service import DoclingOCRService

logger = logging.getLogger(__name__)


class IngestionAgent:
    """Agent A1 — Data Ingestion & Feature Pipeline (Local Version).

    Takes a customer's document folder and produces structured data
    matching the Home Credit dataset format for feature engineering.
    """

    def __init__(self):
        self.ocr_service = DoclingOCRService()
        self.llm_extractor = LLMFieldExtractor()
        self.cic = CICService()
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

        # Known PDF filename prefixes (01_ to 05_) — skip others
        KNOWN_PDF_PREFIXES = ("01_", "02_", "03_", "04_", "05_")

        # Doc type detection from filename prefix
        PREFIX_DOC_TYPE = {
            "01_": "cccd",
            "02_": "employment",
            "03_": "household",
            "04_": "housing",
            "05_": "loan_application",
        }

        pdf_files = sorted(customer_dir.glob("*.pdf"))

        # ── Docling + EasyOCR + LLM extraction ────────────────────────
        # OCR runs sequentially (local CPU), LLM calls run in PARALLEL
        logger.info("  Engine: DOCLING + EasyOCR + LLM extraction")

        # Step 1: Run OCR on all PDFs sequentially (fast, local CPU)
        ocr_tasks = []  # list of (doc_type, ocr_text)
        for pdf_path in pdf_files:
            if not pdf_path.name.startswith(KNOWN_PDF_PREFIXES):
                logger.info(f"  Skipping non-input PDF: {pdf_path.name}")
                continue

            prefix = pdf_path.name[:3]
            doc_type = PREFIX_DOC_TYPE.get(prefix, "unknown")
            logger.info(f"  Docling OCR: {pdf_path.name} → {doc_type}")

            ocr_result = self.ocr_service.extract(pdf_path)
            ocr_text = ocr_result["markdown"] or ocr_result["raw_text"]
            raw_texts[doc_type] = ocr_text

            if not ocr_text.strip():
                logger.warning(f"  Empty OCR text for {pdf_path.name}, skipping LLM")
                continue

            ocr_tasks.append((doc_type, ocr_text))

        # Step 2: Run ALL LLM extractions in PARALLEL (async gather)
        # Stagger delay between calls to avoid Tier 1 rate limit (15 RPM)
        STAGGER_DELAY = 0.2  # seconds between each API call start
        if ocr_tasks:
            async def _extract_all():
                semaphore = asyncio.Semaphore(5)  # max 5 concurrent

                async def _extract_one(idx, dt, text):
                    # Stagger: each call starts 0.2s after previous
                    await asyncio.sleep(idx * STAGGER_DELAY)
                    async with semaphore:
                        return dt, await self.llm_extractor.extract_async(dt, text)

                tasks = [_extract_one(i, dt, text) for i, (dt, text) in enumerate(ocr_tasks)]
                return await asyncio.gather(*tasks)

            # Always run in a separate thread to avoid conflicts with
            # any existing event loop (uvicorn async or sync test_pipeline)
            llm_results = []
            llm_error = []

            def _run():
                try:
                    llm_results.extend(asyncio.run(_extract_all()))
                except Exception as e:
                    llm_error.append(e)

            logger.info(f"  LLM extraction: {len(ocr_tasks)} PDFs in PARALLEL...")
            t = threading.Thread(target=_run)
            t.start()
            t.join()

            if llm_error:
                raise llm_error[0]

            for doc_type, (fields, conf) in llm_results:
                doc_fields[doc_type] = fields
                confidence_map.update({f"{doc_type}.{k}": v for k, v in conf.items()})
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

        # ── Channel 3: Internal DB ─────────────────────────────────────
        internal_path = customer_dir / "08_internal_db.json"
        internal_dfs = self.internal_db.read(internal_path if internal_path.exists() else None)

        # ── Build application_row (matching application_train columns) ──
        application_row = self._build_application_row(
            doc_fields, cic_result, applicant_id
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

            # Contact flags (from Dơn vay)
            "FLAG_MOBIL": 1,  # Always 1 for application
            "FLAG_EMP_PHONE": loan_app.get("flag_emp_phone", 1 if employment.get("employer_phone") else 0),
            "FLAG_WORK_PHONE": loan_app.get("flag_work_phone", 1 if employment.get("employer_phone") else 0),
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
            "REG_CITY_NOT_LIVE_CITY": 0 if housing.get("reg_city_same_live_city") else 1,
            "REG_CITY_NOT_WORK_CITY": 0 if housing.get("reg_city_same_work_city") else 1,
            "LIVE_CITY_NOT_WORK_CITY": 0 if housing.get("live_city_same_work_city") else 1,

            # Application process (from PDF if available, else auto-captured)
            "WEEKDAY_APPR_PROCESS_START": loan_app.get("weekday_appr",
                today.strftime("%A").upper()[:3] if today.weekday() < 5 else "MONDAY"),
            "HOUR_APPR_PROCESS_START": loan_app.get("hour_appr", datetime.now().hour),

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
            **self._build_document_flags(doc_fields, loan_app),
        }

        # ── Post-processing: sentinel values for non-employed applicants ──
        # Home Credit uses DAYS_EMPLOYED = 365243 (≈1000 years) as sentinel
        # for Pensioner/Unemployed — LLM may extract nonsensical dates.
        income_type = row.get("NAME_INCOME_TYPE", "")
        if income_type in ("Pensioner", "Unemployed", "Student", "Maternity leave"):
            row["DAYS_EMPLOYED"] = 365243
            logger.info(f"  Post-process: {income_type} → DAYS_EMPLOYED set to HC sentinel 365243")

        # Also guard against absurdly large DAYS_EMPLOYED from LLM date errors
        days_emp = row.get("DAYS_EMPLOYED")
        if days_emp is not None and isinstance(days_emp, (int, float)):
            if days_emp > 0:
                # DAYS_EMPLOYED should be negative (past) or 365243 (sentinel)
                # A positive value means a future date — likely extraction error
                row["DAYS_EMPLOYED"] = 365243
                logger.warning(f"  Post-process: DAYS_EMPLOYED={days_emp} > 0 → corrected to 365243")

        return row

    def _build_housing_features(self, housing: dict) -> dict[str, Any]:
        """Build normalized housing features from housing survey data.

        Maps housing survey fields to the 46 housing columns in application_train.

        Strategy:
        - If individual normalized fields (*_norm) are available from the PDF,
          use them directly for _AVG, and derive _MODE/_MEDI with small offsets.
        - Otherwise, fall back to computing from raw fields (floors, area, year)
          or using apartment_quality as a last-resort proxy.
        """
        housing_feats: dict[str, Any] = {}

        # ── Map from parsed normalized fields to Home Credit columns ──
        # Each base column maps to a parsed field name from the PDF.
        NORM_FIELD_MAP = {
            "APARTMENTS":            "apartments_norm",
            "BASEMENTAREA":          "basementarea_norm",
            "YEARS_BEGINEXPLUATATION":"years_beginexpluatation_norm",
            "YEARS_BUILD":           "years_build_norm",
            "COMMONAREA":            "commonarea_norm",
            "ELEVATORS":             "elevators_norm",
            "ENTRANCES":             "entrances_norm",
            "FLOORSMAX":             "floorsmax_norm",
            "FLOORSMIN":             "floorsmin_norm",
            "LANDAREA":              "landarea_norm",
            "LIVINGAPARTMENTS":      "livingapartments_norm",
            "LIVINGAREA":            "livingarea_norm",
            "NONLIVINGAPARTMENTS":   "nonlivingapartments_norm",
            "NONLIVINGAREA":         "nonlivingarea_norm",
        }

        # Check if we have individual normalized fields from the PDF
        has_norm_fields = any(housing.get(nf) is not None for nf in NORM_FIELD_MAP.values())

        if has_norm_fields:
            # ── Path A: Use individually parsed normalized values ──
            for hc_base, parsed_field in NORM_FIELD_MAP.items():
                val = housing.get(parsed_field)
                if val is not None:
                    fval = float(val)
                    housing_feats[hc_base + "_AVG"] = fval
                    # Use separate _MODE/_MEDI values if available, else fall back to AVG
                    mode_field = parsed_field.replace("_norm", "_mode_norm")
                    medi_field = parsed_field.replace("_norm", "_medi_norm")
                    mode_val = housing.get(mode_field)
                    medi_val = housing.get(medi_field)
                    housing_feats[hc_base + "_MODE"] = float(mode_val) if mode_val is not None else fval
                    housing_feats[hc_base + "_MEDI"] = float(medi_val) if medi_val is not None else fval
                else:
                    housing_feats[hc_base + "_AVG"] = None
                    housing_feats[hc_base + "_MODE"] = None
                    housing_feats[hc_base + "_MEDI"] = None
        else:
            # ── Path B: Fallback — use apartment_quality as proxy ──
            quality = housing.get("apartment_quality")
            quality_norm = None
            if quality:
                try:
                    nums = [float(n) for n in str(quality).split("/")
                            if n.strip().replace(".", "").isdigit()]
                    if len(nums) == 2:
                        quality_norm = nums[0] / nums[1]
                    elif len(nums) == 1:
                        quality_norm = nums[0] / 10
                except (ValueError, ZeroDivisionError):
                    pass

            for hc_base in NORM_FIELD_MAP:
                for suffix in ["_AVG", "_MODE", "_MEDI"]:
                    housing_feats[hc_base + suffix] = quality_norm

            # Override specific values where we have raw data
            def _safe_float(val):
                if val is None:
                    return None
                try:
                    return float(val)
                except (ValueError, TypeError):
                    import re as _re
                    nums = _re.findall(r"[\d.]+", str(val))
                    return float(nums[0]) if nums else None

            max_floors_val = _safe_float(housing.get("max_floors"))
            if max_floors_val and max_floors_val > 0:
                floor_norm = min(1.0, max_floors_val / 50)
                for s in ["_AVG", "_MODE", "_MEDI"]:
                    housing_feats["FLOORSMAX" + s] = floor_norm

            living_area = housing.get("living_area")
            if isinstance(living_area, str):
                nums = [float(n) for n in living_area.split()
                        if n.replace(".", "").isdigit()]
                living_area = nums[0] if nums else None
            living_area_val = _safe_float(living_area) if living_area else None
            if living_area_val and living_area_val > 0:
                area_norm = min(1.0, living_area_val / 200)
                for s in ["_AVG", "_MODE", "_MEDI"]:
                    housing_feats["LIVINGAREA" + s] = area_norm

            year_built_val = _safe_float(housing.get("year_built"))
            if year_built_val and year_built_val > 1900:
                year_norm = min(1.0, max(0, (year_built_val - 1950) / 80))
                for s in ["_AVG", "_MODE", "_MEDI"]:
                    housing_feats["YEARS_BUILD" + s] = year_norm

            if housing.get("has_elevator"):
                elev = 1.0 if str(housing["has_elevator"]).lower() in (
                    "co", "yes", "true", "1") else 0.0
                for s in ["_AVG", "_MODE", "_MEDI"]:
                    housing_feats["ELEVATORS" + s] = elev

        # ── Categorical housing fields (same for both paths) ──
        fond = housing.get("fond_kapremont")
        # Normalize "N/A" string to None  
        if fond and str(fond).strip().upper() in ("N/A", "NA", "NONE", "-"):
            fond = None
        housing_feats["FONDKAPREMONT_MODE"] = fond
        housing_feats["HOUSETYPE_MODE"] = housing.get("housetype_mode", housing.get("housing_type", "block of flats"))
        housing_feats["TOTALAREA_MODE"] = housing.get("totalarea_norm", 0.0)
        housing_feats["WALLSMATERIAL_MODE"] = housing.get("wall_material", "Panel")
        es_raw = str(housing.get("emergency_state", "")).strip().lower()
        housing_feats["EMERGENCYSTATE_MODE"] = (
            "No" if (es_raw in ("no", "")
                     or "khong" in es_raw
                     or "binh thuong" in es_raw
                     or "normal" in es_raw)
            else "Yes"
        )

        return housing_feats

    def _build_document_flags(self, doc_fields: dict, loan_app: dict) -> dict[str, int]:
        """Build FLAG_DOCUMENT_2 through FLAG_DOCUMENT_21."""
        flags = {}
        # Document submission flags
        # If parsed from PDF (flag_document_N fields), use those values directly.
        # Otherwise fall back to auto-detecting which doc types were submitted.
        has_parsed_flags = any(
            loan_app.get(f"flag_document_{i}") is not None for i in range(2, 22)
        )

        if has_parsed_flags:
            # Path A: Use explicit FLAG_DOCUMENT values from PDF
            for i in range(2, 22):
                flags[f"FLAG_DOCUMENT_{i}"] = int(loan_app.get(f"flag_document_{i}", 0))
        else:
            # Path B: Fallback — auto-detect from submitted documents
            has_cccd = "cccd" in doc_fields
            has_employment = "employment" in doc_fields
            has_household = "household" in doc_fields
            has_housing = "housing" in doc_fields
            has_loan = "loan_application" in doc_fields

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
        g = str(gender_str).strip().lower()
        if g in ("nam", "male", "m", "nam giới"):
            return "M"
        elif g in ("nữ", "nu", "female", "f", "nữ giới"):
            return "F"
        return "XNA"

    @staticmethod
    def _map_marital_status(status: str | None) -> str:
        if not status:
            return "Married"
        s = str(status)
        # Pass through valid English enum values
        VALID_STATUSES = {
            "Married", "Single / not married", "Separated",
            "Widow", "Civil marriage", "Unknown",
        }
        if s in VALID_STATUSES:
            return s
        # Vietnamese mapping fallback
        s_lower = s.lower()
        if "ket hon" in s_lower or "married" in s_lower or "da ket" in s_lower:
            return "Married"
        elif "doc than" in s_lower or "single" in s_lower:
            return "Single / not married"
        elif "ly hon" in s_lower or "divorced" in s_lower:
            return "Separated"
        elif "goa" in s_lower or "widow" in s_lower:
            return "Widow"
        return "Married"

    @staticmethod
    def _map_income_type(contract_type: str | None) -> str:
        if not contract_type:
            return "Working"
        ct = str(contract_type)
        # Pass through valid English enum values
        VALID_INCOME_TYPES = {
            "Working", "Commercial associate", "Pensioner",
            "State servant", "Unemployed", "Student",
            "Businessman", "Maternity leave",
        }
        if ct in VALID_INCOME_TYPES:
            return ct
        # Otherwise try Vietnamese mapping
        ct_lower = ct.lower()
        if "huu" in ct_lower or "pension" in ct_lower:
            return "Pensioner"
        elif "kinh doanh" in ct_lower or "business" in ct_lower:
            return "Commercial associate"
        elif "nha nuoc" in ct_lower or "state" in ct_lower:
            return "State servant"
        return "Working"

    @staticmethod
    def _map_org_type(employer_name: str | None) -> str:
        if not employer_name:
            return "Business Entity Type 3"
        name = str(employer_name)
        # Pass through valid English enum values (there are 58 types in HC dataset)
        VALID_ORG_TYPES = {
            "Business Entity Type 1", "Business Entity Type 2", "Business Entity Type 3",
            "XNA", "Self-employed", "Other", "Medicine", "Government",
            "School", "Kindergarten", "Construction", "Trade: type 7",
            "Industry: type 11", "Military", "Services", "Security Ministries",
            "Transport: type 2", "Transport: type 4", "Police", "Restaurant",
            "Agriculture", "University", "Industry: type 9", "Bank",
            "Industry: type 3", "Industry: type 1", "Postal", "Trade: type 2",
            "Security", "Trade: type 6", "Industry: type 7",
            "Housing", "Electricity", "Hotel", "Cleaning", "Culture",
            "Telecom", "Insurance", "Emergency", "Legal Services",
            "Advertising", "Mobile", "Realtor", "Industry: type 5",
            "Religion", "Trade: type 1", "Trade: type 3",
            "Industry: type 4", "Industry: type 2",
        }
        if name in VALID_ORG_TYPES:
            return name
        # Otherwise try Vietnamese mapping
        name_lower = name.lower()
        if "cong nghe" in name_lower or "tech" in name_lower or "it" in name_lower:
            return "Business Entity Type 3"
        elif "ngan hang" in name_lower or "bank" in name_lower:
            return "Bank"
        elif "benh vien" in name_lower or "hospital" in name_lower:
            return "Medicine"
        elif "truong" in name_lower or "school" in name_lower or "university" in name_lower:
            return "School"
        return "Business Entity Type 3"

    @staticmethod
    def _map_occupation(position: str | None) -> str:
        if not position:
            return "Laborers"
        p = str(position)
        # Pass through valid English enum values
        VALID_OCCUPATIONS = {
            "Laborers", "Core staff", "Accountants", "Managers",
            "Drivers", "Sales staff", "Cleaning staff", "Cooking staff",
            "Security staff", "Medicine staff", "Private service staff",
            "High skill tech staff", "Low-skill Laborers", "Waiters/barmen staff",
            "Secretaries", "HR staff", "IT staff", "Realty agents",
        }
        if p in VALID_OCCUPATIONS:
            return p
        # Otherwise try Vietnamese mapping
        p_lower = p.lower()
        if "engineer" in p_lower or "developer" in p_lower or "ky su" in p_lower:
            return "Core staff"
        elif "manager" in p_lower or "director" in p_lower or "giam doc" in p_lower:
            return "Managers"
        elif "accountant" in p_lower or "ke toan" in p_lower:
            return "Accountants"
        elif "driver" in p_lower or "lai xe" in p_lower:
            return "Drivers"
        elif "sale" in p_lower:
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
