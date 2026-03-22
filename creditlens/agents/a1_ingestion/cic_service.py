"""
CreditLens A1 — CIC API Client (Local Mock + Production).

# LOCAL_SUB: For production, replace mock JSON file reading with real CIC API calls.
# See LOCAL_SUBSTITUTIONS.md for migration guide.

Reads CIC data from mock JSON files or queries real CIC API.
Returns structured bureau records, external scores, and social circle data.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class CICService:
    """CIC (Credit Information Center) API client.

    # LOCAL_SUB: Mock reads from JSON file. Production: send HTTP request to CIC API.

    In local mode, reads from pre-generated mock JSON files.
    In production mode, calls the real CIC API endpoint.
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

    def query(self, cic_data_path: str | Path | None = None) -> dict[str, Any]:
        """Query CIC for credit history.

        Args:
            cic_data_path: Path to mock CIC JSON file (local mode).

        Returns:
            Structured CIC response with bureau records.
        """
        if self.use_mock and cic_data_path:
            return self._read_mock(Path(cic_data_path))
        elif self.use_mock:
            return self._default_thin_file()
        else:
            # LOCAL_SUB: Implement real CIC API call here
            raise NotImplementedError("Real CIC API not implemented. See LOCAL_SUBSTITUTIONS.md")

    def _read_mock(self, path: Path) -> dict[str, Any]:
        """Read mock CIC response from JSON file."""
        if not path.exists():
            logger.warning(f"CIC mock file not found: {path}")
            return self._default_thin_file()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"CIC mock loaded: {len(data.get('bureau_records', []))} bureau records")

        # Extract structured features matching dataset columns
        result = {
            # External scores (application_train columns)
            "EXT_SOURCE_1": data.get("ext_source_scores", {}).get("EXT_SOURCE_1"),
            "EXT_SOURCE_2": data.get("ext_source_scores", {}).get("EXT_SOURCE_2"),
            "EXT_SOURCE_3": data.get("ext_source_scores", {}).get("EXT_SOURCE_3"),

            # Credit inquiry counts (application_train columns)
            "AMT_REQ_CREDIT_BUREAU_HOUR": data.get("credit_inquiry_counts", {}).get("AMT_REQ_CREDIT_BUREAU_HOUR", 0),
            "AMT_REQ_CREDIT_BUREAU_DAY": data.get("credit_inquiry_counts", {}).get("AMT_REQ_CREDIT_BUREAU_DAY", 0),
            "AMT_REQ_CREDIT_BUREAU_WEEK": data.get("credit_inquiry_counts", {}).get("AMT_REQ_CREDIT_BUREAU_WEEK", 0),
            "AMT_REQ_CREDIT_BUREAU_MON": data.get("credit_inquiry_counts", {}).get("AMT_REQ_CREDIT_BUREAU_MON", 0),
            "AMT_REQ_CREDIT_BUREAU_QRT": data.get("credit_inquiry_counts", {}).get("AMT_REQ_CREDIT_BUREAU_QRT", 0),
            "AMT_REQ_CREDIT_BUREAU_YEAR": data.get("credit_inquiry_counts", {}).get("AMT_REQ_CREDIT_BUREAU_YEAR", 0),

            # Social circle (application_train columns)
            "OBS_30_CNT_SOCIAL_CIRCLE": data.get("social_circle", {}).get("OBS_30_CNT_SOCIAL_CIRCLE", 0),
            "DEF_30_CNT_SOCIAL_CIRCLE": data.get("social_circle", {}).get("DEF_30_CNT_SOCIAL_CIRCLE", 0),
            "OBS_60_CNT_SOCIAL_CIRCLE": data.get("social_circle", {}).get("OBS_60_CNT_SOCIAL_CIRCLE", 0),
            "DEF_60_CNT_SOCIAL_CIRCLE": data.get("social_circle", {}).get("DEF_60_CNT_SOCIAL_CIRCLE", 0),

            # Bureau records (bureau.csv equivalent)
            "bureau_records": data.get("bureau_records", []),

            # Thin file flag
            "thin_file_flag": data.get("thin_file_flag", False),
            "cic_score": data.get("cic_score_equivalent"),
            "debt_group": data.get("debt_group", 1),
        }

        return result

    def _default_thin_file(self) -> dict[str, Any]:
        """Default response when no CIC data available (thin-file)."""
        return {
            "EXT_SOURCE_1": None,
            "EXT_SOURCE_2": None,
            "EXT_SOURCE_3": None,
            "AMT_REQ_CREDIT_BUREAU_HOUR": 0,
            "AMT_REQ_CREDIT_BUREAU_DAY": 0,
            "AMT_REQ_CREDIT_BUREAU_WEEK": 0,
            "AMT_REQ_CREDIT_BUREAU_MON": 0,
            "AMT_REQ_CREDIT_BUREAU_QRT": 0,
            "AMT_REQ_CREDIT_BUREAU_YEAR": 0,
            "OBS_30_CNT_SOCIAL_CIRCLE": 0,
            "DEF_30_CNT_SOCIAL_CIRCLE": 0,
            "OBS_60_CNT_SOCIAL_CIRCLE": 0,
            "DEF_60_CNT_SOCIAL_CIRCLE": 0,
            "bureau_records": [],
            "thin_file_flag": True,
            "cic_score": None,
            "debt_group": None,
        }
