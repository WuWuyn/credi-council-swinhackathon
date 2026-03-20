"""
CreditLens A1 — CIC API Service (Mock + Production).

Client for querying the Credit Information Center (CIC) for credit history.
Includes a mock implementation for development/testing.
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)


class CICService:
    """CIC API client.

    In production, connects to the real CIC API.
    In development, uses mock data that simulates realistic responses.
    """

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock

    def query(self, applicant_id: str) -> dict[str, Any]:
        """Query CIC for credit history.

        Args:
            applicant_id: Applicant identifier (CCCD or internal ID).

        Returns:
            CIC response dict with credit bureau data.
        """
        if self.use_mock:
            return self._mock_query(applicant_id)

        # TODO: Implement real CIC API integration
        raise NotImplementedError("Real CIC API integration not yet implemented")

    def _mock_query(self, applicant_id: str) -> dict[str, Any]:
        """Generate mock CIC response for development.

        Simulates various scenarios: normal, thin-file, and bad debt.
        """
        # Use hash of applicant_id for deterministic mock data
        seed = hash(applicant_id) % 100

        # 20% chance thin-file (no CIC record)
        if seed < 20:
            return {
                "cic_score": None,
                "debt_group": None,
                "num_active_loans": 0,
                "total_outstanding": 0,
                "worst_ever_group": None,
                "thin_file_flag": True,
                "response_status": "NO_RECORD",
            }

        # 10% chance bad debt (group 3-5)
        if seed < 30:
            return {
                "cic_score": random.randint(150, 400),
                "debt_group": random.choice([3, 4, 5]),
                "num_active_loans": random.randint(1, 5),
                "total_outstanding": random.randint(50_000_000, 500_000_000),
                "worst_ever_group": random.choice([3, 4, 5]),
                "thin_file_flag": False,
                "response_status": "OK",
            }

        # 70% normal record
        return {
            "cic_score": random.randint(450, 750),
            "debt_group": random.choice([1, 1, 1, 2]),
            "num_active_loans": random.randint(0, 3),
            "total_outstanding": random.randint(0, 200_000_000),
            "worst_ever_group": random.choice([1, 1, 2]),
            "thin_file_flag": False,
            "response_status": "OK",
        }
