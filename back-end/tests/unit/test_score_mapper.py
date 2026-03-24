"""
Unit tests for score mapper.
"""

import pytest
from creditlens.agents.a3_scoring.score_mapper import (
    pd_to_credit_score,
    credit_score_to_risk_band,
    map_prediction,
)


class TestPdToCreditScore:
    """Tests for PD → credit score mapping."""

    def test_low_pd_gives_high_score(self):
        """PD 1% should map to AAA range (720-850)."""
        score = pd_to_credit_score(1.0)
        assert 720 <= score <= 850

    def test_medium_pd_gives_medium_score(self):
        """PD 10% should be in A range (560-639)."""
        score = pd_to_credit_score(10.0)
        assert 500 <= score <= 700

    def test_high_pd_gives_low_score(self):
        """PD 50% should be in CC range (300-459)."""
        score = pd_to_credit_score(50.0)
        assert 300 <= score <= 500

    def test_score_is_monotonic(self):
        """Higher PD should always give lower score."""
        pds = [1, 5, 10, 20, 40, 80]
        scores = [pd_to_credit_score(pd) for pd in pds]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], f"Score not monotonic at PD={pds[i+1]}"

    def test_score_clamped_to_range(self):
        """Score should always be between 300-850."""
        for pd in [0.001, 0.01, 0.1, 1, 10, 50, 99.99]:
            score = pd_to_credit_score(pd)
            assert 300 <= score <= 850, f"Score {score} out of range for PD={pd}"


class TestRiskBandClassification:
    """Tests for credit score → risk band mapping."""

    def test_aaa_band(self):
        band = credit_score_to_risk_band(750)
        assert band.band == "AAA"

    def test_aa_band(self):
        band = credit_score_to_risk_band(680)
        assert band.band == "AA"

    def test_cc_band(self):
        band = credit_score_to_risk_band(350)
        assert band.band == "CC"

    def test_boundary_values(self):
        """Test boundary between bands."""
        assert credit_score_to_risk_band(720).band == "AAA"
        assert credit_score_to_risk_band(719).band == "AA"
        assert credit_score_to_risk_band(640).band == "AA"
        assert credit_score_to_risk_band(639).band == "A"


class TestMapPrediction:
    """Tests for the full prediction mapping pipeline."""

    def test_full_mapping(self):
        result = map_prediction(0.05)  # 5% default probability
        assert "credit_score" in result
        assert "pd_pct" in result
        assert "risk_band" in result
        assert "auto_decision" in result
        assert result["pd_pct"] == 5.0
