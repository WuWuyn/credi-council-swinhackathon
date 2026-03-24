"""
Unit tests for decision rules (hard overrides).
"""

import pytest
from credicouncil.agents.a3_scoring.decision_rules import apply_hard_overrides


class TestHardOverrides:
    """Tests for policy-based hard override rules."""

    def test_cic_group_4_forces_reject(self):
        """CIC Group 4 should always result in REJECT."""
        result = apply_hard_overrides(
            credit_score=750,  # Even with excellent score
            risk_band="AAA",
            auto_decision="AUTO_APPROVE",
            structured_feats={"debt_group": 4},
            overall_confidence=0.90,
        )
        assert result["final_decision"] == "REJECT"
        assert result["override_applied"] is True

    def test_cic_group_5_forces_reject(self):
        """CIC Group 5 should always result in REJECT."""
        result = apply_hard_overrides(
            credit_score=700,
            risk_band="AA",
            auto_decision="APPROVE_REVIEW",
            structured_feats={"debt_group": 5},
            overall_confidence=0.85,
        )
        assert result["final_decision"] == "REJECT"

    def test_large_loan_escalates(self):
        """Loan > 10B VND should ESCALATE."""
        result = apply_hard_overrides(
            credit_score=800,
            risk_band="AAA",
            auto_decision="AUTO_APPROVE",
            structured_feats={"loan_amount_vnd": 15_000_000_000},
            overall_confidence=0.95,
        )
        assert result["final_decision"] == "ESCALATE"

    def test_low_confidence_triggers_review(self):
        """Confidence < 65% should trigger HUMAN REVIEW."""
        result = apply_hard_overrides(
            credit_score=700,
            risk_band="AA",
            auto_decision="APPROVE_REVIEW",
            structured_feats={"debt_group": 1},
            overall_confidence=0.55,
        )
        assert result["final_decision"] == "REVIEW"

    def test_thin_file_low_score_conditional(self):
        """Thin-file + score < 560 should add collateral condition."""
        result = apply_hard_overrides(
            credit_score=520,
            risk_band="BBB",
            auto_decision="CONDITIONAL",
            structured_feats={"thin_file_flag": True},
            overall_confidence=0.75,
        )
        assert len(result["additional_conditions"]) > 0

    def test_no_override_normal_case(self):
        """Normal case: no overrides should be applied."""
        result = apply_hard_overrides(
            credit_score=700,
            risk_band="AA",
            auto_decision="APPROVE_REVIEW",
            structured_feats={"debt_group": 1, "loan_amount_vnd": 50_000_000},
            overall_confidence=0.85,
        )
        assert result["override_applied"] is False
        assert result["final_decision"] == "APPROVE_REVIEW"
