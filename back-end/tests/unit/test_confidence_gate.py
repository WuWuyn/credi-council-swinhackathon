"""
Unit tests for confidence gate logic.
"""

import pytest
from creditlens.orchestrator.confidence_gate import confidence_gate
from creditlens.state.credit_state import RoutingDecision


def test_confidence_gate_halt_on_critical_missing():
    """Test that pipeline HALTs when critical field is below threshold."""
    state = {
        "confidence_map": {
            "identity_verified": 0.42,  # below 0.85 → HALT
            "monthly_income_or_inflow": 0.90,
            "debt_group": 0.95,
        },
        "missing_fields": [],
        "warnings": [],
    }
    result = confidence_gate(state)
    assert result["routing"] == RoutingDecision.HALT.value


def test_confidence_gate_proceed_with_high_confidence():
    """Test AUTO_PROCEED when all fields have high confidence."""
    state = {
        "confidence_map": {
            "identity_verified": 0.95,
            "monthly_income_or_inflow": 0.90,
            "debt_group": 0.95,
            "employment_duration": 0.85,
            "income_stability_index": 0.88,
            "debt_service_behavior": 0.80,
            "collateral_value": 0.75,
            "regular_bill_payment": 0.70,
            "overdraft_count": 0.65,
        },
        "missing_fields": [],
        "warnings": [],
    }
    result = confidence_gate(state)
    assert result["routing"] in (
        RoutingDecision.PROCEED.value,
        RoutingDecision.PROCEED_WITH_WARNINGS.value,
    )


def test_confidence_gate_escalate_on_low_overall():
    """Test ESCALATE when overall confidence is below 0.65."""
    state = {
        "confidence_map": {
            "identity_verified": 0.86,  # just above critical threshold
            "monthly_income_or_inflow": 0.86,
            "debt_group": 0.86,
            "employment_duration": 0.10,
            "income_stability_index": 0.10,
            "debt_service_behavior": 0.10,
            "collateral_value": 0.10,
            "regular_bill_payment": 0.10,
            "overdraft_count": 0.10,
            "transaction_network": 0.10,
        },
        "missing_fields": [],
        "warnings": [],
    }
    result = confidence_gate(state)
    assert result["routing"] == RoutingDecision.ESCALATE_TO_HUMAN.value
