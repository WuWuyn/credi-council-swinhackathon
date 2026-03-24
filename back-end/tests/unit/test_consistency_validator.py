"""
Unit tests for consistency validator.
"""

import pytest
from credicouncil.agents.a4_report_generator.consistency_validator import validate_narrative_consistency


class TestConsistencyValidator:
    """Tests for SHAP-narrative consistency validation."""

    def test_passes_when_narrative_references_shap(self):
        """Validator should PASS when narrative mentions SHAP factors."""
        shap_output = {
            "top_positive_factors": [
                {"feature": "salary_pattern_detected", "shap": 0.089, "value": True,
                 "label_vi": "Phát hiện giao dịch lương đều đặn"},
                {"feature": "income_stability_index", "shap": 0.072, "value": 0.81,
                 "label_vi": "Thu nhập ổn định 6 tháng"},
            ],
            "top_negative_factors": [
                {"feature": "dti_ratio", "shap": -0.063, "value": 0.48,
                 "label_vi": "Tỷ lệ nợ/thu nhập ở mức cao"},
            ],
        }

        narrative = {
            "character_assessment": {
                "narrative": "Khách hàng có giao dịch lương đều đặn trong 6 tháng."
            },
            "capacity_assessment": {
                "narrative": "Thu nhập ổn định nhưng tỷ lệ nợ/thu nhập ở mức cao."
            },
            "capital_assessment": {
                "narrative": "Giao dịch lương cho thấy nguồn thu nhập ổn định."
            },
            "conditions_assessment": {
                "narrative": "Thu nhập ổn định hỗ trợ khả năng trả nợ."
            },
        }

        result = validate_narrative_consistency(shap_output, narrative)
        assert result["passed"] is True
        assert result["shap_coverage"] > 0

    def test_fails_when_narrative_lacks_shap_grounding(self):
        """Validator should FAIL when narrative doesn't reference any SHAP factors."""
        shap_output = {
            "top_positive_factors": [
                {"feature": "salary_pattern_detected", "shap": 0.089,
                 "label_vi": "Phát hiện giao dịch lương đều đặn"},
            ],
            "top_negative_factors": [
                {"feature": "dti_ratio", "shap": -0.063,
                 "label_vi": "Tỷ lệ nợ/thu nhập ở mức cao"},
            ],
        }

        narrative = {
            "character_assessment": {
                "narrative": "Khách hàng là người tốt."  # No SHAP references
            },
            "capacity_assessment": {
                "narrative": "Có khả năng trả nợ."  # No SHAP references
            },
            "capital_assessment": {
                "narrative": "Tài sản đủ."
            },
            "conditions_assessment": {
                "narrative": "Điều kiện tốt."
            },
        }

        result = validate_narrative_consistency(shap_output, narrative)
        assert result["passed"] is False
        assert len(result["violations"]) > 0

    def test_handles_empty_narrative(self):
        """Validator should flag empty narratives as violations."""
        shap_output = {
            "top_positive_factors": [
                {"feature": "x", "shap": 0.1, "label_vi": "Test factor"},
            ],
            "top_negative_factors": [],
        }

        narrative = {
            "character_assessment": {"narrative": ""},
            "capacity_assessment": {"narrative": ""},
            "capital_assessment": {"narrative": ""},
            "conditions_assessment": {"narrative": ""},
        }

        result = validate_narrative_consistency(shap_output, narrative)
        assert result["passed"] is False
