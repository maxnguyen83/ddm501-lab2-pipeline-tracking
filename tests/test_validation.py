"""
Tests for pipeline/validation.py — the data quality gate in front of training.

Each level (schema, statistics, semantics) is exercised with the kind of broken
data it exists to catch, and the report is checked to carry all three levels.
"""

import numpy as np
import pytest

from pipeline.config import TARGET
from pipeline.validation import DataValidationError, validate_dataset


class TestValidation:
    """The quality gate in front of training."""

    def test_clean_data_passes(self, raw):
        """validate_dataset(raw) returns passed=True and n_errors == 0."""
        report = validate_dataset(raw)
        assert report["passed"] is True
        assert report["n_errors"] == 0
        assert report["n_rows"] == len(raw)
        assert report["schema_errors"] == []
        assert report["statistical_errors"] == []
        assert report["semantic_errors"] == []

    def test_missing_column_is_caught(self, small):
        """Drop LIMIT_BAL and assert DataValidationError is raised."""
        with pytest.raises(DataValidationError, match="LIMIT_BAL"):
            validate_dataset(small.drop(columns=["LIMIT_BAL"]))

    def test_out_of_domain_category_is_caught(self, small):
        """Set SEX to 7 on a few rows and assert it raises."""
        bad = small.copy()
        bad.loc[:4, "SEX"] = 7
        with pytest.raises(DataValidationError, match="SEX"):
            validate_dataset(bad)

    def test_impossible_age_is_caught(self, small):
        """Set AGE to 400 on a few rows and assert it raises."""
        bad = small.copy()
        bad.loc[:4, "AGE"] = 400
        with pytest.raises(DataValidationError, match="AGE"):
            validate_dataset(bad)

    def test_negative_payment_is_caught(self, small):
        """Set PAY_AMT1 to -100 on a few rows and assert it raises."""
        bad = small.copy()
        bad.loc[:4, "PAY_AMT1"] = -100
        with pytest.raises(DataValidationError, match="PAY_AMT1"):
            validate_dataset(bad)

    def test_degenerate_target_is_caught(self, small):
        """Set the whole target column to 0 and assert it raises.

        This is the check that matters most. Without it, a broken upstream
        extract produces a model with 100% accuracy that predicts "no default"
        for every applicant, and nothing in the pipeline objects.
        """
        bad = small.copy()
        bad[TARGET] = 0
        with pytest.raises(DataValidationError, match="positive rate"):
            validate_dataset(bad)

    def test_report_is_returned_when_not_raising(self, small):
        """With raise_on_error=False, a bad frame returns passed=False."""
        bad = small.drop(columns=["AGE"]).copy()
        bad.loc[:4, "SEX"] = 7
        bad[TARGET] = 0
        report = validate_dataset(bad, raise_on_error=False)
        assert report["passed"] is False
        # One error per level: all three are reported together, not one at a time.
        assert report["schema_errors"]
        assert report["statistical_errors"]
        assert report["semantic_errors"]
        assert report["n_errors"] == (
            len(report["schema_errors"]) + len(report["statistical_errors"])
            + len(report["semantic_errors"])
        )

    def test_too_few_rows_and_missing_values_are_caught(self, small):
        """Below MIN_ROWS, or a column above the missing-value budget, fails level 2."""
        bad = small.head(100).copy()
        bad["BILL_AMT1"] = bad["BILL_AMT1"].astype(float)
        bad.loc[:20, "BILL_AMT1"] = np.nan
        report = validate_dataset(bad, raise_on_error=False)
        joined = " ".join(report["statistical_errors"])
        assert "too few rows" in joined
        assert "BILL_AMT1" in joined

    def test_non_numeric_column_is_caught(self, small):
        """A column that arrives as text fails the schema level."""
        bad = small.copy()
        bad["LIMIT_BAL"] = bad["LIMIT_BAL"].astype(str)
        report = validate_dataset(bad, raise_on_error=False)
        assert any("LIMIT_BAL" in e for e in report["schema_errors"])
