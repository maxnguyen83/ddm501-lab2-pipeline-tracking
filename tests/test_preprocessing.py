"""
Tests for pipeline/preprocessing.py — derived features and the transformer.
"""

import numpy as np
import pandas as pd

from pipeline.config import DERIVED_FEATURES, PAY_FEATURES, TARGET
from pipeline.preprocessing import add_derived_features, build_preprocessor


class TestPreprocessing:
    """Feature engineering and the transformer."""

    def test_derived_features_are_added(self, small):
        """Every name in DERIVED_FEATURES appears in the output."""
        out = add_derived_features(small)
        assert all(f in out.columns for f in DERIVED_FEATURES)
        assert len(out) == len(small)

    def test_original_frame_is_not_mutated(self, small):
        """Calling add_derived_features must not widen the caller's frame."""
        frame = small.copy()
        before = list(frame.columns)
        add_derived_features(frame)
        add_derived_features(frame)
        assert list(frame.columns) == before

    def test_utilisation_ratio_is_sensible(self, small):
        """Non-null values are within [0, 5] — no inf from a zero limit."""
        frame = small.copy()
        frame.loc[:4, "LIMIT_BAL"] = 0
        frame.loc[:4, "BILL_AMT1"] = 0
        out = add_derived_features(frame)
        for col in ("utilisation_ratio", "payment_ratio"):
            values = out[col]
            assert not np.isinf(values).any(), f"{col} contains inf"
            assert values.dropna().between(0, 5).all()
        # A zero limit is "not applicable", not infinitely utilised.
        assert out.loc[:4, "utilisation_ratio"].isna().all()

    def test_max_delay_matches_pay_columns(self, small):
        """max_delay equals the row-wise max of the six PAY_* columns."""
        out = add_derived_features(small)
        pd.testing.assert_series_equal(
            out["max_delay"], small[PAY_FEATURES].max(axis=1), check_names=False
        )
        assert (out["n_months_delayed"] == (small[PAY_FEATURES] > 0).sum(axis=1)).all()
        assert out["n_months_delayed"].between(0, 6).all()

    def test_preprocessor_one_hots_categoricals(self, small):
        """Fitted output has more columns than the input, and the same rows."""
        X = add_derived_features(small.drop(columns=[TARGET]))
        pre = build_preprocessor(list(X.columns))
        assert not hasattr(pre, "transformers_"), "must be returned unfitted"
        Xt = pre.fit_transform(X)
        assert Xt.shape[0] == len(X)
        assert Xt.shape[1] > X.shape[1]
        assert np.isfinite(Xt).all()

    def test_preprocessor_handles_unseen_category(self, small):
        """Fit, then transform a frame with EDUCATION=99. It must not raise.

        This is handle_unknown="ignore" doing its job. A category the model has
        never seen will turn up in production; the request should still get an
        answer rather than a 500.
        """
        X = add_derived_features(small.drop(columns=[TARGET]))
        pre = build_preprocessor(list(X.columns)).fit(X)
        unseen = X.head(5).copy()
        unseen["EDUCATION"] = 99
        out = pre.transform(unseen)
        assert out.shape == (5, pre.transform(X.head(5)).shape[1])
