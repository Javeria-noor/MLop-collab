"""Reusable cleaning helpers for the Telco churn dataset."""

import pandas as pd


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges to numeric; blank strings become NaN.

    Returns a copy, the input DataFrame is not modified. Missing values are
    left as NaN on purpose: imputing them must be fit on the training split
    only, so that no information leaks from the test data.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    return out
