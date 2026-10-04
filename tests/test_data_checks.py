"""Data checks on the committed sample (schema, value ranges, null counts)."""

from pathlib import Path

import pandas as pd
import pytest

SAMPLE = Path(__file__).parent / "data" / "sample.csv"

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(SAMPLE)


def test_schema(df):
    assert list(df.columns) == EXPECTED_COLUMNS


def test_no_null_values(df):
    assert df.isna().sum().sum() == 0


def test_customer_id_unique(df):
    assert df["customerID"].is_unique


def test_value_ranges(df):
    assert df["tenure"].between(0, 72).all()
    assert df["MonthlyCharges"].between(15, 125).all()
    assert df["SeniorCitizen"].isin([0, 1]).all()
    assert df["gender"].isin(["Male", "Female"]).all()
    assert df["Churn"].isin(["Yes", "No"]).all()


def test_total_charges_blanks_are_rare(df):
    blank = df["TotalCharges"].astype(str).str.strip() == ""
    assert blank.mean() <= 0.05
    numeric = pd.to_numeric(df["TotalCharges"], errors="coerce").dropna()
    assert (numeric >= 0).all()


def test_both_classes_present(df):
    assert df["Churn"].nunique() == 2
