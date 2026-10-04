import pandas as pd

from src.cleaning import clean_total_charges


def test_blank_total_charges_become_nan():
    df = pd.DataFrame({"TotalCharges": ["29.85", " ", "1889.5"]})
    out = clean_total_charges(df)
    assert out["TotalCharges"].isna().sum() == 1
    assert out["TotalCharges"].dtype == "float64"


def test_input_is_not_modified():
    df = pd.DataFrame({"TotalCharges": ["1.0", " "]})
    clean_total_charges(df)
    assert df["TotalCharges"].iloc[1] == " "
