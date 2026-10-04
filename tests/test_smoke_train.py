"""Smoke train: fit the pipeline on a small sample to prove it runs end to end."""

from pathlib import Path

import pandas as pd

from src.train import build_pipeline, split_data

SAMPLE = Path(__file__).parent / "data" / "sample.csv"


def test_pipeline_trains_and_predicts():
    df = pd.read_csv(SAMPLE)
    X_train, X_test, y_train, y_test = split_data(df)

    model = build_pipeline()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    assert len(preds) == len(y_test)
    assert set(preds) <= {0, 1}
    assert (preds == y_test).mean() > 0.5
