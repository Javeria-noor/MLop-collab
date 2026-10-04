"""Telco Customer Churn: logistic regression pipeline.

Starter code adapted from:
https://www.kaggle.com/code/zahrayousefpournavid/telco-customer-churn-pipeline-logistic-regression

Run from the repo root: uv run python src/train.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

os.environ["PYTHONWARNINGS"] = "ignore"

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

SEED = 42
SPLIT_SEED = 22
TEST_SIZE = 0.20

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "PaymentMethod",
    "Contract",
    "tenure_group",
    "num_services_group",
]

PARAM_GRID = [
    {
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs", "liblinear"],
        "model__C": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 6, 10, 20, 50, 100],
    },
    {
        "model__penalty": ["l1"],
        "model__solver": ["liblinear"],
        "model__C": [0.001, 0.01, 0.1, 0.5, 1, 2, 5, 6, 10, 20, 50, 100],
    },
]


class TelcoFeatureEngineer(BaseEstimator, TransformerMixin):
    """Cleaning and feature engineering. Medians are learned in fit() only."""

    def __init__(self):
        self.tenure_median_ = None
        self.total_charges_median_ = None
        self.monthly_charge_median_ = None

    @staticmethod
    def _count_all_services(X):
        binary_cols = [
            "PhoneService",
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        mapping = {
            "Yes": 1,
            "No": 0,
            "No internet service": 0,
            "No phone service": 0,
        }
        count = X[binary_cols].apply(lambda x: x.map(mapping)).fillna(0).sum(axis=1)
        internet = (X["InternetService"] != "No").astype(int)
        return count + internet

    @staticmethod
    def _count_addons(X):
        cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
        mapping = {"Yes": 1, "No": 0, "No internet service": 0}
        return X[cols].apply(lambda x: x.map(mapping)).fillna(0).sum(axis=1)

    @staticmethod
    def _clean_total_charges(X):
        X["TotalCharges"] = X["TotalCharges"].astype(str).str.strip()
        X = X.replace(["", " ", "  ", "NULL", "N/A", "-"], pd.NA)
        X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
        return X

    def fit(self, X, y=None):
        X_temp = self._clean_total_charges(X.copy())
        self.tenure_median_ = X_temp.groupby("tenure")["TotalCharges"].median()
        self.total_charges_median_ = X_temp["TotalCharges"].median()
        self.monthly_charge_median_ = X_temp["MonthlyCharges"].median()
        return self

    def transform(self, X):
        X = X.copy()
        if "customerID" in X.columns:
            X = X.drop("customerID", axis=1)

        X = self._clean_total_charges(X)
        X["TotalCharges"] = X["TotalCharges"].fillna(
            X["tenure"].map(self.tenure_median_)
        )
        X["TotalCharges"] = X["TotalCharges"].fillna(self.total_charges_median_)

        X["AvgCharges"] = X["TotalCharges"] / X["tenure"].replace(0, np.nan).fillna(1)

        auto_pay = ["Bank transfer (automatic)", "Credit card (automatic)"]
        X["AutoPayment"] = X["PaymentMethod"].isin(auto_pay).astype(int)

        X["num_services"] = self._count_all_services(X)
        X["addon_count"] = self._count_addons(X)

        X["HighMonthlyCharges"] = (
            X["MonthlyCharges"] > self.monthly_charge_median_
        ).astype(int)

        X["FamilyStatus"] = (X["Partner"] == "Yes").astype(int) + (
            X["Dependents"] == "Yes"
        ).astype(int)

        X["MonthlyChargesPerService"] = np.where(
            X["num_services"] > 0, X["MonthlyCharges"] / X["num_services"], 0
        )

        X["IsNewCustomer"] = (X["tenure"] <= 6).astype(int)
        X["IsLongTermCustomer"] = (X["tenure"] >= 48).astype(int)

        X["num_services_group"] = pd.cut(
            X["num_services"], bins=[-1, 2, 4, np.inf], labels=["0-2", "3-4", "5+"]
        )

        X["MonthToMonth_HighCharge"] = (
            (X["Contract"] == "Month-to-month")
            & (X["MonthlyCharges"] > self.monthly_charge_median_)
        ).astype(int)

        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[-1, 6, 12, 24, 48, 72],
            labels=["0-6", "7-12", "13-24", "25-48", "49-72"],
        )

        return X


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=True,
    )

    return Pipeline(
        steps=[
            ("feature_engineering", TelcoFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", random_state=SEED, max_iter=1500
                ),
            ),
        ]
    )


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def split_data(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SPLIT_SEED
    )

    y_train = y_train.map({"Yes": 1, "No": 0})
    y_test = y_test.map({"Yes": 1, "No": 0})

    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)

    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print("\nAccuracy:", accuracy_score(y_test, pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, pred))
    print("Churn Precision:", precision_score(y_test, pred))
    print("Churn Recall:", recall_score(y_test, pred))
    print("Churn F1:", f1_score(y_test, pred))


def main():
    df = load_data()
    print("Shape:", df.shape)

    X_train, X_test, y_train, y_test = split_data(df)
    print("Training shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    grid = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best CV F1 Score:", grid.best_score_)

    evaluate(grid.best_estimator_, X_test, y_test)


if __name__ == "__main__":
    main()
