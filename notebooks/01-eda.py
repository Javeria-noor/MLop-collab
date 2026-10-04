# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
# ---

# %% [markdown]
# # 01 - Exploratory data analysis: Telco Customer Churn
#
# Goal: understand the dataset before building the pipeline.
# The cleaning step is imported from `src/cleaning.py` (unit-tested in `tests/`).

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path("..").resolve()))
from src.cleaning import clean_total_charges  # noqa: E402

# %% [markdown]
# ## 1. Load the raw data

# %%
DATA_PATH = Path("..") / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
raw = pd.read_csv(DATA_PATH)
raw.shape

# %%
raw.head()

# %%
raw.dtypes

# %% [markdown]
# `TotalCharges` is stored as text because some rows contain a blank string.
# We convert it with the tested helper `clean_total_charges`, which leaves the
# blanks as NaN. Any imputation is done later, inside the model pipeline and
# fit on the training split only (no leakage).

# %%
df = clean_total_charges(raw)
print(
    df["TotalCharges"].dtype, "| missing TotalCharges:", df["TotalCharges"].isna().sum()
)

# %%
missing = df.isna().sum()
missing[missing > 0]

# %% [markdown]
# ## 2. Target: churn

# %%
df["Churn"].value_counts(normalize=True).round(3)

# %%
df["Churn"].value_counts().plot(kind="bar", title="Churn counts")
plt.show()

# %% [markdown]
# ## 3. Churn vs. customer features

# %%
df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean()).round(3)

# %%
df.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]].mean().round(2)

# %%
df.boxplot(column="MonthlyCharges", by="Churn")
plt.show()

# %% [markdown]
# ## 4. Observations
#
# - The classes are imbalanced: most customers do not churn.
# - Churn rate differs strongly by contract type.
# - Churners differ from non-churners in tenure and monthly charges.
#
# Confirm each point against the outputs above before quoting it in REPORT.md.
