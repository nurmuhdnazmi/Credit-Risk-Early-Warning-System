"""
Day 1 — Data ingestion script (v2)
Loads a trimmed subset of the Lending Club `loan.csv` dataset into MySQL.
 
Repayment behaviour is built from REAL summary fields in the dataset
(total_pymnt, last_pymnt_d, last_pymnt_amnt, delinq_2yrs, total_rec_late_fee)
rather than simulated data. This is a snapshot per loan, not a full
payment-by-payment ledger — document that distinction in your README.
 
Prereqs:
    pip install pandas sqlalchemy mysql-connector-python
"""
 
import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

# ------------------------------------------------------------
# 1. CONFIG — edit these for your local setup
# ------------------------------------------------------------
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "credit_risk_platform")
 
RAW_CSV_PATH = "archive/loan.csv"      # <-- confirm path matches your machine
SAMPLE_SIZE = 75000
RANDOM_STATE = 42
 
# Only load the columns we actually need — cuts memory use dramatically
# on a 151-column, 2M+ row file.
USE_COLS = [
    "annual_inc", "emp_length", "addr_state", "dti", "home_ownership",
    "loan_amnt", "int_rate", "term", "grade", "sub_grade", "purpose",
    "issue_d", "loan_status",
    "total_pymnt", "last_pymnt_d", "last_pymnt_amnt",
    "delinq_2yrs", "total_rec_late_fee",
]
 
engine = create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
)
 
# ------------------------------------------------------------
# 2. LOAD + FILTER + SAMPLE
# ------------------------------------------------------------
print("Loading CSV (selected columns only)...")
raw = pd.read_csv(RAW_CSV_PATH, usecols=USE_COLS, low_memory=False)
print(f"Loaded {len(raw):,} raw rows.")
 
resolved_statuses = ["Fully Paid", "Charged Off", "Default"]
raw = raw[raw["loan_status"].isin(resolved_statuses)]
print(f"{len(raw):,} rows remain after keeping only resolved loans.")
 
df = raw.sample(n=min(SAMPLE_SIZE, len(raw)), random_state=RANDOM_STATE).copy()
df.reset_index(drop=True, inplace=True)
print(f"Working with {len(df):,} loans after sampling.")
 
# ------------------------------------------------------------
# 3. BUILD customers TABLE
# Note: Lending Club has no true customer_id or age field —
# each loan's borrower attributes are treated as one customer row.
# Document both limitations in your README.
# ------------------------------------------------------------
customers = pd.DataFrame({
    "annual_income": df["annual_inc"],
    "employment_length": (
        df["emp_length"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    ),
    "region": df["addr_state"],
    "dti": df["dti"],
    "home_ownership": df["home_ownership"],
})
customers.index = range(1, len(customers) + 1)
customers.index.name = "customer_id"
 
customers.to_sql("customers", engine, if_exists="append", index=True, index_label="customer_id")
print(f"Inserted {len(customers):,} customers.")
 
# ------------------------------------------------------------
# 4. BUILD loans TABLE
# ------------------------------------------------------------
int_rate_clean = (
    df["int_rate"].astype(str).str.rstrip("%").astype(float)
)
term_clean = df["term"].astype(str).str.extract(r"(\d+)")[0].astype(int)
 
loans = pd.DataFrame({
    "customer_id": customers.index,
    "loan_amount": df["loan_amnt"].values,
    "interest_rate": int_rate_clean.values,
    "term_months": term_clean.values,
    "grade": df["grade"].values,
    "sub_grade": df["sub_grade"].values,
    "purpose": df["purpose"].values,
    "issue_date": pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce").values,
    "default_flag": df["loan_status"].isin(["Charged Off", "Default"]).astype(int).values,
})
loans.index = range(1, len(loans) + 1)
loans.index.name = "loan_id"
 
loans.to_sql("loans", engine, if_exists="append", index=True, index_label="loan_id")
print(f"Inserted {len(loans):,} loans.")
 
# ------------------------------------------------------------
# 5. BUILD repayments TABLE (real summary fields, one row per loan)
# ------------------------------------------------------------
repayments = pd.DataFrame({
    "loan_id": loans.index,
    "payment_date": pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce").values,
    "amount_paid": df["last_pymnt_amnt"].values,
    # days_late isn't directly available at this granularity — using
    # delinq_2yrs as a real (if coarser) proxy signal instead of
    # fabricating a days_late number. Handle this honestly in your
    # feature engineering / README rather than treating it as exact.
    "days_late": (df["delinq_2yrs"].fillna(0) > 0).astype(int).values,  # 1 = had a delinquency, 0 = none
})
repayments.to_sql("repayments", engine, if_exists="append", index=False)
print(f"Inserted {len(repayments):,} repayment snapshot rows.")
 
print(
    "\nDone. NOTE FOR YOUR README: `repayments` here is a one-row-per-loan "
    "snapshot built from real aggregate fields (last payment, total paid, "
    "delinquency flag), not a full payment-by-payment ledger — the raw "
    "dataset doesn't provide that level of granularity. Be explicit about "
    "this when describing the feature engineering in Day 2."
)
 
