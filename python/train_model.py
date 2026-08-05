import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, precision_recall_curve
)
from xgboost import XGBClassifier
import joblib
import os
from dotenv import load_dotenv


load_dotenv()

DB_USER = "root"
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = "localhost"
DB_NAME = "credit_risk_platform"
MODEL_VERSION = "v1"
MODEL_OUTPUT_PATH = f"../models/xgb_model_{MODEL_VERSION}.joblib"
os.makedirs("../models", exist_ok=True)

engine = create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
)

print("=" * 60)
print("STAGE 1: Loading data from final_model_features view")
print("=" * 60)

df = pd.read_sql("SELECT * FROM final_model_features", engine)
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns.\n")

print("=" * 60)
print("STAGE 2: EDA")
print("=" * 60)

print("\n--- Missing values per column ---")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\n--- Class balance (default_flag) ---")
print(df["default_flag"].value_counts(normalize=True).round(4))

print("\n--- Numeric feature summary ---")
print(df.describe().T[["mean", "std", "min", "max"]])


print("\n" + "=" * 60)
print("STAGE 3: Preprocessing")
print("=" * 60)

# employment_length has NaNs (some borrowers report "< 1 year" or missing) —
# fill with 0 rather than dropping rows.
df["employment_length"] = df["employment_length"].fillna(0)

# Some annual_income / dti rows may be missing or zero — drop those,
# since they're a small fraction and imputing income is risky.
before = len(df)
df = df.dropna(subset=["annual_income", "dti", "loan_to_income_ratio"])
df = df[df["annual_income"] > 0]
print(f"Dropped {before - len(df):,} rows with missing/invalid income or dti.")

# Data quality fixes found during EDA:
# 1. dti == 999 is a known Lending Club placeholder for "missing/invalid",
#    not a real debt-to-income ratio. Real dti values are essentially
#    never above ~60-70. Drop rows using this placeholder.
before = len(df)
df = df[df["dti"] < 100]
print(f"Dropped {before - len(df):,} rows with dti placeholder value (>=100).")

# 2. annual_income has extreme outliers (max seen: $9.2M) that are either
#    data entry errors or such rare edge cases they destabilize a linear
#    model. Cap at the 99th percentile rather than dropping the rows
#    outright, so we don't lose otherwise-valid loans.
income_cap = df["annual_income"].quantile(0.99)
n_capped = (df["annual_income"] > income_cap).sum()
df["annual_income"] = df["annual_income"].clip(upper=income_cap)
print(f"Capped {n_capped:,} extreme annual_income values at the 99th percentile (${income_cap:,.0f}).")

# 3. Recompute loan_to_income_ratio after the income fix above, since it
#    was originally computed in SQL using the uncapped income.
df["loan_to_income_ratio"] = (df["loan_amount"] / df["annual_income"]).round(4)

# 4. Group rare `purpose` categories into "other". A handful of loans in
#    a rare category can otherwise cause quasi-complete separation in
#    Logistic Regression (that category's coefficient tries to grow
#    toward infinity to fit a tiny, possibly homogeneous group perfectly)
#    — this is the actual cause of the overflow warnings, not the
#    earlier scale/outlier issues.
purpose_counts = df["purpose"].value_counts()
rare_purposes = purpose_counts[purpose_counts < 200].index
n_regrouped = df["purpose"].isin(rare_purposes).sum()
df["purpose"] = df["purpose"].where(~df["purpose"].isin(rare_purposes), "other")
print(f"Regrouped {n_regrouped:,} rows from rare purpose categories into 'other'.")

numeric_features = [
    "loan_amount", "interest_rate", "term_months", "dti",
    "annual_income", "employment_length", "loan_to_income_ratio",
    "had_delinquency", "interest_rate_percentile_in_grade",
    "grade_default_rate", "region_default_rate", "loan_amount_rank_in_grade",
]
categorical_features = ["grade", "home_ownership", "purpose"]

X = df[numeric_features + categorical_features]
y = df["default_flag"]
loan_ids = df["loan_id"]  # keep aside — needed later to write scores back to MySQL

X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
    X, y, loan_ids, test_size=0.2, random_state=42, stratify=y
)
# Held-out calibration split, carved out of what would've been the train set —
# XGBoost trains on X_train only, then CalibratedClassifierCV is fit on
# X_calib, which it's never seen, so the calibration mapping isn't just
# re-fitting the model's own overconfidence back onto itself.
X_train, X_calib, y_train, y_calib, ids_train, ids_calib = train_test_split(
    X_temp, y_temp, ids_temp, test_size=0.25, random_state=42, stratify=y_temp
)
print(f"Train: {len(X_train):,} rows | Calibration: {len(X_calib):,} rows | Test: {len(X_test):,} rows")

print("\n" + "=" * 60)
print("STAGE 4: Training")
print("=" * 60)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", StandardScaler(), numeric_features),
    ],
)

# Baseline model — simple, interpretable, good sanity check
logreg_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced", C=0.1, solver="liblinear")),
])
logreg_pipeline.fit(X_train, y_train)

# Main model — XGBoost, handles the class imbalance via scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )),
])
xgb_pipeline.fit(X_train, y_train)

# Calibration — scale_pos_weight biases XGBoost's raw probabilities toward
# the extremes (it's optimizing for ranking/class separation, not calibrated
# probability estimates), so a "70% risk score" ends up not meaning "70% of
# these default". CalibratedClassifierCV learns a correction mapping on the
# held-out calibration set, using the already-fitted pipeline frozen as-is
# (no refitting on calibration data).
calibrated_model = CalibratedClassifierCV(FrozenEstimator(xgb_pipeline), method="isotonic")
calibrated_model.fit(X_calib, y_calib)


print("STAGE 5: Evaluation")
print("=" * 60)

for name, pipe in [
    ("Logistic Regression", logreg_pipeline),
    ("XGBoost (raw)", xgb_pipeline),
    ("XGBoost (calibrated)", calibrated_model),
]:
    preds = pipe.predict(X_test)
    probs = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    print(f"\n--- {name} ---")
    print(classification_report(y_test, preds, digits=3))
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

# Save both the raw pipeline (SHAP needs the actual XGBoost booster — it
# can't explain through the calibration wrapper) and the calibrated model
# (used for every risk score / probability shown downstream).
joblib.dump(
    {"pipeline": xgb_pipeline, "calibrated_model": calibrated_model},
    MODEL_OUTPUT_PATH,
)
print(f"\nSaved XGBoost pipeline + calibrated model to {MODEL_OUTPUT_PATH}")

joblib.dump(
    {"X_test": X_test, "y_test": y_test, "ids_test": ids_test},
    "../models/test_set.joblib"
)
print("Saved test set for downstream SHAP explainability + risk scoring.")
