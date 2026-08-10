"""
SHAP explainability, decision layer, and Expected Loss.
Loads the trained model + test set, scores every loan, works out EL,
generates SHAP explanations, and writes everything back to MySQL.

pip install shap matplotlib
"""

import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from sklearn.calibration import calibration_curve
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from datetime import date

load_dotenv()

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "credit_risk_platform")
MODEL_VERSION = "v1"

# LGD isn't modeled, just a flat assumption — 40-45% is a commonly cited
# range for unsecured consumer credit.
LGD_ASSUMPTION = 0.45

engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

saved_model = joblib.load(f"../models/xgb_model_{MODEL_VERSION}.joblib")
xgb_pipeline = saved_model["pipeline"]  # raw preprocess+model, needed for SHAP
calibrated_model = saved_model["calibrated_model"]  # source of every probability used below
saved = joblib.load("../models/test_set.joblib")
X_test, y_test, ids_test = saved["X_test"], saved["y_test"], saved["ids_test"]
print(f"Loaded model and test set ({len(X_test):,} rows).")

probs = calibrated_model.predict_proba(X_test)[:, 1]
risk_scores = (probs * 100).round(2)

# Fixed score cutoffs (75/50/25) were tuned against the old, inflated
# pre-calibration scores — a true calibrated 75%+ default probability is
# rare at a ~20% base rate, so they no longer carve up the portfolio
# meaningfully. Tiers are now percentile-based against the actual scored
# population: top 5% get manual review, next 15% reduced limit, next 30%
# increased monitoring, bottom 50% standard.
p50, p80, p95 = np.percentile(risk_scores, [50, 80, 95])
print(f"\nPercentile-based tier cutoffs: P50={p50:.1f}  P80={p80:.1f}  P95={p95:.1f}")


def recommend_action(score):
    if score >= p95:
        return "Manual review"
    elif score >= p80:
        return "Reduce credit limit"
    elif score >= p50:
        return "Increase monitoring"
    else:
        return "Standard monitoring"


recommended_actions = [recommend_action(s) for s in risk_scores]

# EL = PD x LGD x EAD. EAD is approximated as the loan's original amount —
# a real EAD would account for amortization, but that's not available here.
exposure_at_default = X_test["loan_amount"].values
expected_loss = probs * LGD_ASSUMPTION * exposure_at_default

results = pd.DataFrame({
    "loan_id": ids_test.values,
    "risk_score": risk_scores,
    "probability_of_default": probs.round(5),
    "exposure_at_default": exposure_at_default,
    "expected_loss": expected_loss.round(2),
    "recommended_action": recommended_actions,
    "actual_default": y_test.values,
})

print(f"Portfolio Expected Loss (test set): ${results['expected_loss'].sum():,.0f}")
print(results.groupby("recommended_action")["expected_loss"].sum().round(0))

print("\ndefault rate by recommended action (should climb with severity):")
print(results.groupby("recommended_action")["actual_default"].mean().round(3))

preprocessor = xgb_pipeline.named_steps["preprocess"]
model = xgb_pipeline.named_steps["model"]
X_test_transformed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_transformed)

plt.figure()
shap.summary_plot(shap_values, X_test_transformed, feature_names=feature_names, show=False, plot_type="bar")
plt.tight_layout()
plt.savefig("../models/shap_feature_importance.png", dpi=150)
plt.close()
print("saved shap_feature_importance.png")


def top_features_for_row(row_idx, n=3):
    row_shap = shap_values[row_idx]
    top_idx = np.argsort(np.abs(row_shap))[::-1][:n]
    return "; ".join(f"{feature_names[i]} ({row_shap[i]:+.3f})" for i in top_idx)


results["top_shap_drivers"] = [top_features_for_row(i) for i in range(len(results))]
print(results[["loan_id", "risk_score", "recommended_action", "top_shap_drivers"]].head(5).to_string(index=False))

# Calibration curve — compares predicted probability against actual default rate
fraction_of_positives, mean_predicted_value = calibration_curve(y_test, probs, n_bins=10)
print(f"Calibration curve — mean predicted probability per bin: {np.round(mean_predicted_value, 4).tolist()}")
print(f"Calibration curve — fraction of actual defaults per bin: {np.round(fraction_of_positives, 4).tolist()}")

plt.figure()
plt.plot(mean_predicted_value, fraction_of_positives, marker="o", label="XGBoost")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of actual defaults")
plt.title("Calibration curve")
plt.legend()
plt.tight_layout()
plt.savefig("../models/calibration_curve.png", dpi=150)
plt.close()
print("saved calibration_curve.png — points below the diagonal mean the model overstates risk there, above means it understates it")

writeback = pd.DataFrame({
    "loan_id": results["loan_id"],
    "model_version": MODEL_VERSION,
    "risk_score": results["risk_score"],
    "probability_of_default": results["probability_of_default"],
    "lgd_assumption": LGD_ASSUMPTION,
    "exposure_at_default": results["exposure_at_default"],
    "expected_loss": results["expected_loss"],
    "prediction_date": date.today(),
    "recommended_action": results["recommended_action"],
})
writeback.to_sql("risk_scores", engine, if_exists="append", index=False)
print(f"wrote {len(writeback):,} rows to risk_scores")

preds = calibrated_model.predict(X_test)
print(
    "\nAccuracy/precision/recall use the default 0.5 threshold; post-calibration "
    "probabilities reflect the true ~20% base rate, so few loans cross 0.5 — "
    "ROC-AUC is the reliable metric here."
)
metrics_row = {
    "model_version": MODEL_VERSION,
    "trained_date": date.today(),
    "accuracy": round(accuracy_score(y_test, preds), 4),
    "precision_score": round(precision_score(y_test, preds), 4),
    "recall_score": round(recall_score(y_test, preds), 4),
    "roc_auc": round(roc_auc_score(y_test, probs), 4),
    "notes": (
        "XGBoost, scale_pos_weight for class imbalance, isotonic calibration, "
        "percentile-based action tiers. Accuracy/precision/recall use 0.5 "
        "threshold, not meaningful post-calibration (~20% base rate) — use ROC-AUC."
    ),
}

# model_version is the primary key, so re-running the pipeline for the same
# version (no retrain) overwrites the row instead of failing on conflict.
from sqlalchemy import text

with engine.begin() as conn:
    conn.execute(text("""
        INSERT INTO model_monitoring
            (model_version, trained_date, accuracy, precision_score, recall_score, roc_auc, notes)
        VALUES
            (:model_version, :trained_date, :accuracy, :precision_score, :recall_score, :roc_auc, :notes)
        ON DUPLICATE KEY UPDATE
            trained_date = VALUES(trained_date),
            accuracy = VALUES(accuracy),
            precision_score = VALUES(precision_score),
            recall_score = VALUES(recall_score),
            roc_auc = VALUES(roc_auc),
            notes = VALUES(notes)
    """), metrics_row)

print(pd.DataFrame([metrics_row]).to_string(index=False))
