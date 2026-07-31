import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from datetime import date

# CONFIG
DB_USER = "root"
DB_PASSWORD = "***REDACTED-ROTATED***"
DB_HOST = "localhost"
DB_NAME = "credit_risk_platform"
MODEL_VERSION = "v1"

engine = create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
)


print("=" * 60)
print("STAGE 1: Loading saved model + test set")
print("=" * 60)

xgb_pipeline = joblib.load(f"../models/xgb_model_{MODEL_VERSION}.joblib")
saved = joblib.load("../models/test_set.joblib")
X_test, y_test, ids_test = saved["X_test"], saved["y_test"], saved["ids_test"]
print(f"Loaded model and test set ({len(X_test):,} rows).")


# PREDICTIONS -> RISK SCORE (0-100 scale)

print("\n" + "=" * 60)
print("STAGE 2: Generating risk scores")
print("=" * 60)

probs = xgb_pipeline.predict_proba(X_test)[:, 1]
risk_scores = (probs * 100).round(2)

# Decision layer: risk score -> recommended action
def recommend_action(score):
    if score >= 75:
        return "Manual review"
    elif score >= 50:
        return "Reduce credit limit"
    elif score >= 25:
        return "Increase monitoring"
    else:
        return "Standard monitoring"

recommended_actions = [recommend_action(s) for s in risk_scores]

results = pd.DataFrame({
    "loan_id": ids_test.values,
    "risk_score": risk_scores,
    "recommended_action": recommended_actions,
    "actual_default": y_test.values,
})

print("\nRecommended action distribution:")
print(results["recommended_action"].value_counts())

print("\nSanity check — actual default rate by recommended action "
      "(should increase as action gets more severe):")
print(results.groupby("recommended_action")["actual_default"].mean().round(3))

# SHAP EXPLAINABILITY

print("\n" + "=" * 60)
print("STAGE 3: SHAP explainability")
print("=" * 60)

# Pull out the fitted preprocessor + model from the pipeline
preprocessor = xgb_pipeline.named_steps["preprocess"]
model = xgb_pipeline.named_steps["model"]

X_test_transformed = preprocessor.transform(X_test)
feature_names = preprocessor.get_feature_names_out()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_transformed)

# Global feature importance plot — saved for your README / Power BI page
plt.figure()
shap.summary_plot(
    shap_values, X_test_transformed, feature_names=feature_names,
    show=False, plot_type="bar"
)
plt.tight_layout()
plt.savefig("../models/shap_feature_importance.png", dpi=150)
plt.close()
print("Saved SHAP global feature importance plot to ../models/shap_feature_importance.png")


def top_features_for_row(row_idx, n=3):
    row_shap = shap_values[row_idx]
    top_idx = np.argsort(np.abs(row_shap))[::-1][:n]
    return "; ".join(f"{feature_names[i]} ({row_shap[i]:+.3f})" for i in top_idx)

results["top_shap_drivers"] = [top_features_for_row(i) for i in range(len(results))]

print("\nExample explained predictions:")
print(results[["loan_id", "risk_score", "recommended_action", "top_shap_drivers"]].head(5).to_string(index=False))

# WRITE risk_scores BACK TO MYSQL
print("\n" + "=" * 60)
print("STAGE 4: Writing risk_scores to MySQL")
print("=" * 60)

writeback = pd.DataFrame({
    "loan_id": results["loan_id"],
    "model_version": MODEL_VERSION,
    "risk_score": results["risk_score"],
    "prediction_date": date.today(),
    "recommended_action": results["recommended_action"],
})
writeback.to_sql("risk_scores", engine, if_exists="append", index=False)
print(f"Wrote {len(writeback):,} rows to risk_scores.")

# WRITE MODEL METRICS TO model_monitoring
print("\n" + "=" * 60)
print("STAGE 5: Writing model metrics to model_monitoring")
print("=" * 60)

preds = xgb_pipeline.predict(X_test)
metrics = pd.DataFrame([{
    "model_version": MODEL_VERSION,
    "trained_date": date.today(),
    "accuracy": round(accuracy_score(y_test, preds), 4),
    "precision_score": round(precision_score(y_test, preds), 4),
    "recall_score": round(recall_score(y_test, preds), 4),
    "roc_auc": round(roc_auc_score(y_test, probs), 4),
    "notes": "XGBoost, scale_pos_weight for class imbalance, SHAP explainability added",
}])
metrics.to_sql("model_monitoring", engine, if_exists="append", index=False)
print("Wrote model metrics:")
print(metrics.to_string(index=False))

print("\nDone. Day 4 complete — risk_scores and model_monitoring are now")
print("populated and ready for the Power BI dashboard on Day 5.")
