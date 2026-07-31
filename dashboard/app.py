import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import altair as alt
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Credit Risk Decision Intelligence Platform", layout="wide", page_icon="◆")

DB_USER = "root"
DB_PASSWORD = os.getenv("DB_PASSWORD")  
DB_HOST = "localhost"
DB_NAME = "credit_risk_platform"

engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #0B1220;
        --surface: #121A2B;
        --surface-2: #182036;
        --border: #26314A;
        --text: #E8ECF4;
        --muted: #8B96AC;
        --gold: #C9A24B;
        --gold-dim: #8A7238;
        --risk-low: #4C8B6C;
        --risk-mid: #C9A24B;
        --risk-high: #C1543C;
        --risk-severe: #8B2E20;
    }

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    h1, h2, h3 {
        font-family: 'Fraunces', serif;
        font-weight: 500;
        letter-spacing: -0.01em;
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    .brand-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 10px;
    }

    .brand-icon {
        color: var(--gold);
        flex-shrink: 0;
    }

    .brand-wordmark {
        font-family: 'Fraunces', serif;
        font-size: 22px;
        font-weight: 500;
        letter-spacing: 0.04em;
        color: var(--gold);
    }

    .brand-full {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10.5px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 10px;
    }

    .brand-name {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 11.5px;
        color: var(--muted);
        margin-bottom: 18px;
    }

    .brand-divider {
        border-top: 1px solid var(--border);
        margin-bottom: 10px;
    }

    .nav-link {
        display: flex;
        align-items: center;
        gap: 11px;
        padding: 9px 12px;
        margin-bottom: 3px;
        border-radius: 8px;
        text-decoration: none;
        color: var(--muted);
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 14px;
        transition: background 0.15s ease;
    }

    .nav-link:hover {
        background: var(--surface-2);
        color: var(--text);
    }

    .nav-link-active {
        background: var(--surface-2);
        color: var(--gold);
        border-left: 2px solid var(--gold);
        padding-left: 10px;
    }

    .nav-link svg {
        flex-shrink: 0;
    }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--gold-dim);
        margin-bottom: 4px;
    }

    .kpi-row {
        display: flex;
        gap: 1px;
        background: var(--border);
        border: 1px solid var(--border);
        margin: 18px 0 28px 0;
    }

    .kpi-card {
        background: var(--surface);
        flex: 1;
        padding: 18px 22px;
        border-top: 2px solid var(--gold);
    }

    .kpi-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10.5px;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 8px;
    }

    .kpi-value {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 28px;
        font-weight: 500;
        color: var(--text);
        font-variant-numeric: tabular-nums;
    }

    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 20px 22px;
        margin-bottom: 20px;
    }

    .panel-title {
        font-family: 'Fraunces', serif;
        font-size: 17px;
        color: var(--text);
        margin-bottom: 4px;
    }

    .panel-sub {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 12.5px;
        color: var(--muted);
        margin-bottom: 14px;
    }

    .tier-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 5px 12px;
        border: 1px solid;
    }

    .driver-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 9px 0;
        border-bottom: 1px solid var(--border);
    }

    .driver-name {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12.5px;
        color: var(--text);
        width: 260px;
        flex-shrink: 0;
    }

    .driver-bar-track {
        flex: 1;
        height: 6px;
        background: var(--surface-2);
        position: relative;
    }

    .driver-bar-fill {
        height: 100%;
    }

    .driver-dir {
        font-family: 'IBM Plex Sans', sans-serif;
        font-size: 11.5px;
        width: 110px;
        text-align: right;
        flex-shrink: 0;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
    }

    hr {
        border-color: var(--border);
    }
    </style>
    """, unsafe_allow_html=True)


COIN_ICON = """<svg class="brand-icon" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.4"><circle cx="12" cy="12" r="9"/><path d="M12 7 L12 17 M9.3 9.5 Q9.3 7.8 12 7.8 Q14.7 7.8 14.7 9.5 Q14.7 11.2 12 11.2 Q9.3 11.2 9.3 12.8 Q9.3 14.5 12 14.5 Q14.7 14.5 14.7 12.8"/></svg>"""

ICON_OVERVIEW = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="4" y="10" width="3.2" height="9"/><rect x="10.4" y="5" width="3.2" height="14"/><rect x="16.8" y="13" width="3.2" height="6"/></svg>"""

ICON_ANALYTICS = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 19 L4 5"/><path d="M4 19 L20 19"/><circle cx="8" cy="14" r="1.3" fill="currentColor" stroke="none"/><circle cx="13" cy="9" r="1.3" fill="currentColor" stroke="none"/><circle cx="17.5" cy="12" r="1.3" fill="currentColor" stroke="none"/></svg>"""

ICON_EXPLAIN = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="10" cy="10" r="6"/><path d="M15 15 L20 20"/></svg>"""


def render_sidebar_brand():
    st.sidebar.markdown(f"""
    <div class="brand-row">{COIN_ICON}<span class="brand-wordmark">CRIP</span></div>
    <div class="brand-full">Credit Risk Intelligence Platform</div>
    <div class="brand-name">Nurmuhammad Nazmi</div>
    <div class="brand-divider"></div>
    """, unsafe_allow_html=True)


def kpi_row(items):
    cards = "".join(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)


def panel_open(title, subtitle=""):
    st.markdown(f"""
    <div class="panel">
    <div class="panel-title">{title}</div>
    <div class="panel-sub">{subtitle}</div>
    """, unsafe_allow_html=True)


def panel_close():
    st.markdown("</div>", unsafe_allow_html=True)


TIER_COLORS = {
    "Standard monitoring": "#4C8B6C",
    "Increase monitoring": "#C9A24B",
    "Reduce credit limit": "#C1543C",
    "Manual review": "#8B2E20",
}
TIER_ORDER = ["Standard monitoring", "Increase monitoring", "Reduce credit limit", "Manual review"]


def styled_bar_chart(df, x_col, y_col, color_map=None, height=260):
    base = alt.Chart(df).mark_bar(size=28).encode(
        x=alt.X(f"{x_col}:N", sort=None, axis=alt.Axis(labelAngle=0, labelColor="#8B96AC", titleColor="#8B96AC", domainColor="#26314A", tickColor="#26314A")),
        y=alt.Y(f"{y_col}:Q", axis=alt.Axis(labelColor="#8B96AC", titleColor="#8B96AC", domainColor="#26314A", tickColor="#26314A", gridColor="#182036")),
    )
    if color_map:
        base = base.encode(color=alt.Color(f"{x_col}:N", scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), legend=None))
    else:
        base = base.encode(color=alt.value("#C9A24B"))
    return base.properties(height=height, background="transparent").configure_view(strokeWidth=0)


@st.cache_data(ttl=300)
def load_features():
    return pd.read_sql("SELECT * FROM final_model_features", engine)


@st.cache_data(ttl=300)
def load_risk_scores():
    return pd.read_sql("SELECT * FROM risk_scores", engine)


@st.cache_data(ttl=300)
def load_model_monitoring():
    return pd.read_sql("SELECT * FROM model_monitoring", engine)


@st.cache_resource
def load_model_and_explainer():
    pipeline = joblib.load("../models/xgb_model_v1.joblib")
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    explainer = shap.TreeExplainer(model)
    return pipeline, preprocessor, explainer


inject_css()
render_sidebar_brand()

features_df = load_features()
risk_df = load_risk_scores()
monitoring_df = load_model_monitoring()

NAV_ITEMS = [
    ("overview", "Executive Overview", ICON_OVERVIEW),
    ("analytics", "Portfolio Analytics", ICON_ANALYTICS),
    ("explain", "Loan Explainability", ICON_EXPLAIN),
]

current_page = st.query_params.get("page", "overview")

nav_html = ""
for key, label, icon in NAV_ITEMS:
    css_class = "nav-link nav-link-active" if key == current_page else "nav-link"
    nav_html += f'<a href="?page={key}" target="_self" class="{css_class}">{icon}<span>{label}</span></a>'
st.sidebar.markdown(nav_html, unsafe_allow_html=True)

page = current_page

if page == "overview":
    st.markdown('<div class="eyebrow">Credit Risk Decision Intelligence Platform</div>', unsafe_allow_html=True)
    st.title("Executive Overview")

    total_loans = len(features_df)
    default_rate = features_df["default_flag"].mean()
    avg_risk_score = risk_df["risk_score"].mean() if not risk_df.empty else np.nan
    manual_review_count = (risk_df["recommended_action"] == "Manual review").sum() if not risk_df.empty else 0

    kpi_row([
        ("Total Loans", f"{total_loans:,}"),
        ("Default Rate", f"{default_rate:.1%}"),
        ("Avg Risk Score", f"{avg_risk_score:.1f}"),
        ("Manual Review", f"{manual_review_count:,}"),
    ])

    if not monitoring_df.empty:
        latest = monitoring_df.sort_values("trained_date").iloc[-1]
        st.markdown('<div class="eyebrow">Current Model — v' + str(latest["model_version"]) + '</div>', unsafe_allow_html=True)
        kpi_row([
            ("Accuracy", f"{latest['accuracy']:.1%}"),
            ("Precision", f"{latest['precision_score']:.1%}"),
            ("Recall", f"{latest['recall_score']:.1%}"),
            ("ROC-AUC", f"{latest['roc_auc']:.3f}"),
        ])

    panel_open("Does the risk tiering actually work?", "Default rate should climb as recommended action gets more severe — this is the real test of the system.")
    if not risk_df.empty:
        merged = risk_df.merge(features_df[["loan_id", "default_flag"]], on="loan_id")
        tier_stats = merged.groupby("recommended_action")["default_flag"].mean().reindex(TIER_ORDER).reset_index()
        tier_stats.columns = ["tier", "default_rate"]
        chart = styled_bar_chart(tier_stats, "tier", "default_rate", color_map=TIER_COLORS)
        st.altair_chart(chart, use_container_width=True)
    panel_close()

elif page == "analytics":
    st.markdown('<div class="eyebrow">Credit Risk Decision Intelligence Platform</div>', unsafe_allow_html=True)
    st.title("Portfolio Analytics")

    col1, col2 = st.columns(2)

    with col1:
        panel_open("Default Rate by Grade")
        grade_stats = features_df.groupby("grade")["default_flag"].mean().reset_index()
        grade_stats.columns = ["grade", "default_rate"]
        st.altair_chart(styled_bar_chart(grade_stats, "grade", "default_rate"), use_container_width=True)
        panel_close()

    with col2:
        panel_open("Risk Score Distribution")
        if not risk_df.empty:
            hist = alt.Chart(risk_df).mark_bar(color="#C9A24B", size=6).encode(
                x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=30), axis=alt.Axis(labelColor="#8B96AC", titleColor="#8B96AC", domainColor="#26314A", tickColor="#26314A")),
                y=alt.Y("count():Q", axis=alt.Axis(labelColor="#8B96AC", titleColor="#8B96AC", domainColor="#26314A", tickColor="#26314A", gridColor="#182036")),
            ).properties(height=260, background="transparent").configure_view(strokeWidth=0)
            st.altair_chart(hist, use_container_width=True)
        panel_close()

    col3, col4 = st.columns(2)

    with col3:
        panel_open("Risk Tier Breakdown")
        if not risk_df.empty:
            tier_counts = risk_df["recommended_action"].value_counts().reindex(TIER_ORDER).reset_index()
            tier_counts.columns = ["tier", "count"]
            st.altair_chart(styled_bar_chart(tier_counts, "tier", "count", color_map=TIER_COLORS), use_container_width=True)
        panel_close()

    with col4:
        panel_open("Default Rate by Loan Purpose", "Purposes with fewer than 300 loans excluded to avoid noise.")
        purpose_stats = (
            features_df.groupby("purpose")["default_flag"]
            .agg(["mean", "count"])
            .query("count > 300")
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        purpose_stats.columns = ["purpose", "default_rate", "count"]
        st.altair_chart(styled_bar_chart(purpose_stats, "purpose", "default_rate"), use_container_width=True)
        panel_close()

    panel_open("Full feature table", "Searchable — click a column header to sort.")
    st.dataframe(features_df, use_container_width=True)
    panel_close()

else:
    st.markdown('<div class="eyebrow">Credit Risk Decision Intelligence Platform</div>', unsafe_allow_html=True)
    st.title("Loan Explainability")
    st.caption("Look up any loan to see its risk score and what's actually driving it.")

    loan_ids = sorted(features_df["loan_id"].unique().tolist())
    selected_id = st.selectbox("Loan ID", loan_ids, label_visibility="collapsed")

    row = features_df[features_df["loan_id"] == selected_id].iloc[0]

    pipeline, preprocessor, explainer = load_model_and_explainer()

    numeric_features = [
        "loan_amount", "interest_rate", "term_months", "dti",
        "annual_income", "employment_length", "loan_to_income_ratio",
        "had_delinquency", "interest_rate_percentile_in_grade",
        "grade_default_rate", "region_default_rate", "loan_amount_rank_in_grade",
    ]
    categorical_features = ["grade", "home_ownership", "purpose"]
    X_row = pd.DataFrame([row[numeric_features + categorical_features]])

    probs = pipeline.predict_proba(X_row)[:, 1][0]
    risk_score = round(probs * 100, 1)

    if risk_score >= 75:
        action = "Manual review"
    elif risk_score >= 50:
        action = "Reduce credit limit"
    elif risk_score >= 25:
        action = "Increase monitoring"
    else:
        action = "Standard monitoring"

    badge_color = TIER_COLORS[action]
    outcome = "Defaulted" if row["default_flag"] == 1 else "Did not default"

    kpi_row([
        ("Risk Score", f"{risk_score}"),
        ("Actual Outcome", outcome),
    ])

    st.markdown(
        f'<div class="tier-badge" style="color:{badge_color}; border-color:{badge_color};">{action}</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    panel_open("What's driving this score", "SHAP values computed live for this loan, using the same trained model.")

    X_transformed = preprocessor.transform(X_row)
    feature_names = preprocessor.get_feature_names_out()
    shap_values = explainer.shap_values(X_transformed)[0]

    top_idx = np.argsort(np.abs(shap_values))[::-1][:5]
    max_abs = max(abs(shap_values[i]) for i in top_idx)

    rows_html = ""
    for i in top_idx:
        val = shap_values[i]
        pct = (abs(val) / max_abs) * 100
        color = "#C1543C" if val > 0 else "#4C8B6C"
        direction = "Increases risk" if val > 0 else "Decreases risk"
        clean_name = feature_names[i].replace("num__", "").replace("cat__", "").replace("_", " ")
        rows_html += f"""
        <div class="driver-row">
            <div class="driver-name">{clean_name}</div>
            <div class="driver-bar-track"><div class="driver-bar-fill" style="width:{pct}%; background:{color};"></div></div>
            <div class="driver-dir" style="color:{color};">{direction}</div>
        </div>
        """
    st.markdown(rows_html, unsafe_allow_html=True)
    panel_close()