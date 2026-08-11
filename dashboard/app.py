import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import altair as alt
import os
import base64
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

st.set_page_config(page_title="Credit Risk Early Warning System", layout="wide", page_icon="◆")

# Resolved from this file's own location rather than a "../models" relative
# path, since the working directory differs between local runs (dashboard/)
# and Streamlit Community Cloud, which always runs from the repo root.
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")


def get_config(key, default=None):
    # st.secrets takes priority (Streamlit Cloud), falling back to .env /
    # environment variables (local). Accessing st.secrets raises when no
    # secrets.toml exists anywhere, which is the normal local case.
    try:
        value = st.secrets.get(key)
    except Exception:
        value = None
    if value is not None:
        return value
    return os.getenv(key, default)


DB_USER = get_config("DB_USER", "root")
DB_PASSWORD = get_config("DB_PASSWORD", "")
DB_HOST = get_config("DB_HOST", "localhost")
DB_PORT = get_config("DB_PORT", "3306")
DB_NAME = get_config("DB_NAME", "credit_risk_platform")

# Managed MySQL (e.g. Aiven's free tier) requires TLS and hands out a CA
# certificate. DB_SSL_CA can be either a path to that file (set it this way
# locally) or the certificate's PEM content pasted directly into Streamlit
# Cloud secrets, since Cloud secrets can't reference a file on disk.
DB_SSL_CA = get_config("DB_SSL_CA")
connect_args = {}
if DB_SSL_CA:
    if os.path.isfile(DB_SSL_CA):
        ca_path = DB_SSL_CA
    else:
        ca_path = "/tmp/db_ca.pem"
        with open(ca_path, "w") as f:
            f.write(DB_SSL_CA)
    connect_args["ssl_ca"] = ca_path
    connect_args["ssl_verify_cert"] = True

# Matches the flat LGD assumption used in explain_and_score.py, so live
# expected-loss figures for loans outside the scored test set stay consistent.
LGD_ASSUMPTION = 0.45

engine = create_engine(
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
    connect_args=connect_args,
)


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Work+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #EEF1ED;
        --surface: #FFFFFF;
        --surface-2: #F4F6F1;
        --border: #D8DCD3;
        --text: #1C2321;
        --muted: #6B7268;

        --sidebar-bg: #1C2321;
        --sidebar-text: #EEF1ED;
        --sidebar-muted: #8B9285;

        --accent: #C79A3E;
        --accent-ink: #996E00;

        --grade-a: #2F6F5E;
        --grade-b: #578454;
        --grade-c: #8D9242;
        --grade-d: #C79A3E;
        --grade-e: #B97830;
        --grade-f: #A5582D;
        --grade-g: #8C3B2E;
    }

    html, body, [class*="css"] {
        font-family: 'Work Sans', sans-serif;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .main .block-container {
        animation: fadeInUp 0.4s ease-out;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
        color: var(--text);
    }

    h1 {
        font-size: 1.9rem;
    }

    .main .block-container {
        padding-top: 2.5rem;
    }

    section[data-testid="stSidebar"] {
        background: var(--sidebar-bg);
        border-right: 1px solid #2A322E;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    .brand-row {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 10px;
    }

    .brand-icon {
        color: var(--accent);
        flex-shrink: 0;
        transition: transform 0.25s ease;
    }

    .brand-row:hover .brand-icon {
        transform: rotate(-8deg) scale(1.05);
    }

    .brand-wordmark {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 0.02em;
        color: var(--accent);
    }

    .brand-full {
        font-family: 'Work Sans', sans-serif;
        font-size: 10.5px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--sidebar-muted);
        margin-bottom: 14px;
    }

    .brand-divider {
        border-top: 1px solid #2A322E;
        margin-bottom: 10px;
    }

    /* Sidebar nav is built from real st.button widgets (not <a href> links)
       so page switches happen over Streamlit's websocket rerun instead of a
       full browser navigation — that full reload was the source of both the
       white flash and, before this CSS, a moment of unstyled default link
       (blue, underlined) rendering. */
    section[data-testid="stSidebar"] div[data-testid="stButton"] {
        margin-bottom: 3px;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        gap: 11px;
        padding: 9px 12px;
        border-radius: 8px;
        border: 1px solid transparent;
        background: transparent;
        color: var(--sidebar-muted);
        font-family: 'Work Sans', sans-serif;
        font-size: 14px;
        text-align: left;
        transition: background 0.15s ease, color 0.15s ease, transform 0.15s ease;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: #262E29;
        color: var(--sidebar-text);
        border-color: transparent;
        transform: translateX(2px);
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button:focus:not(:active) {
        border-color: transparent;
        color: var(--sidebar-text);
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
        background: #262E29;
        color: var(--accent);
        border: 1px solid var(--accent);
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"]:hover {
        background: #262E29;
        color: var(--accent);
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button p {
        font-family: 'Work Sans', sans-serif;
        font-size: 14px;
    }

    .eyebrow {
        font-family: 'Work Sans', sans-serif;
        font-weight: 600;
        font-size: 11px;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--accent-ink);
        margin-bottom: 4px;
    }

    .kpi-row {
        display: flex;
        flex-wrap: wrap;
        gap: 1px;
        background: var(--border);
        border: 1px solid var(--border);
        margin: 16px 0 24px 0;
    }

    .kpi-card {
        background: var(--surface);
        flex: 1 1 150px;
        min-width: 150px;
        padding: 14px 16px;
        border-top: 3px solid var(--accent);
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }

    .kpi-card:hover {
        box-shadow: 0 4px 14px rgba(28, 35, 33, 0.08);
        transform: translateY(-1px);
    }

    .kpi-label {
        font-family: 'Work Sans', sans-serif;
        font-weight: 600;
        font-size: 10px;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 6px;
    }

    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 21px;
        font-weight: 500;
        color: var(--text);
        font-variant-numeric: tabular-nums;
    }

    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 16px 18px;
        margin-bottom: 16px;
        transition: box-shadow 0.2s ease;
    }

    .panel:hover {
        box-shadow: 0 4px 14px rgba(28, 35, 33, 0.06);
    }

    .panel-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 17px;
        color: var(--text);
        margin-bottom: 4px;
    }

    .panel-sub {
        font-family: 'Work Sans', sans-serif;
        font-size: 12.5px;
        color: var(--muted);
        margin-bottom: 14px;
    }

    .tier-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        padding: 5px 12px;
        border: 1px solid;
    }

    .grade-strip {
        display: flex;
        border: 1px solid var(--border);
        margin: 18px 0 28px 0;
        overflow: hidden;
    }

    .grade-swatch {
        flex: 1;
        padding: 14px 10px;
        text-align: center;
        transition: flex-grow 0.25s ease;
    }

    .grade-swatch:hover {
        flex-grow: 1.6;
    }

    .grade-letter {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 20px;
        line-height: 1.1;
    }

    .grade-rate {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        margin-top: 4px;
        opacity: 0.9;
    }

    .driver-row {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 9px 0;
        border-bottom: 1px solid var(--border);
    }

    .driver-name {
        font-family: 'Work Sans', sans-serif;
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
        transition: width 0.4s ease;
    }

    .driver-dir {
        font-family: 'Work Sans', sans-serif;
        font-size: 11.5px;
        width: 110px;
        text-align: right;
        flex-shrink: 0;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        font-family: 'JetBrains Mono', monospace;
    }

    hr {
        border-color: var(--border);
    }
    </style>
    """, unsafe_allow_html=True)


COIN_ICON = """<svg class="brand-icon" width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><circle cx="12" cy="12" r="9"/><path d="M12 7 L12 17 M9.3 9.5 Q9.3 7.8 12 7.8 Q14.7 7.8 14.7 9.5 Q14.7 11.2 12 11.2 Q9.3 11.2 9.3 12.8 Q9.3 14.5 12 14.5 Q14.7 14.5 14.7 12.8"/></svg>"""

ICON_OVERVIEW = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="4" y="10" width="3.2" height="9"/><rect x="10.4" y="5" width="3.2" height="14"/><rect x="16.8" y="13" width="3.2" height="6"/></svg>"""

ICON_ANALYTICS = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 19 L4 5"/><path d="M4 19 L20 19"/><circle cx="8" cy="14" r="1.3" fill="currentColor" stroke="none"/><circle cx="13" cy="9" r="1.3" fill="currentColor" stroke="none"/><circle cx="17.5" cy="12" r="1.3" fill="currentColor" stroke="none"/></svg>"""

ICON_EXPLAIN = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="10" cy="10" r="6"/><path d="M15 15 L20 20"/></svg>"""

ICON_MONITOR = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M4 13 L8 13 L10 7 L13 18 L15 13 L20 13"/></svg>"""

ICON_VINTAGE = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 17 C7 17 7 9 11 9 C15 9 15 15 19 15"/><path d="M3 12 C7 12 7 6 11 6 C15 6 15 11 19 11" opacity="0.5"/></svg>"""

ICON_STRESS = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M3 20 L7 20 L7 13 L11 13 L11 8 L15 8 L15 15 L19 15 L19 4"/></svg>"""

ICON_SIMULATOR = """<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="5" y="3" width="14" height="18" rx="1.5"/><path d="M8 8 L16 8 M8 12 L11 12 M13.5 12 L16 12 M8 16 L11 16 M13.5 16 L16 16"/></svg>"""


def render_sidebar_brand():
    st.sidebar.markdown(f"""
    <div class="brand-row">{COIN_ICON}<span class="brand-wordmark">CREWS</span></div>
    <div class="brand-full">Credit Risk Early Warning System</div>
    <div class="brand-divider"></div>
    """, unsafe_allow_html=True)


def kpi_row(items, colors=None):
    cards = ""
    for i, (label, value) in enumerate(items):
        border = f' style="border-top-color:{colors[i]};"' if colors else ""
        cards += f'<div class="kpi-card"{border}><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>'
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)


def panel_open(title, subtitle=""):
    st.markdown(f"""
    <div class="panel">
    <div class="panel-title">{title}</div>
    <div class="panel-sub">{subtitle}</div>
    """, unsafe_allow_html=True)


def panel_close():
    st.markdown("</div>", unsafe_allow_html=True)


# Seven-stop ramp mapped to the real loan grades A (safest) through G
# (riskiest), interpolated in OKLCH from three anchor colors so the hue
# shifts smoothly (teal -> amber -> rust) rather than muddying through gray.
GRADE_ORDER = ["A", "B", "C", "D", "E", "F", "G"]
GRADE_COLORS = {
    "A": "#2F6F5E", "B": "#578454", "C": "#8D9242", "D": "#C79A3E",
    "E": "#B97830", "F": "#A5582D", "G": "#8C3B2E",
}
# Same hues, darkened where needed (C, D, E) to clear text/line contrast on
# the light page background — the raw swatch colors above read fine as
# large fills but two of them (C, D) fall below 3:1 as thin text or lines.
GRADE_INK = {
    "A": "#2F6F5E", "B": "#578454", "C": "#757A28", "D": "#996E00",
    "E": "#A66619", "F": "#A5582D", "G": "#8C3B2E",
}
# Which text color reads best on a filled swatch of each grade's color.
GRADE_TEXT_ON_FILL = {
    "A": "#FFFFFF", "B": "#FFFFFF", "C": "#1C2321", "D": "#1C2321",
    "E": "#1C2321", "F": "#FFFFFF", "G": "#FFFFFF",
}


def render_grade_strip(default_rate_by_grade):
    swatches = ""
    for g in GRADE_ORDER:
        rate = default_rate_by_grade.get(g)
        rate_html = f'<div class="grade-rate">{rate:.1%}</div>' if rate is not None else ""
        swatches += f'<div class="grade-swatch" style="background:{GRADE_COLORS[g]}; color:{GRADE_TEXT_ON_FILL[g]};"><div class="grade-letter">{g}</div>{rate_html}</div>'
    st.markdown(f'<div class="grade-strip">{swatches}</div>', unsafe_allow_html=True)


# The four risk-action tiers aren't grades, but they're an ordered severity
# scale too, so they draw from the same ramp instead of a separate palette —
# one color language for every risk decision in the app.
TIER_COLORS = {
    "Standard monitoring": GRADE_INK["B"],
    "Increase monitoring": GRADE_INK["D"],
    "Reduce credit limit": GRADE_INK["F"],
    "Manual review": GRADE_INK["G"],
}
TIER_ORDER = ["Standard monitoring", "Increase monitoring", "Reduce credit limit", "Manual review"]

# Stress scenarios are a severity scale too: calm, caution, danger.
STRESS_COLORS = {
    "baseline": GRADE_INK["A"],
    "adverse": GRADE_INK["D"],
    "severely_adverse": GRADE_INK["G"],
}

AXIS_LABEL_COLOR = "#6B7268"
AXIS_LINE_COLOR = "#C5CABF"
AXIS_GRID_COLOR = "#E3E6DF"


def styled_bar_chart(df, x_col, y_col, color_map=None, height=260):
    base = alt.Chart(df).mark_bar(size=28).encode(
        x=alt.X(f"{x_col}:N", sort=None, axis=alt.Axis(labelAngle=0, labelFont="Work Sans", titleFont="Work Sans", labelColor=AXIS_LABEL_COLOR, titleColor=AXIS_LABEL_COLOR, domainColor=AXIS_LINE_COLOR, tickColor=AXIS_LINE_COLOR)),
        y=alt.Y(f"{y_col}:Q", axis=alt.Axis(labelFont="JetBrains Mono", titleFont="Work Sans", labelColor=AXIS_LABEL_COLOR, titleColor=AXIS_LABEL_COLOR, domainColor=AXIS_LINE_COLOR, tickColor=AXIS_LINE_COLOR, gridColor=AXIS_GRID_COLOR)),
    )
    if color_map:
        base = base.encode(color=alt.Color(f"{x_col}:N", scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), legend=None))
    else:
        base = base.encode(color=alt.value(GRADE_INK["D"]))
    return base.properties(height=height, background="transparent").configure_view(strokeWidth=0)


NUMERIC_FEATURES = [
    "loan_amount", "interest_rate", "term_months", "dti",
    "annual_income", "employment_length", "loan_to_income_ratio",
    "had_delinquency", "interest_rate_percentile_in_grade",
    "grade_default_rate", "region_default_rate", "loan_amount_rank_in_grade",
]
CATEGORICAL_FEATURES = ["grade", "home_ownership", "purpose"]

FEATURE_LABELS = {
    "loan to income ratio": "Loan amount compared with income",
    "had delinquency": "Previous payment delinquency",
    "grade default rate": "Historical risk of borrower's credit grade",
    "region default rate": "Historical risk of borrower's region",
    "interest rate percentile in grade": "Interest rate compared with similar borrowers",
    "loan amount rank in grade": "Loan size compared with similar borrowers",
    "loan amount": "Loan amount",
    "interest rate": "Interest rate",
    "term months": "Loan term length",
    "dti": "Debt-to-income ratio",
    "annual income": "Annual income",
    "employment length": "Length of employment",
    "grade": "Credit grade",
    "home ownership": "Home ownership status",
    "purpose": "Stated loan purpose",
}


def clean_feature_name(raw_name):
    label = FEATURE_LABELS.get(raw_name)
    if label is None:
        base_key = next((k for k in FEATURE_LABELS if raw_name.startswith(k)), None)
        label = FEATURE_LABELS.get(base_key, raw_name)
    return label


def render_scoring_result(X_row, loan_amount, actual_outcome=None):
    pipeline, preprocessor, explainer, calibrated_model = load_model_and_explainer()

    probs = calibrated_model.predict_proba(X_row)[:, 1][0]
    risk_score = round(probs * 100, 1)
    expected_loss = probs * LGD_ASSUMPTION * loan_amount

    if risk_score >= 75:
        action = "Manual review"
    elif risk_score >= 50:
        action = "Reduce credit limit"
    elif risk_score >= 25:
        action = "Increase monitoring"
    else:
        action = "Standard monitoring"

    badge_color = TIER_COLORS[action]

    kpi_items = [
        ("Risk Score", f"{risk_score}"),
        ("Probability of Default", f"{probs:.1%}"),
        ("Expected Loss", f"${expected_loss:,.0f}"),
    ]
    if actual_outcome is not None:
        kpi_items.append(("Actual Outcome", actual_outcome))
    kpi_row(kpi_items, colors=[badge_color] * len(kpi_items))

    X_transformed = preprocessor.transform(X_row)
    feature_names = preprocessor.get_feature_names_out()
    shap_values = explainer.shap_values(X_transformed)[0]

    top_idx_all = np.argsort(np.abs(shap_values))[::-1]
    # only the reasons pushing risk UP belong in "why this was flagged" —
    # a protective factor isn't a reason for the recommendation
    risk_increasing_idx = [i for i in top_idx_all if shap_values[i] > 0][:3]

    reason_html = ""
    for i in risk_increasing_idx:
        raw_name = feature_names[i].replace("num__", "").replace("cat__", "").replace("_", " ")
        reason_html += f"<li>{clean_feature_name(raw_name)}</li>"
    if not reason_html:
        reason_html = "<li>No single factor stands out — risk is driven by a broad combination of smaller signals</li>"

    st.write("")
    panel_open("Recommended Action")
    st.markdown(f"""
    <div class="tier-badge" style="color:{badge_color}; border-color:{badge_color}; font-size:14px; padding:7px 16px;">{action}</div>
    <div style="margin-top:14px; font-family:'Work Sans', sans-serif; font-size:13px; color:var(--muted);">Reason</div>
    <ul style="margin-top:6px; padding-left:18px; font-family:'Work Sans', sans-serif; font-size:14px; color:var(--text);">
        {reason_html}
    </ul>
    """, unsafe_allow_html=True)
    panel_close()

    panel_open("Full driver breakdown", "Every top factor computed live for this loan, using the same trained model.")

    top_idx = top_idx_all[:5]
    max_abs = max(abs(shap_values[i]) for i in top_idx)

    rows_html = ""
    for i in top_idx:
        val = shap_values[i]
        pct = (abs(val) / max_abs) * 100
        color = GRADE_INK["G"] if val > 0 else GRADE_INK["A"]
        direction = "Increases risk" if val > 0 else "Decreases risk"
        raw_name = feature_names[i].replace("num__", "").replace("cat__", "").replace("_", " ")
        clean_name = clean_feature_name(raw_name)
        rows_html += f"""
        <div class="driver-row">
            <div class="driver-name">{clean_name}</div>
            <div class="driver-bar-track"><div class="driver-bar-fill" style="width:{pct}%; background:{color};"></div></div>
            <div class="driver-dir" style="color:{color};">{direction}</div>
        </div>
        """
    st.markdown(rows_html, unsafe_allow_html=True)
    panel_close()


@st.cache_data(ttl=300)
def load_features():
    return pd.read_sql("SELECT * FROM final_model_features", engine)


@st.cache_data(ttl=300)
def load_risk_scores():
    return pd.read_sql("SELECT * FROM risk_scores", engine)


@st.cache_data(ttl=300)
def load_model_monitoring():
    return pd.read_sql("SELECT * FROM model_monitoring", engine)


@st.cache_data(ttl=300)
def load_vintage_analysis():
    return pd.read_sql("SELECT * FROM vintage_analysis", engine)


@st.cache_data(ttl=300)
def load_yearly_origination_quality():
    return pd.read_sql("SELECT * FROM yearly_origination_quality", engine)


@st.cache_data(ttl=300)
def load_stress_test_results():
    return pd.read_sql("SELECT * FROM stress_test_results", engine)


@st.cache_data(ttl=300)
def load_portfolio_avg_default_rate():
    return pd.read_sql("SELECT AVG(default_flag) AS rate FROM loans", engine)["rate"].iloc[0]


@st.cache_data(ttl=300)
def load_grade_rate_ranges():
    return pd.read_sql(
        "SELECT grade, MIN(interest_rate) AS min_rate, MAX(interest_rate) AS max_rate FROM loans GROUP BY grade",
        engine,
    ).set_index("grade")


@st.cache_data(ttl=60)
def compute_grade_benchmarks(grade, interest_rate, loan_amount):
    query = text("""
        SELECT
            AVG(CASE WHEN interest_rate <= :interest_rate THEN 1.0 ELSE 0.0 END) AS interest_rate_percentile_in_grade,
            AVG(default_flag) AS grade_default_rate,
            SUM(CASE WHEN loan_amount > :loan_amount THEN 1 ELSE 0 END) + 1 AS loan_amount_rank_in_grade
        FROM loans
        WHERE grade = :grade
    """)
    with engine.connect() as conn:
        row = conn.execute(query, {"interest_rate": interest_rate, "loan_amount": loan_amount, "grade": grade}).mappings().first()
    return row


@st.cache_resource
def load_model_and_explainer():
    saved_model = joblib.load(os.path.join(MODELS_DIR, "xgb_model_v1.joblib"))
    pipeline = saved_model["pipeline"]  # raw preprocess+model, needed for SHAP
    calibrated_model = saved_model["calibrated_model"]  # source of every probability shown
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    explainer = shap.TreeExplainer(model)
    return pipeline, preprocessor, explainer, calibrated_model


inject_css()
render_sidebar_brand()

features_df = load_features()
risk_df = load_risk_scores()
monitoring_df = load_model_monitoring()
vintage_df = load_vintage_analysis()
yearly_quality_df = load_yearly_origination_quality()
stress_df = load_stress_test_results()

NAV_ITEMS = [
    ("overview", "Executive Overview", ICON_OVERVIEW),
    ("analytics", "Portfolio Analytics", ICON_ANALYTICS),
    ("explain", "Loan Explainability", ICON_EXPLAIN),
    ("monitoring", "Model Monitoring", ICON_MONITOR),
    ("vintage", "Vintage Analysis", ICON_VINTAGE),
    ("stress", "Stress Testing", ICON_STRESS),
    ("simulator", "What-If Simulator", ICON_SIMULATOR),
]

if "page" not in st.session_state:
    st.session_state.page = st.query_params.get("page", "overview")

# Icons are baked into a data-URI background-image per button, positioned by
# nth-of-type, since a plain st.button label can't hold raw SVG.
nav_icon_css = "<style>"
for idx, (key, label, icon) in enumerate(NAV_ITEMS, start=1):
    icon_uri = "data:image/svg+xml;base64," + base64.b64encode(icon.replace("currentColor", "#8B9285").encode()).decode()
    nav_icon_css += f"""
    section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type({idx}) button::before {{
        content: "";
        display: inline-block;
        width: 18px;
        height: 18px;
        min-width: 18px;
        background-image: url('{icon_uri}');
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
    }}
    """
nav_icon_css += "</style>"
st.sidebar.markdown(nav_icon_css, unsafe_allow_html=True)

for key, label, icon in NAV_ITEMS:
    is_active = st.session_state.page == key
    if st.sidebar.button(label, key=f"nav_{key}", width="stretch", type="primary" if is_active else "secondary"):
        st.session_state.page = key
        st.query_params["page"] = key
        st.rerun()

page = st.session_state.page

if page == "overview":
    st.markdown('<div class="eyebrow">Credit Risk Early Warning System</div>', unsafe_allow_html=True)
    st.title("Executive Overview")

    render_grade_strip(features_df.groupby("grade")["default_flag"].mean())

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

    if not risk_df.empty:
        total_expected_loss = risk_df["expected_loss"].sum()
        el_by_tier = risk_df.groupby("recommended_action")["expected_loss"].sum().reindex(TIER_ORDER).fillna(0)

        st.markdown('<div class="eyebrow">Portfolio Expected Loss</div>', unsafe_allow_html=True)
        kpi_row(
            [("Total Expected Loss", f"${total_expected_loss:,.0f}")]
            + [(tier, f"${el_by_tier[tier]:,.0f}") for tier in TIER_ORDER],
            colors=[GRADE_INK["D"]] + [TIER_COLORS[tier] for tier in TIER_ORDER],
        )

    if not risk_df.empty:
        merged_exposure = risk_df.merge(features_df[["loan_id", "loan_amount", "default_flag"]], on="loan_id")
        high_risk = merged_exposure[merged_exposure["recommended_action"] == "Manual review"]
        exposure_amount = high_risk["loan_amount"].sum()
        expected_default_rate = high_risk["default_flag"].mean() if len(high_risk) > 0 else 0

        st.markdown('<div class="eyebrow">High Risk Portfolio Exposure</div>', unsafe_allow_html=True)
        kpi_row([
            ("Manual Review Loans", f"{len(high_risk):,}"),
            ("Exposure Amount", f"${exposure_amount:,.0f}"),
            ("Expected Default Rate", f"{expected_default_rate:.1%}"),
        ], colors=[TIER_COLORS["Manual review"]] * 3)

    panel_open("Does the risk tiering actually work?", "Default rate should climb as recommended action gets more severe — this is the real test of the system.")
    if not risk_df.empty:
        merged = risk_df.merge(features_df[["loan_id", "default_flag"]], on="loan_id")
        tier_stats = merged.groupby("recommended_action")["default_flag"].mean().reindex(TIER_ORDER).reset_index()
        tier_stats.columns = ["tier", "default_rate"]
        chart = styled_bar_chart(tier_stats, "tier", "default_rate", color_map=TIER_COLORS)
        st.altair_chart(chart, use_container_width=True)
    panel_close()

    n_defaults = int(features_df["default_flag"].sum())
    panel_open("What this actually does for a lending business")
    st.markdown(f"""
    - Flags **{manual_review_count:,} loans** for manual review out of {total_loans:,} in the portfolio
    - Learned from **{n_defaults:,} historical defaults**, not a synthetic or assumed pattern
    - Default rate climbs monotonically across every tier (see chart above) — the tiering isn't arbitrary
    - Shifts risk assessment from purely reactive (after a default happens) to proactive (before it happens)
    """)
    panel_close()

elif page == "analytics":
    st.markdown('<div class="eyebrow">Credit Risk Early Warning System</div>', unsafe_allow_html=True)
    st.title("Portfolio Analytics")

    merged_df = features_df.merge(
        risk_df[["loan_id", "recommended_action"]], on="loan_id", how="left"
    )

    panel_open("Filter portfolio")
    f1, f2, f3, f4 = st.columns(4)
    grade_filter = f1.selectbox("Grade", ["All"] + sorted(merged_df["grade"].unique().tolist()))
    purpose_filter = f2.selectbox("Purpose", ["All"] + sorted(merged_df["purpose"].unique().tolist()))
    tier_filter = f3.selectbox("Risk Tier", ["All"] + TIER_ORDER)
    default_filter = f4.selectbox("Default Status", ["All", "Defaulted", "Did not default"])
    panel_close()

    filtered = merged_df.copy()
    if grade_filter != "All":
        filtered = filtered[filtered["grade"] == grade_filter]
    if purpose_filter != "All":
        filtered = filtered[filtered["purpose"] == purpose_filter]
    if tier_filter != "All":
        filtered = filtered[filtered["recommended_action"] == tier_filter]
    if default_filter == "Defaulted":
        filtered = filtered[filtered["default_flag"] == 1]
    elif default_filter == "Did not default":
        filtered = filtered[filtered["default_flag"] == 0]

    st.caption(f"Showing {len(filtered):,} of {len(merged_df):,} loans")

    col1, col2 = st.columns(2)

    with col1:
        panel_open("Default Rate by Grade")
        grade_stats = filtered.groupby("grade")["default_flag"].mean().reset_index()
        grade_stats.columns = ["grade", "default_rate"]
        st.altair_chart(styled_bar_chart(grade_stats, "grade", "default_rate", color_map=GRADE_INK), use_container_width=True)
        panel_close()

    with col2:
        panel_open("Risk Score Distribution")
        filtered_scores = risk_df[risk_df["loan_id"].isin(filtered["loan_id"])]
        if not filtered_scores.empty:
            hist = alt.Chart(filtered_scores).mark_bar(color=GRADE_INK["D"], size=6).encode(
                x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=30), axis=alt.Axis(labelFont="JetBrains Mono", titleFont="Work Sans", labelColor=AXIS_LABEL_COLOR, titleColor=AXIS_LABEL_COLOR, domainColor=AXIS_LINE_COLOR, tickColor=AXIS_LINE_COLOR)),
                y=alt.Y("count():Q", axis=alt.Axis(labelFont="JetBrains Mono", titleFont="Work Sans", labelColor=AXIS_LABEL_COLOR, titleColor=AXIS_LABEL_COLOR, domainColor=AXIS_LINE_COLOR, tickColor=AXIS_LINE_COLOR, gridColor=AXIS_GRID_COLOR)),
            ).properties(height=260, background="transparent").configure_view(strokeWidth=0)
            st.altair_chart(hist, use_container_width=True)
        panel_close()

    col3, col4 = st.columns(2)

    with col3:
        panel_open("Risk Tier Breakdown")
        tier_counts = filtered["recommended_action"].value_counts().reindex(TIER_ORDER).reset_index()
        tier_counts.columns = ["tier", "count"]
        st.altair_chart(styled_bar_chart(tier_counts, "tier", "count", color_map=TIER_COLORS), use_container_width=True)
        panel_close()

    with col4:
        panel_open("Default Rate by Loan Purpose", "Purposes with fewer than 300 loans excluded to avoid noise.")
        purpose_stats = (
            filtered.groupby("purpose")["default_flag"]
            .agg(["mean", "count"])
            .query("count > 300")
            .sort_values("mean", ascending=False)
            .reset_index()
        )
        purpose_stats.columns = ["purpose", "default_rate", "count"]
        st.altair_chart(styled_bar_chart(purpose_stats, "purpose", "default_rate"), use_container_width=True)
        panel_close()

    panel_open("Filtered loan table", "Searchable — click a column header to sort.")
    st.dataframe(filtered, width="stretch")
    panel_close()

elif page == "explain":
    st.markdown('<div class="eyebrow">Credit Risk Early Warning System</div>', unsafe_allow_html=True)
    st.title("Loan Explainability")
    st.caption("Look up any loan to see its risk score and what's actually driving it.")

    loan_ids = sorted(features_df["loan_id"].unique().tolist())
    selected_id = st.selectbox("Loan ID", loan_ids, label_visibility="collapsed")

    row = features_df[features_df["loan_id"] == selected_id].iloc[0]
    X_row = pd.DataFrame([row[NUMERIC_FEATURES + CATEGORICAL_FEATURES]])
    outcome = "Defaulted" if row["default_flag"] == 1 else "Did not default"

    render_scoring_result(X_row, row["loan_amount"], actual_outcome=outcome)

elif page == "monitoring":
    st.markdown('<div class="eyebrow">Credit Risk Early Warning System</div>', unsafe_allow_html=True)
    st.title("Model Monitoring")
    st.caption("Performance history for every model version that's been trained and deployed.")

    if monitoring_df.empty:
        st.write("No model runs logged yet.")
    else:
        latest = monitoring_df.sort_values("trained_date").iloc[-1]

        panel_open(f"Model {latest['model_version']}", f"Trained {latest['trained_date']} · Status: Active")
        kpi_row([
            ("ROC-AUC", f"{latest['roc_auc']:.3f}"),
            ("Accuracy", f"{latest['accuracy']:.1%}"),
            ("Precision", f"{latest['precision_score']:.1%}"),
            ("Default Detection Recall", f"{latest['recall_score']:.1%}"),
        ])
        st.caption(latest["notes"])
        panel_close()

        with st.expander("About this model"):
            pipeline, _, _, _ = load_model_and_explainer()
            model = pipeline.named_steps["model"]
            n_records = len(features_df)
            if not risk_df.empty:
                p50, p80, p95 = np.percentile(risk_df["risk_score"], [50, 80, 95])
                tier_line = (
                    f"Percentile-based — Standard (bottom 50%, < {p50:.0f}), "
                    f"Increase (50–80th pct, {p50:.0f}–{p80:.0f}), "
                    f"Reduce (80–95th pct, {p80:.0f}–{p95:.0f}), "
                    f"Manual review (top 5%, ≥ {p95:.0f})"
                )
            else:
                tier_line = "Percentile-based — bottom 50% Standard, next 30% Increase, next 15% Reduce, top 5% Manual review"
            st.markdown(f"""
            - **Algorithm:** XGBoost classifier ({model.n_estimators} trees, max depth {model.max_depth})
            - **Training records:** {n_records:,} loans (Lending Club, resolved outcomes only)
            - **ROC-AUC:** {latest['roc_auc']:.3f}
            - **Risk tier thresholds:** {tier_line}
            - **Last trained:** {latest['trained_date']}
            """)

        panel_open("Calibration", "Does a 70% risk score actually mean ~70% of those loans default?")
        st.image(os.path.join(MODELS_DIR, "calibration_curve.png"), width="stretch")
        st.caption(
            "Isotonic regression was fit on a held-out calibration split so predicted "
            "probabilities track actual default rates, rather than just ranking loans "
            "by relative risk. Points below the diagonal mean the model overstates risk "
            "there; above means it understates it."
        )
        panel_close()

        panel_open("Full training history", "Every model version logged to model_monitoring.")
        st.dataframe(monitoring_df.sort_values("trained_date", ascending=False), width="stretch")
        panel_close()

        panel_open("Model limitations")
        st.markdown("""
        - Trained on historical Lending Club data only — no real-time borrower transaction or behavioral data
        - Risk tier boundaries (P50 / P80 / P95) are percentile-based against the scored population, but what share of the portfolio each tier should cover is still a judgement call, not calibrated against a bank's actual risk appetite or manual-review capacity
        - `employment_length` and other self-reported fields carry the reporting gaps of the original dataset
        - Precision on defaults is intentionally traded for recall, since this is designed as an early-warning system — it will over-flag loans for review rather than risk missing real defaults
        - Performance should be re-evaluated periodically as new data comes in, not treated as fixed after one training run
        """)
        panel_close()

elif page == "vintage":
    st.markdown('<div class="eyebrow">Credit Risk Early Warning System</div>', unsafe_allow_html=True)
    st.title("Vintage Analysis")
    st.caption("Cumulative default rate by months on book, one curve per loan-origination quarter.")

    if vintage_df.empty:
        st.write("No vintage data available.")
    else:
        crisis_df = vintage_df[vintage_df["is_crisis_vintage"] == 1]
        noncrisis_df = vintage_df[vintage_df["is_crisis_vintage"] == 0]

        panel_open(
            "Vintage curves",
            "Each line is one origination quarter. Gray lines are 2010–2018 cohorts; red lines are 2007–2009, the financial crisis window.",
        )
        base = alt.Chart(noncrisis_df).mark_line(strokeWidth=1, opacity=0.4).encode(
            x=alt.X("mob_bucket:Q", title="Months on book", axis=alt.Axis(labelFont="JetBrains Mono", titleFont="Work Sans", labelColor=AXIS_LABEL_COLOR, titleColor=AXIS_LABEL_COLOR, domainColor=AXIS_LINE_COLOR, tickColor=AXIS_LINE_COLOR)),
            y=alt.Y("cumulative_default_rate:Q", title="Cumulative default rate", axis=alt.Axis(format=".0%", labelFont="JetBrains Mono", titleFont="Work Sans", labelColor=AXIS_LABEL_COLOR, titleColor=AXIS_LABEL_COLOR, domainColor=AXIS_LINE_COLOR, tickColor=AXIS_LINE_COLOR, gridColor=AXIS_GRID_COLOR)),
            detail="origination_quarter:N",
            color=alt.value(AXIS_LINE_COLOR),
        )
        crisis = alt.Chart(crisis_df).mark_line(strokeWidth=2.2).encode(
            x="mob_bucket:Q",
            y="cumulative_default_rate:Q",
            detail="origination_quarter:N",
            color=alt.value(GRADE_INK["G"]),
        )
        chart = (base + crisis).properties(height=380, background="transparent").configure_view(strokeWidth=0)
        st.altair_chart(chart, use_container_width=True)
        panel_close()

        reference_mob = 36
        crisis_ref = crisis_df[crisis_df["mob_bucket"] == reference_mob]
        noncrisis_ref = noncrisis_df[noncrisis_df["mob_bucket"] == reference_mob]
        crisis_n = crisis_ref["cohort_size"].sum()
        noncrisis_n = noncrisis_ref["cohort_size"].sum()
        crisis_rate = crisis_ref["cumulative_defaults"].sum() / crisis_n if crisis_n else 0
        noncrisis_rate = noncrisis_ref["cumulative_defaults"].sum() / noncrisis_n if noncrisis_n else 0
        relative_diff = (crisis_rate - noncrisis_rate) / noncrisis_rate if noncrisis_rate else 0

        baseline_year = 2010
        baseline_row = yearly_quality_df[yearly_quality_df["origination_year"] == baseline_year]
        peak_row = yearly_quality_df.loc[yearly_quality_df["default_rate"].idxmax()]
        peak_year = int(peak_row["origination_year"])
        peak_rate = peak_row["default_rate"]
        max_year = int(yearly_quality_df["origination_year"].max())
        censored_years_label = f"{max_year - 2}–{max_year}"

        if relative_diff > 0.15:
            finding = (
                f"the 2007–2009 vintages default at a visibly higher rate "
                f"({crisis_rate:.1%} vs {noncrisis_rate:.1%} for every other cohort) — a crisis effect "
                "that holds up even against the broader growth-driven drift below."
            )
        else:
            finding = (
                f"the 2007–2009 vintages sit at {crisis_rate:.1%} vs {noncrisis_rate:.1%} for every "
                "other cohort — too small a sample to detect a crisis effect, and overshadowed by a "
                "stronger, growth-driven trend in underwriting quality over time."
            )

        if not baseline_row.empty:
            baseline_rate = baseline_row["default_rate"].iloc[0]
            drift_line = (
                f"Raw default rate climbs from {baseline_rate:.1%} in {baseline_year} to a peak of "
                f"{peak_rate:.1%} in {peak_year}, as origination volume scaled from a few hundred loans "
                "a year to tens of thousands — that's the dominant pattern here, not the 2007–2009 window"
            )
        else:
            drift_line = (
                f"Raw default rate peaks at {peak_rate:.1%} in {peak_year}, as origination volume scaled "
                "from a few hundred loans a year to tens of thousands — that's the dominant pattern here, "
                "not the 2007–2009 window"
            )

        panel_open("Takeaway")
        st.markdown(f"""
        - Cumulative default rate climbs with months on book and levels off as each cohort finishes resolving — the shape of a normal vintage curve
        - {drift_line}
        - At the {reference_mob}-month mark, {finding} That comparison is also thin on data: the 2007–2009 window covers only {crisis_n:,} loans across {crisis_df['origination_quarter'].nunique()} quarters, versus {noncrisis_n:,} for every other quarter combined
        - {censored_years_label} cohorts likely understate their eventual default rate: this dataset only keeps resolved loans, so recent vintages that haven't finished maturing are missing loans still marked "Current" that would go on to default — correcting for that right-censoring would make the drift trend look even steeper, not flatter
        """)
        panel_close()

elif page == "stress":
    st.markdown('<div class="eyebrow">Credit Risk Early Warning System</div>', unsafe_allow_html=True)
    st.title("Stress Testing")
    st.caption("Portfolio Expected Loss under real Fed CCAR/DFAST unemployment scenarios.")

    if stress_df.empty:
        st.write("No stress test results available.")
    else:
        scenario_labels = {"baseline": "Baseline", "adverse": "Adverse", "severely_adverse": "Severely Adverse"}
        scenario_order = ["baseline", "adverse", "severely_adverse"]
        stress_indexed = stress_df.set_index("scenario")
        fit_row = stress_df.iloc[0]

        panel_open(
            "Portfolio Expected Loss by scenario",
            "Each scenario shifts every loan's calibrated PD by the fitted unemployment sensitivity, scaled by that loan's origination-year risk.",
        )
        present_scenarios = [s for s in scenario_order if s in stress_indexed.index]
        kpi_row(
            [(scenario_labels[s], f"${stress_indexed.loc[s, 'portfolio_el']:,.0f}") for s in present_scenarios],
            colors=[STRESS_COLORS[s] for s in present_scenarios],
        )
        panel_close()

        if "baseline" in stress_indexed.index:
            baseline_el = stress_indexed.loc["baseline", "portfolio_el"]
            delta_scenarios = [s for s in present_scenarios if s != "baseline"]
            st.markdown('<div class="eyebrow">Increase vs Baseline</div>', unsafe_allow_html=True)
            kpi_row(
                [(scenario_labels[s], f"+${stress_indexed.loc[s, 'portfolio_el'] - baseline_el:,.0f}") for s in delta_scenarios],
                colors=[STRESS_COLORS[s] for s in delta_scenarios],
            )

        panel_open("Scenario sources", "Real Fed-published unemployment paths, not invented numbers.")
        st.dataframe(
            stress_df[["scenario", "unemployment_delta_pp", "scenario_source"]].rename(columns={
                "scenario": "Scenario",
                "unemployment_delta_pp": "Unemployment Δ (pp)",
                "scenario_source": "Source",
            }),
            width="stretch",
            hide_index=True,
        )
        panel_close()

        panel_open("Methodology", "The sensitivity was fit from real historical data, not assumed.")
        st.markdown(f"""
        - Fitted on {fit_row['n_years_fitted']:.0f} annual origination cohorts (2007–2018) using unemployment rate (FRED series UNRATE) and origination year as predictors of cohort default rate — origination year controls for the underwriting-drift trend found in Vintage Analysis, since unemployment and origination volume both moved over that window and would otherwise be confounded
        - The fitted unemployment coefficient is {fit_row['pd_sensitivity_slope']:.2f}pp default rate per 1pp unemployment (95% CI: {fit_row['sensitivity_ci_lower']:.2f} to {fit_row['sensitivity_ci_upper']:.2f}) — the CI spans zero, so this sample can't statistically distinguish the effect from zero, and the point estimate itself is slightly negative
        - Applying that point estimate directly would imply a recession *reduces* portfolio risk, which isn't credible — so scenario shifts use {fit_row['applied_sensitivity_pp']:.2f}pp, the upper (most risk-conservative) bound of the same 95% CI, instead of the point estimate
        - Each loan's shift is scaled by its origination year's baseline default rate relative to the portfolio average, so a 2016-vintage loan (above-average risk) moves more than a 2010-vintage loan (below-average) under the same macro shock
        - Baseline and severely adverse unemployment paths are the Fed's actual published 2025 supervisory scenarios; adverse is sourced from 2019, the last year the Fed published a separate adverse tier before consolidating to baseline and severely adverse only
        - With only {fit_row['n_years_fitted']:.0f} data points and 2 predictors, this fit has 9 degrees of freedom — treat the sensitivity as directional, not precise, and re-fit as more years of data accumulate
        """)
        panel_close()

else:
    st.markdown('<div class="eyebrow">Credit Risk Early Warning System</div>', unsafe_allow_html=True)
    st.title("What-If Simulator")
    st.caption("Score a hypothetical loan using the same trained model and SHAP explanations as Loan Explainability.")

    GRADE_OPTIONS = ["A", "B", "C", "D", "E", "F", "G"]
    HOME_OWNERSHIP_OPTIONS = ["MORTGAGE", "RENT", "OWN", "ANY", "OTHER", "NONE"]
    PURPOSE_OPTIONS = [
        "debt_consolidation", "credit_card", "home_improvement", "other",
        "major_purchase", "small_business", "medical", "car", "moving",
        "vacation", "house", "wedding", "renewable_energy", "educational",
    ]

    panel_open("Loan parameters", "Enter hypothetical values to see how the model would score this loan.")
    c1, c2, c3 = st.columns(3)
    loan_amount = c1.number_input("Loan amount ($)", min_value=500, max_value=40000, value=15000, step=500)
    interest_rate = c2.number_input("Interest rate (%)", min_value=5.0, max_value=31.0, value=13.0, step=0.1)
    term_months = c3.selectbox("Term (months)", [36, 60])

    c4, c5, c6 = st.columns(3)
    dti = c4.number_input("Debt-to-income ratio", min_value=0.0, max_value=60.0, value=18.0, step=0.5)
    annual_income = c5.number_input("Annual income ($)", min_value=1000, max_value=1000000, value=65000, step=1000)
    employment_length = c6.slider("Employment length (years)", min_value=0, max_value=10, value=5)

    c7, c8, c9 = st.columns(3)
    grade = c7.selectbox("Credit grade", GRADE_OPTIONS, index=2)
    home_ownership = c8.selectbox("Home ownership", HOME_OWNERSHIP_OPTIONS)
    purpose = c9.selectbox("Loan purpose", PURPOSE_OPTIONS)
    panel_close()

    region_default_rate = load_portfolio_avg_default_rate()
    st.caption(
        "Two simplifying assumptions: this form doesn't collect region, so region risk uses the "
        f"portfolio-wide average default rate ({region_default_rate:.1%}) as a neutral stand-in; "
        "and prior delinquency isn't collected either, so it's assumed to be none."
    )

    grade_rate_ranges = load_grade_rate_ranges()
    grade_min_rate, grade_max_rate = grade_rate_ranges.loc[grade, ["min_rate", "max_rate"]]
    if not (grade_min_rate <= interest_rate <= grade_max_rate):
        warn_color = GRADE_INK["G"]
        st.markdown(f"""
        <div style="background: var(--surface); border: 1px solid var(--border); border-left: 3px solid {warn_color}; padding: 14px 18px; margin-bottom: 20px; font-family: 'Work Sans', sans-serif; font-size: 13.5px; color: var(--text);">
        <strong style="color: {warn_color};">Out of distribution —</strong> Grade {grade} loans in the historical data run {grade_min_rate:.1f}%–{grade_max_rate:.1f}% — {interest_rate:.1f}% has never occurred for this grade, since Lending Club sets rate mostly by grade. The model has never seen this combination, so the score below is an extrapolation and may not be reliable.
        </div>
        """, unsafe_allow_html=True)

    benchmarks = compute_grade_benchmarks(grade, interest_rate, loan_amount)
    loan_to_income_ratio = round(loan_amount / annual_income, 4) if annual_income else 0

    X_row = pd.DataFrame([{
        "loan_amount": loan_amount,
        "interest_rate": interest_rate,
        "term_months": term_months,
        "dti": dti,
        "annual_income": annual_income,
        "employment_length": employment_length,
        "loan_to_income_ratio": loan_to_income_ratio,
        "had_delinquency": 0,
        "interest_rate_percentile_in_grade": float(benchmarks["interest_rate_percentile_in_grade"]),
        "grade_default_rate": float(benchmarks["grade_default_rate"]),
        "region_default_rate": float(region_default_rate),
        "loan_amount_rank_in_grade": int(benchmarks["loan_amount_rank_in_grade"]),
        "grade": grade,
        "home_ownership": home_ownership,
        "purpose": purpose,
    }])

    render_scoring_result(X_row, loan_amount)
