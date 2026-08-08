"""
Macro-calibrated stress test.
Fits how portfolio default rate has historically moved with US unemployment,
applies real Fed CCAR/DFAST scenario unemployment paths to shift each loan's
calibrated PD, and recomputes portfolio Expected Loss per scenario.

pip install statsmodels
"""

import os
import pandas as pd
import statsmodels.api as sm
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from datetime import date

load_dotenv()

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "credit_risk_platform")

LGD_ASSUMPTION = 0.45

FRED_UNRATE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"

engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}")

unrate = pd.read_csv(FRED_UNRATE_URL, parse_dates=["observation_date"])
unrate["year"] = unrate["observation_date"].dt.year
annual_unrate = unrate.groupby("year")["UNRATE"].mean().rename("unemployment_rate")

yearly_quality = pd.read_sql("SELECT * FROM yearly_origination_quality", engine)
fit_df = yearly_quality.merge(annual_unrate, left_on="origination_year", right_index=True)
fit_df["default_rate_pp"] = fit_df["default_rate"] * 100

# A naive unemployment-only fit on this data comes back negative (unemployment
# fell 2010-2016 while default rate rose from origination-volume growth, per
# the underwriting drift found in Vintage Analysis) — the two trends are
# confounded. origination_year is added as a linear control so the
# unemployment coefficient reflects unemployment's effect net of that drift.
n_years = len(fit_df)
X = sm.add_constant(fit_df[["unemployment_rate", "origination_year"]])
y = fit_df["default_rate_pp"]
model = sm.OLS(y, X).fit()

sensitivity_slope = model.params["unemployment_rate"]
sensitivity_ci_lower, sensitivity_ci_upper = model.conf_int(alpha=0.05).loc["unemployment_rate"]
sensitivity_r_squared = model.rsquared
trend_slope = model.params["origination_year"]
predictor_corr = fit_df["unemployment_rate"].corr(fit_df["origination_year"])

print(f"Fitted on {n_years} annual cohorts (origination years {fit_df['origination_year'].min()}-{fit_df['origination_year'].max()}), "
      f"{model.df_resid:.0f} degrees of freedom after controlling for the origination-year trend.")
print(f"Unemployment sensitivity: {sensitivity_slope:.3f} pp default rate per 1pp unemployment "
      f"(95% CI: {sensitivity_ci_lower:.3f} to {sensitivity_ci_upper:.3f})")
print(f"Origination-year trend: {trend_slope:+.3f} pp default rate per year — model R²={sensitivity_r_squared:.3f}")
print(f"Correlation between the two predictors: {predictor_corr:.2f}")

# The point estimate isn't statistically distinguishable from zero (CI spans
# zero) and is slightly negative — applying it directly would imply a
# recession *reduces* portfolio risk, which isn't credible and isn't what
# this data actually supports. The upper bound of the 95% CI is the most
# risk-conservative value still consistent with the fit, so scenario shifts
# use that instead of the point estimate. Both are reported to the dashboard.
applied_sensitivity = max(sensitivity_ci_upper, 0)
print(f"Applied sensitivity for scenario shifts: {applied_sensitivity:.3f} (95% CI upper bound, floored at 0)")

# Real Fed-published unemployment paths, expressed as the move (in percentage
# points) each scenario projects. Adverse is sourced from the 2019 cycle,
# the last year the Fed published a separate adverse tier before
# consolidating to baseline/severely-adverse only from 2020 onward.
SCENARIOS = {
    "baseline": {
        "delta_pp": 0.2,
        "source": "Fed 2025 baseline scenario: unemployment to 4.3% (Q1 2025) vs ~4.1% prevailing",
    },
    "adverse": {
        "delta_pp": 3.1,
        "source": "Fed 2019 adverse scenario (last year published separately): peaks at 7% vs ~3.9% start",
    },
    "severely_adverse": {
        "delta_pp": 5.9,
        "source": "Fed 2025 severely adverse scenario: +5.9pp to a peak of 10%",
    },
}

loans = pd.read_sql(
    """
    SELECT rs.loan_id, rs.probability_of_default, rs.exposure_at_default,
           YEAR(l.issue_date) AS origination_year
    FROM risk_scores rs
    JOIN loans l ON l.loan_id = rs.loan_id
    """,
    engine,
)
loans = loans.merge(yearly_quality[["origination_year", "default_rate"]], on="origination_year")

portfolio_avg_default_rate = yearly_quality["n_loans"].mul(yearly_quality["default_rate"]).sum() / yearly_quality["n_loans"].sum()
loans["origination_year_risk_scalar"] = loans["default_rate"] / portfolio_avg_default_rate

print(f"\nPortfolio-wide raw default rate: {portfolio_avg_default_rate:.1%}")

for scenario, params in SCENARIOS.items():
    pd_shift = (applied_sensitivity * params["delta_pp"] / 100) * loans["origination_year_risk_scalar"]
    shifted_pd = (loans["probability_of_default"] + pd_shift).clip(0, 1)
    shifted_el = shifted_pd * LGD_ASSUMPTION * loans["exposure_at_default"]
    params["portfolio_el"] = shifted_el.sum()
    print(f"{scenario}: +{params['delta_pp']}pp unemployment -> portfolio EL ${params['portfolio_el']:,.0f}")

run_date = date.today()
with engine.begin() as conn:
    for scenario, params in SCENARIOS.items():
        conn.execute(text("""
            INSERT INTO stress_test_results
                (scenario, unemployment_delta_pp, scenario_source, portfolio_el,
                 pd_sensitivity_slope, sensitivity_r_squared, sensitivity_ci_lower,
                 sensitivity_ci_upper, applied_sensitivity_pp, n_years_fitted, run_date)
            VALUES
                (:scenario, :delta_pp, :source, :portfolio_el,
                 :slope, :r_squared, :ci_lower, :ci_upper, :applied, :n_years, :run_date)
            ON DUPLICATE KEY UPDATE
                unemployment_delta_pp = VALUES(unemployment_delta_pp),
                scenario_source = VALUES(scenario_source),
                portfolio_el = VALUES(portfolio_el),
                pd_sensitivity_slope = VALUES(pd_sensitivity_slope),
                sensitivity_r_squared = VALUES(sensitivity_r_squared),
                sensitivity_ci_lower = VALUES(sensitivity_ci_lower),
                sensitivity_ci_upper = VALUES(sensitivity_ci_upper),
                applied_sensitivity_pp = VALUES(applied_sensitivity_pp),
                n_years_fitted = VALUES(n_years_fitted),
                run_date = VALUES(run_date)
        """), {
            "scenario": scenario,
            "delta_pp": params["delta_pp"],
            "source": params["source"],
            "portfolio_el": params["portfolio_el"],
            "slope": sensitivity_slope,
            "r_squared": sensitivity_r_squared,
            "ci_lower": sensitivity_ci_lower,
            "ci_upper": sensitivity_ci_upper,
            "applied": applied_sensitivity,
            "n_years": n_years,
            "run_date": run_date,
        })

print(f"\nWrote {len(SCENARIOS)} scenario rows to stress_test_results")
