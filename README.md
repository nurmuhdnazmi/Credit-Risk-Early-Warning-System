# Credit Risk Early Warning System

## A model told me a loan had an 80% chance of default. It was wrong.

Early in this project, the model was flagging loans as extremely high risk at a rate that didn't match reality. A calibration curve made it visible: at the high end of the scale, predicted risk was running well ahead of how often those loans actually defaulted. The model wasn't broken. It had been trained with a technique that improves classification at the cost of honest probabilities, and nobody had checked. The fix, and the exact numbers proving it worked, are below.

Fixing that, and the string of similar findings that followed it, is what this project actually is. Not a notebook that trains XGBoost and reports an accuracy score, but a working pipeline that tries to get the numbers right, catches itself when they aren't, and says so.

## Why this project

The objective was never to maximize predictive accuracy. It was to build a risk system where a probability, a risk tier, and an expected loss estimate could all be traced back to the underlying data, and challenged the moment the assumptions behind them stopped holding. Most of the interesting work here isn't the model itself. It's everything that happens after the model produces a number and someone has to decide whether to trust it.

## Key results

| | |
|---|---:|
| Loans analyzed | 75,000 |
| Base default rate | 20.3% |
| ROC-AUC | 0.708 |
| Recall on defaults | 63% |
| Expected Loss before calibration fix | $45.4M |
| Expected Loss after calibration fix | $21.1M |
| SQL query speedup from indexing | 64% |
| Dashboard pages | 7 |

## Dashboard

Screenshots go here once the cosmetic pass is finished.

### Executive Overview
`models/screenshot_overview.png`

### Vintage Analysis
`models/screenshot_vintage.png`

### Stress Testing
`models/screenshot_stress.png`

### What-If Loan Simulator
`models/screenshot_simulator.png`

## What it does

This is a full credit risk pipeline for a portfolio of 75,000 resolved Lending Club loans, built on MySQL and Python, ending in a Streamlit dashboard. It goes further than prediction:

- A normalized MySQL schema with SQL views that do real feature engineering (window functions, CTEs, peer benchmarking) rather than pushing everything into pandas
- An XGBoost model with a Logistic Regression baseline, trained on data that was actually inspected first, not just fed in raw
- SHAP explainability at both the portfolio and individual loan level, with plain English labels instead of raw column names
- An Expected Loss engine (PD times LGD times EAD) closing the loop by writing risk scores back into the database
- A calibration fix using isotonic regression, because a risk score should mean what it says
- Vintage and cohort analysis tracking default rates by months on book across loan origination quarters
- Stress testing calibrated against real Federal Reserve CCAR scenarios and real historical unemployment data, not invented multipliers
- An interactive simulator that scores a hypothetical loan live and flags when your inputs fall outside anything the model has actually seen

## The architecture

```
                Lending Club raw data
                    2.2M loans
                         |
                         v
                      MySQL
              normalized schema, 5 tables
                         |
                         v
              SQL feature engineering
         views, window functions, peer benchmarking
                         |
                         v
                   Python pipeline
              EDA, cleaning, XGBoost training
                         |
                         v
              Isotonic calibration
            fit on a held out split
                         |
        -----------------------------------
        |               |                  |
        v               v                  v
      SHAP        Expected Loss      Stress Test
  explainability   PD x LGD x EAD    FRED + Fed scenarios
        |               |                  |
        -----------------------------------
                         |
                         v
                      MySQL
           risk_scores, model_monitoring
        written back, closing the loop
                         |
                         v
                Streamlit dashboard
                   7 pages, live queries
```

The loop matters more than any single stage. Every number on the dashboard traces back to a query against the same database the model wrote into, not a cached export.

## What the data actually showed

A few findings surfaced during this build that were genuinely unexpected, and are worth more than a bullet point.

**The calibration problem.** The model used `scale_pos_weight` to handle the roughly 20/80 class imbalance between defaulted and healthy loans. This is standard practice for improving recall, but it distorts the model's raw probabilities in the process. A loan scored at 80% risk was not actually defaulting anywhere near 80% of the time. Recalibrating with isotonic regression on a held out split fixed this, and it mattered financially: the portfolio's Expected Loss estimate dropped from an inflated $45.4M to a trustworthy $21.1M once the fix was in.

The fix holds up under inspection, not just in theory. Across the bins covering roughly 7% to 56% predicted probability, which is where the bulk of the portfolio actually falls, predicted and observed default rates now track closely: 45.8% predicted against 45.7% observed, 56.3% predicted against 56.0% observed, and similarly tight across the rest of that range. The two highest bins deviate more (one lands at 100%, the other at 67%), but each covers only a handful of loans, a known instability of isotonic regression in a sparse tail rather than a new problem the fix introduced.

**Underwriting drift, not a financial crisis effect.** The dataset spans 2007 to 2018, covering the financial crisis. The instinct was to expect loans originated during 2007 to 2009 to default more often. They didn't. At the 36 month mark, crisis era loans defaulted at 15.3% against 19.6% for every other cohort. The real story turned out to be something else: raw default rates climbed steadily from 12.5% in 2010 to 24.8% in 2016, tracking the platform's growth from a small early user base to a mass market product reaching further down the credit spectrum. The crisis window is also too small a sample (372 loans) to say anything statistically confident either way.

**A stress test that almost lied by omission.** Fitting default rate against historical unemployment data produced a negative sensitivity: rising unemployment appeared to reduce risk. That's backwards, and the cause was a confound with the underwriting drift finding above, since unemployment fell over the same years that origination volume rose. Controlling for that removed the confound but also removed statistical significance (the 95% confidence interval spans zero). Rather than quietly pick whichever number looked better, the model uses the conservative upper bound of that interval for scenario shifts, and says exactly why in the dashboard.

**Interest rate is not a free input.** Testing the what-if simulator with an inconsistent combination (grade G credit, a rate that in reality only applies to grade A or B) produced a lower risk score for the worse credit grade. The dataset showed why: Lending Club sets interest rate almost entirely by grade, so that combination had never occurred in training. The simulator now checks entered values against each grade's real historical range and warns when an input is asking the model to guess.

## Model performance

XGBoost, calibrated with isotonic regression on a held out split.

| Metric | Value |
|---|---|
| ROC-AUC | 0.708 |
| Recall on defaults | 63% |
| Training data | 75,000 resolved loans, 2007 to 2018 |
| Base default rate | 20.3% |

Precision and recall at a fixed 0.5 threshold are close to meaningless after calibration, since predicted probabilities now correctly reflect the ~20% base rate and few loans cross 0.5. ROC-AUC is threshold independent and is the metric to trust here.

## A real, measured SQL optimization

Indexing was benchmarked rather than assumed. On a targeted seven day date range query joined against repayments, an index on `issue_date` reduced execution time from 43.5ms to 15.6ms, a 64% improvement, confirmed with `EXPLAIN ANALYZE`. On a wider full year range covering roughly 29% of the table, the same index was correctly ignored by MySQL's optimizer in favor of a full table scan, since scanning that much of the table is genuinely cheaper than an index lookup at that selectivity. Both results are in the repository, because a working optimization and a case where an index doesn't help are both real findings.

## What's deliberately not in here

A few things were considered and cut, on purpose:

- **Monte Carlo simulation with correlated defaults.** A real technique (single factor Gaussian copula, the same family of model behind Basel capital requirements) but one that needs to be understood well enough to defend under questioning, not just generated. Cut rather than shipped shallow.
- **Population stability index and fair lending checks.** Both legitimate, both cut to keep the project's depth concentrated rather than spread thin across too many partially finished ideas.
- **A GenAI powered report generator, a chatbot interface, a full rebuild in React and FastAPI.** All would add surface area without adding real analytical depth, which was the actual goal from the start.

## Tech stack

MySQL 8.4, Python 3.9, XGBoost, scikit-learn, SHAP, statsmodels, Streamlit, Altair, pandas, SQLAlchemy.

## Repository structure

```
database/       schema, SQL views, indexing benchmark, migrations
src/            data ingestion script
python/         training, scoring, stress testing scripts
dashboard/      Streamlit app
models/         saved model artifacts and generated plots
```

## Running it locally

You'll need `archive/loan.csv` — the Lending Club loan dataset — placed yourself; it's not included in the repo.

```
pip install -r requirements.txt
mysql -u root -p < database/schema.sql
mysql -u root -p credit_risk_platform < database/views.sql
mysql -u root -p credit_risk_platform < database/add_expected_loss.sql
mysql -u root -p credit_risk_platform < database/vintage_analysis.sql
mysql -u root -p credit_risk_platform < database/add_stress_test.sql
python3 src/ingest.py
cd python
python3 train_model.py
python3 explain_and_score.py
python3 stress_test.py
cd ../dashboard
streamlit run app.py
```

Copy `.env.example` to `.env` at the repo root, in `python/`, and in `dashboard/`, and fill in your own MySQL credentials first. `ingest.py` and the dashboard/pipeline scripts read relative paths (`archive/...`, `../models/...`) assuming they're run from the working directories shown above, not the repo root throughout.

## Limitations, stated plainly

Lending Club provides no true customer identifier or age field, so each loan is treated as one customer record. Repayment history is a snapshot built from real aggregate fields (last payment, total paid, delinquency flag), not a full payment by payment ledger, since that granularity isn't in the source data. Loss Given Default is a flat 45% assumption, not modeled. Risk tier boundaries are percentile based against the scored population and reflect a judgment call about what share of a portfolio a team could realistically review manually, not a figure calibrated against any real institution's actual capacity. The stress test's unemployment sensitivity, fit on only 12 annual data points, should be read as directional rather than precise.

## Author

Built by Nurmuhammad Nazmi.
