USE credit_risk_platform;

CREATE TABLE IF NOT EXISTS stress_test_results (
    scenario                VARCHAR(30) PRIMARY KEY,   -- baseline, adverse, severely_adverse
    unemployment_delta_pp   DECIMAL(4,2),               -- scenario's unemployment rate move, in percentage points
    scenario_source         VARCHAR(255),               -- where the scenario numbers came from
    portfolio_el            DECIMAL(14,2),
    pd_sensitivity_slope    DECIMAL(8,5),                -- fitted point estimate, pp default rate per 1pp unemployment
    sensitivity_r_squared   DECIMAL(5,4),
    sensitivity_ci_lower    DECIMAL(8,5),
    sensitivity_ci_upper    DECIMAL(8,5),
    applied_sensitivity_pp  DECIMAL(8,5),                -- value actually used for the scenario shift (see stress_test.py)
    n_years_fitted          INT,
    run_date                DATE
);
