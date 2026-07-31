USE credit_risk_platform;

-- loan_base_features

CREATE OR REPLACE VIEW loan_base_features AS
SELECT
    l.loan_id,
    l.customer_id,
    l.loan_amount,
    l.interest_rate,
    l.term_months,
    l.grade,
    l.sub_grade,
    l.purpose,
    l.issue_date,
    l.default_flag,
    c.annual_income,
    c.employment_length,
    c.region,
    c.dti,
    c.home_ownership,
    r.amount_paid AS last_payment_amount,
    r.days_late AS had_delinquency,      -- 0/1 flag, see ingestion notes
    -- Loan amount as a share of annual income — a classic affordability signal
    ROUND(l.loan_amount / NULLIF(c.annual_income, 0), 4) AS loan_to_income_ratio
FROM loans l
JOIN customers c ON l.customer_id = c.customer_id
LEFT JOIN repayments r ON l.loan_id = r.loan_id;



--  loan_peer_benchmark_features

CREATE OR REPLACE VIEW loan_peer_benchmark_features AS
SELECT
    loan_id,
    grade,
    region,
    issue_date,
    interest_rate,
    default_flag,

    -- Where does this loan's interest rate sit within its own grade?
    -- (grade should already roughly determine rate, so outliers are interesting)
    PERCENT_RANK() OVER (
        PARTITION BY grade ORDER BY interest_rate
    ) AS interest_rate_percentile_in_grade,

    -- Historical default rate for this grade (peer benchmark)
    AVG(default_flag) OVER (
        PARTITION BY grade
    ) AS grade_default_rate,

    -- Historical default rate for this region (peer benchmark)
    AVG(default_flag) OVER (
        PARTITION BY region
    ) AS region_default_rate,

    -- Rank of loan amount within its own grade (largest = 1)
    RANK() OVER (
        PARTITION BY grade ORDER BY loan_amount DESC
    ) AS loan_amount_rank_in_grade

FROM loan_base_features;



-- monthly_default_trend

CREATE OR REPLACE VIEW monthly_default_trend AS
WITH monthly_stats AS (
    SELECT
        DATE_FORMAT(issue_date, '%Y-%m-01') AS issue_month,
        COUNT(*) AS loans_issued,
        SUM(default_flag) AS loans_defaulted,
        AVG(default_flag) AS monthly_default_rate
    FROM loans
    WHERE issue_date IS NOT NULL
    GROUP BY DATE_FORMAT(issue_date, '%Y-%m-01')
)
SELECT
    issue_month,
    loans_issued,
    loans_defaulted,
    monthly_default_rate,
    -- 3-month rolling average default rate
    AVG(monthly_default_rate) OVER (
        ORDER BY issue_month
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS rolling_3m_default_rate
FROM monthly_stats
ORDER BY issue_month;

-- final_model_features

CREATE OR REPLACE VIEW final_model_features AS
SELECT
    b.loan_id,
    b.loan_amount,
    b.interest_rate,
    b.term_months,
    b.grade,
    b.dti,
    b.annual_income,
    b.employment_length,
    b.home_ownership,
    b.purpose,
    b.loan_to_income_ratio,
    b.had_delinquency,
    p.interest_rate_percentile_in_grade,
    p.grade_default_rate,
    p.region_default_rate,
    p.loan_amount_rank_in_grade,
    b.default_flag
FROM loan_base_features b
JOIN loan_peer_benchmark_features p ON b.loan_id = p.loan_id;




-- Q1: Which purpose categories have the highest default rate,
-- among purposes with meaningful volume (>500 loans)?
SELECT
    purpose,
    COUNT(*) AS n_loans,
    ROUND(AVG(default_flag), 4) AS default_rate
FROM loans
GROUP BY purpose
HAVING COUNT(*) > 500
ORDER BY default_rate DESC;

-- Q2: Top 5 riskiest regions by default rate (min 200 loans, to avoid noise)
SELECT
    c.region,
    COUNT(*) AS n_loans,
    ROUND(AVG(l.default_flag), 4) AS default_rate
FROM loans l
JOIN customers c ON l.customer_id = c.customer_id
GROUP BY c.region
HAVING COUNT(*) > 200
ORDER BY default_rate DESC
LIMIT 5;
