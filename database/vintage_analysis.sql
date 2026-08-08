USE credit_risk_platform;

-- vintage_loan_base
-- months_on_book is approximate: TIMESTAMPDIFF(MONTH, issue_date, payment_date),
-- where payment_date is the loan's last recorded payment (sourced from
-- last_pymnt_d during ingestion). There's no full amortization schedule in
-- this dataset, so this is time-to-resolution, not a true monthly payment
-- history — a loan's default could have happened anywhere up to that point.
CREATE OR REPLACE VIEW vintage_loan_base AS
SELECT
    l.loan_id,
    CONCAT(YEAR(l.issue_date), '-Q', QUARTER(l.issue_date)) AS origination_quarter,
    l.default_flag,
    TIMESTAMPDIFF(MONTH, l.issue_date, r.payment_date) AS months_on_book
FROM loans l
JOIN repayments r ON l.loan_id = r.loan_id
WHERE l.issue_date IS NOT NULL
    AND r.payment_date IS NOT NULL
    AND TIMESTAMPDIFF(MONTH, l.issue_date, r.payment_date) BETWEEN 0 AND 60;

-- vintage_analysis
-- Cumulative default rate per origination-quarter cohort at each
-- months-on-book checkpoint (0, 3, 6, ... 60), for plotting vintage curves.
CREATE OR REPLACE VIEW vintage_analysis AS
WITH RECURSIVE mob_buckets AS (
    SELECT 0 AS mob_bucket
    UNION ALL
    SELECT mob_bucket + 3 FROM mob_buckets WHERE mob_bucket < 60
),
cohort_totals AS (
    SELECT origination_quarter, COUNT(*) AS cohort_size
    FROM vintage_loan_base
    GROUP BY origination_quarter
)
SELECT
    t.origination_quarter,
    t.cohort_size,
    b.mob_bucket,
    SUM(CASE WHEN v.default_flag = 1 AND v.months_on_book <= b.mob_bucket THEN 1 ELSE 0 END) AS cumulative_defaults,
    ROUND(
        SUM(CASE WHEN v.default_flag = 1 AND v.months_on_book <= b.mob_bucket THEN 1 ELSE 0 END) / t.cohort_size,
        4
    ) AS cumulative_default_rate,
    CASE WHEN LEFT(t.origination_quarter, 4) IN ('2007', '2008', '2009') THEN 1 ELSE 0 END AS is_crisis_vintage
FROM cohort_totals t
CROSS JOIN mob_buckets b
JOIN vintage_loan_base v ON v.origination_quarter = t.origination_quarter
GROUP BY t.origination_quarter, t.cohort_size, b.mob_bucket
ORDER BY t.origination_quarter, b.mob_bucket;

-- yearly_origination_quality
-- Raw default rate by origination year, regardless of months-on-book.
-- Used to separate underwriting-quality drift over time from the
-- crisis-window comparison in vintage_analysis above.
CREATE OR REPLACE VIEW yearly_origination_quality AS
SELECT
    YEAR(issue_date) AS origination_year,
    COUNT(*) AS n_loans,
    ROUND(AVG(default_flag), 4) AS default_rate
FROM loans
WHERE issue_date IS NOT NULL
GROUP BY YEAR(issue_date)
ORDER BY origination_year;
