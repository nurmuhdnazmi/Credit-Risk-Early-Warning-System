

USE credit_risk_platform;

-- ------------------------------------------------------------
-- Benchmark query: loans issued in a date range, joined with
-- repayments — a realistic "portfolio review" query pattern.
-- ------------------------------------------------------------

-- STEP 1: Run WITH the index in place (this is the "AFTER" baseline,
-- since schema.sql already created idx_loans_issue_date)
EXPLAIN ANALYZE
SELECT
    l.loan_id,
    l.loan_amount,
    l.issue_date,
    r.amount_paid,
    r.days_late
FROM loans l
JOIN repayments r ON l.loan_id = r.loan_id
WHERE l.issue_date BETWEEN '2015-01-01' AND '2015-12-31';


-- STEP 2: Drop the index to simulate the "before optimization" state
DROP INDEX idx_loans_issue_date ON loans;

-- STEP 3: Run the exact same query again — note the slower plan
-- (likely a full table scan instead of a range scan)
EXPLAIN ANALYZE
SELECT
    l.loan_id,
    l.loan_amount,
    l.issue_date,
    r.amount_paid,
    r.days_late
FROM loans l
JOIN repayments r ON l.loan_id = r.loan_id
WHERE l.issue_date BETWEEN '2015-01-01' AND '2015-12-31';

-- STEP 4: Re-create the index (restore schema to its intended state)
CREATE INDEX idx_loans_issue_date ON loans(issue_date);

-- STEP 5: Confirm it's back and fast again
EXPLAIN ANALYZE
SELECT
    l.loan_id,
    l.loan_amount,
    l.issue_date,
    r.amount_paid,
    r.days_late
FROM loans l
JOIN repayments r ON l.loan_id = r.loan_id
WHERE l.issue_date BETWEEN '2015-01-01' AND '2015-12-31';


-- ------------------------------------------------------------
-- Second benchmark: repayments lookup by loan_id
-- (tests idx_repayments_loan_id)
-- ------------------------------------------------------------

EXPLAIN ANALYZE
SELECT * FROM repayments WHERE loan_id = 12345;

DROP INDEX idx_repayments_loan_id ON repayments;

EXPLAIN ANALYZE
SELECT * FROM repayments WHERE loan_id = 12345;

CREATE INDEX idx_repayments_loan_id ON repayments(loan_id);

EXPLAIN ANALYZE
SELECT * FROM repayments WHERE loan_id = 12345;

-- ------------------------------------------------------------
-- What to write in your README / report:
-- "Query X ran in Y ms with the index on issue_date vs Z ms
--  without it (an N% reduction), on a 75,000-row loans table
--  joined against repayments. EXPLAIN ANALYZE confirmed the
--  planner switched from a full table scan to a range scan
--  using idx_loans_issue_date."
--
-- Use YOUR actual numbers from the output — don't estimate.
-- ------------------------------------------------------------
