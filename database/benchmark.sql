

USE credit_risk_platform;



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



DROP INDEX idx_loans_issue_date ON loans;


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


CREATE INDEX idx_loans_issue_date ON loans(issue_date);


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



EXPLAIN ANALYZE
SELECT * FROM repayments WHERE loan_id = 12345;

DROP INDEX idx_repayments_loan_id ON repayments;

EXPLAIN ANALYZE
SELECT * FROM repayments WHERE loan_id = 12345;

CREATE INDEX idx_repayments_loan_id ON repayments(loan_id);

EXPLAIN ANALYZE
SELECT * FROM repayments WHERE loan_id = 12345;


