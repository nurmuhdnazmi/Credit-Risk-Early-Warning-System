-- ============================================================
-- Credit Risk Early-Warning Platform — Schema
-- Day 1: Normalized core tables
-- ============================================================

CREATE DATABASE IF NOT EXISTS credit_risk_platform;
USE credit_risk_platform;

-- ------------------------------------------------------------
-- 1. customers
-- One row per borrower. Demographic / financial profile.
-- ------------------------------------------------------------
CREATE TABLE customers (
    customer_id         INT PRIMARY KEY AUTO_INCREMENT,
    age                  INT,
    annual_income        DECIMAL(12,2),
    employment_length    INT,           -- years employed
    region               VARCHAR(50),
    dti                  DECIMAL(5,2),  -- debt-to-income ratio
    home_ownership       VARCHAR(20),
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------
-- 2. loans
-- One row per loan. Core loan attributes + outcome label.
-- ------------------------------------------------------------
CREATE TABLE loans (
    loan_id              INT PRIMARY KEY AUTO_INCREMENT,
    customer_id          INT NOT NULL,
    loan_amount          DECIMAL(12,2),
    interest_rate        DECIMAL(5,2),
    term_months          INT,
    grade                VARCHAR(5),
    sub_grade            VARCHAR(5),
    purpose              VARCHAR(50),
    issue_date           DATE,
    default_flag         TINYINT(1) DEFAULT 0,  -- 1 = defaulted
    CONSTRAINT fk_loans_customer
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
        ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- 3. repayments
-- One row per repayment event. Used to build trend/behaviour features.
-- ------------------------------------------------------------
CREATE TABLE repayments (
    repayment_id         INT PRIMARY KEY AUTO_INCREMENT,
    loan_id              INT NOT NULL,
    payment_date         DATE,
    amount_paid          DECIMAL(12,2),
    days_late            INT DEFAULT 0,
    CONSTRAINT fk_repayments_loan
        FOREIGN KEY (loan_id) REFERENCES loans(loan_id)
        ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- 4. risk_scores
-- Output of the model + the decision/recommendation layer.
-- One row per loan per scoring run (model_version allows history).
-- ------------------------------------------------------------
CREATE TABLE risk_scores (
    score_id             INT PRIMARY KEY AUTO_INCREMENT,
    loan_id              INT NOT NULL,
    model_version        VARCHAR(20),
    risk_score           DECIMAL(5,2),   -- e.g. 0-100 probability-scaled
    prediction_date      DATE,
    recommended_action   VARCHAR(100),   -- e.g. 'Monitor', 'Reduce limit', 'Manual review'
    CONSTRAINT fk_riskscores_loan
        FOREIGN KEY (loan_id) REFERENCES loans(loan_id)
        ON DELETE CASCADE
);

-- ------------------------------------------------------------
-- 5. model_monitoring
-- One row per trained model version. Tracks performance over time.
-- ------------------------------------------------------------
CREATE TABLE model_monitoring (
    model_version        VARCHAR(20) PRIMARY KEY,
    trained_date         DATE,
    accuracy             DECIMAL(5,4),
    precision_score      DECIMAL(5,4),
    recall_score         DECIMAL(5,4),
    roc_auc              DECIMAL(5,4),
    notes                VARCHAR(255)
);

-- ------------------------------------------------------------
-- Indexes (added deliberately, not by default —
-- these are the ones you'll benchmark on Day 2)
-- ------------------------------------------------------------
CREATE INDEX idx_loans_customer_id ON loans(customer_id);
CREATE INDEX idx_loans_issue_date ON loans(issue_date);
CREATE INDEX idx_repayments_loan_id ON repayments(loan_id);
CREATE INDEX idx_repayments_payment_date ON repayments(payment_date);
CREATE INDEX idx_riskscores_loan_id ON risk_scores(loan_id);
