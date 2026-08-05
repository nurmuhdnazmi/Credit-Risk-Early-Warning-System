USE credit_risk_platform;

ALTER TABLE risk_scores
    ADD COLUMN probability_of_default DECIMAL(6,5) AFTER risk_score,
    ADD COLUMN lgd_assumption DECIMAL(4,2) AFTER probability_of_default,
    ADD COLUMN exposure_at_default DECIMAL(12,2) AFTER lgd_assumption,
    ADD COLUMN expected_loss DECIMAL(12,2) AFTER exposure_at_default;

TRUNCATE TABLE risk_scores;
