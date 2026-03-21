-- Minimal DB init placeholder for local dev
CREATE TABLE IF NOT EXISTS prediction_audit (
  id SERIAL PRIMARY KEY,
  loan_id TEXT NOT NULL,
  default_probability DOUBLE PRECISION NOT NULL,
  risk_category TEXT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT NOW()
);
