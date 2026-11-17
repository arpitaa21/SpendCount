-- PostgreSQL baseline schema. The development app can also create the same
-- schema through SQLAlchemy metadata; apply this once to a fresh PostgreSQL DB.
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(255) NOT NULL UNIQUE,
  hashed_password VARCHAR(255) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE transactions (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  date DATE NOT NULL,
  description VARCHAR(250) NOT NULL,
  amount DOUBLE PRECISION NOT NULL CHECK (amount > 0),
  transaction_type VARCHAR(10) NOT NULL CHECK (transaction_type IN ('income','expense')),
  category VARCHAR(40) NOT NULL,
  payment_method VARCHAR(40),
  source VARCHAR(30) NOT NULL DEFAULT 'manual',
  notes VARCHAR(500),
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX ix_transactions_user_id ON transactions(user_id);
CREATE INDEX ix_transactions_date ON transactions(date);
CREATE INDEX ix_tx_user_date_type ON transactions(user_id,date,transaction_type);
CREATE TABLE budgets (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  category VARCHAR(40) NOT NULL,
  monthly_limit DOUBLE PRECISION NOT NULL CHECK (monthly_limit > 0),
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
  CONSTRAINT uq_budget_user_category UNIQUE(user_id,category)
);
CREATE INDEX ix_budgets_user_id ON budgets(user_id);
CREATE TABLE income_schedules (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  income_source VARCHAR(50) NOT NULL,
  expected_amount DOUBLE PRECISION NOT NULL CHECK (expected_amount > 0),
  expected_date DATE NOT NULL,
  recurring_frequency VARCHAR(20) NOT NULL DEFAULT 'one-time',
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX ix_income_schedules_user_id ON income_schedules(user_id);
CREATE TABLE forecasts (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  forecast_date DATE NOT NULL,
  predicted_balance DOUBLE PRECISION NOT NULL,
  expected_expenses DOUBLE PRECISION NOT NULL,
  confidence_or_uncertainty DOUBLE PRECISION NOT NULL,
  model_version VARCHAR(40) NOT NULL DEFAULT 'baseline-v1',
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX ix_forecasts_user_id ON forecasts(user_id);
CREATE TABLE alerts (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  alert_type VARCHAR(30) NOT NULL,
  title VARCHAR(120) NOT NULL,
  description VARCHAR(500) NOT NULL,
  severity VARCHAR(20) NOT NULL DEFAULT 'info',
  is_read BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX ix_alerts_user_id ON alerts(user_id);
CREATE TABLE category_corrections (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  transaction_id INTEGER REFERENCES transactions(id) ON DELETE SET NULL,
  description VARCHAR(250) NOT NULL,
  predicted_category VARCHAR(40) NOT NULL,
  corrected_category VARCHAR(40) NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX ix_category_corrections_user_id ON category_corrections(user_id);
