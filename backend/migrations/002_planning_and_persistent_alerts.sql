-- Additive migration for persisted alert state and student planning features.
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS alert_key VARCHAR(120) NOT NULL DEFAULT 'legacy';
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS snoozed_until DATE;
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;
CREATE UNIQUE INDEX IF NOT EXISTS uq_alert_user_key
  ON alerts (user_id, alert_key) WHERE alert_key <> 'legacy';

CREATE TABLE IF NOT EXISTS planned_commitments (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  title VARCHAR(120) NOT NULL,
  category VARCHAR(40) NOT NULL DEFAULT 'Other',
  amount DOUBLE PRECISION NOT NULL CHECK (amount > 0),
  due_date DATE NOT NULL,
  recurring_frequency VARCHAR(20) NOT NULL DEFAULT 'one-time',
  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_planned_commitments_user_id ON planned_commitments(user_id);
CREATE INDEX IF NOT EXISTS ix_planned_commitments_due_date ON planned_commitments(due_date);

CREATE TABLE IF NOT EXISTS planning_settings (
  id SERIAL PRIMARY KEY,
  user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
  emergency_buffer_target DOUBLE PRECISION NOT NULL DEFAULT 0 CHECK (emergency_buffer_target >= 0),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS ix_planning_settings_user_id ON planning_settings(user_id);
