-- PathSense v2 — PostgreSQL schema initialization
-- Runs automatically when the postgres container starts for the first time.

CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS emergency_contacts (
    id            SERIAL PRIMARY KEY,
    user_id       TEXT        NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    phone_number  TEXT        NOT NULL,
    label         TEXT        DEFAULT 'Emergency Contact',
    active        BOOLEAN     DEFAULT TRUE,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_emergency_contacts_user_id ON emergency_contacts(user_id);
CREATE INDEX IF NOT EXISTS idx_emergency_contacts_active  ON emergency_contacts(user_id, active);

CREATE TABLE IF NOT EXISTS sensor_events (
    id              BIGSERIAL PRIMARY KEY,
    user_id         TEXT        NOT NULL,
    session_id      TEXT        NOT NULL,
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL,
    gps_accuracy_m  REAL,
    speed_kmh       REAL,
    heading_deg     REAL,
    heading_change_deg REAL,
    road_condition  TEXT,
    traffic_control TEXT,
    num_vehicles    INT,
    ingested_at     TIMESTAMPTZ DEFAULT NOW(),
    event_ts        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sensor_events_user_session ON sensor_events(user_id, session_id);

CREATE TABLE IF NOT EXISTS risk_predictions (
    id              BIGSERIAL PRIMARY KEY,
    user_id         TEXT        NOT NULL,
    session_id      TEXT,
    lat             DOUBLE PRECISION NOT NULL,
    lon             DOUBLE PRECISION NOT NULL,
    probability     REAL        NOT NULL,
    risk_level      TEXT        NOT NULL,
    message         TEXT,
    model_version   TEXT,
    from_cache      BOOLEAN     DEFAULT FALSE,
    predicted_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_predictions_user ON risk_predictions(user_id, predicted_at DESC);

CREATE TABLE IF NOT EXISTS alert_log (
    id          BIGSERIAL PRIMARY KEY,
    user_id     TEXT        NOT NULL,
    session_id  TEXT,
    risk_level  TEXT        NOT NULL,
    probability REAL        NOT NULL,
    lat         DOUBLE PRECISION,
    lon         DOUBLE PRECISION,
    message     TEXT,
    sent_to     TEXT[],
    triggered_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alert_log_user ON alert_log(user_id, triggered_at DESC);
