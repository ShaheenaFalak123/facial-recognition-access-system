-- Schema for the access-control analytics layer.
--
-- Models a single-identity face-verification gate (like a badge+face
-- checkpoint): a camera captures a face, the system checks it against
-- one claimed/enrolled identity, and records the outcome. This is the
-- natural data model for what ml-service actually does (binary
-- verification against one enrolled person), rather than a general
-- multi-identity recognition log.

CREATE TABLE persons (
    person_id   SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    enrolled_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    status      TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'revoked'))
);

CREATE TABLE cameras (
    camera_id SERIAL PRIMARY KEY,
    location  TEXT NOT NULL,
    zone      TEXT NOT NULL
);

CREATE TABLE recognition_events (
    event_id            BIGSERIAL PRIMARY KEY,
    camera_id           INT NOT NULL REFERENCES cameras(camera_id),
    claimed_person_id   INT NOT NULL REFERENCES persons(person_id),
    confidence_score    REAL NOT NULL CHECK (confidence_score BETWEEN 0 AND 1),
    threshold_used       REAL NOT NULL,
    is_match             BOOLEAN NOT NULL,       -- confidence_score >= threshold_used
    is_genuine_attempt   BOOLEAN NOT NULL,        -- demo-only ground truth: was this actually
                                                   -- the enrolled person's photo, or an impostor's?
                                                   -- A real deployment wouldn't have this column --
                                                   -- you don't know an intruder's true identity.
                                                   -- It exists here only because the underlying
                                                   -- photos come from a labeled dataset, which lets
                                                   -- us sanity-check that the anomaly queries below
                                                   -- actually catch the impostor attempts.
    event_timestamp      TIMESTAMPTZ NOT NULL
);

-- Query patterns this schema needs to support well: "attempts in the
-- last N minutes at a given camera" (anomaly detection) and "attempts
-- over a time range" (dashboards/aggregation) -- both filter/sort on
-- event_timestamp, often combined with camera_id. See
-- queries/04_indexing_before_after.sql for the before/after EXPLAIN
-- ANALYZE comparison that motivates these specific indexes.
CREATE INDEX idx_recognition_events_timestamp ON recognition_events (event_timestamp);
CREATE INDEX idx_recognition_events_camera_timestamp ON recognition_events (camera_id, event_timestamp);
