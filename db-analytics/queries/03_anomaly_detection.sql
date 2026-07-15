-- Flag potential spoofing attempts: 3 or more failed match attempts at
-- the same camera within a trailing 5-minute window.
--
-- Window function used: COUNT(*) with a RANGE frame over an INTERVAL
-- (not ROWS) -- "how many failed attempts happened in the 5 minutes
-- before this one", not "how many of the last N rows", which matters
-- because failed attempts aren't evenly spaced in time.
WITH failed_attempts AS (
    SELECT event_id, camera_id, event_timestamp
    FROM recognition_events
    WHERE NOT is_match
),
with_trailing_count AS (
    SELECT
        event_id,
        camera_id,
        event_timestamp,
        COUNT(*) OVER (
            PARTITION BY camera_id
            ORDER BY event_timestamp
            RANGE BETWEEN INTERVAL '5 minutes' PRECEDING AND CURRENT ROW
        ) AS failed_attempts_in_last_5min
    FROM failed_attempts
)
SELECT
    c.location,
    c.zone,
    w.event_timestamp,
    w.failed_attempts_in_last_5min
FROM with_trailing_count w
JOIN cameras c ON c.camera_id = w.camera_id
WHERE w.failed_attempts_in_last_5min >= 3
ORDER BY w.event_timestamp;
