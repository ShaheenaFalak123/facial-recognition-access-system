-- Flag events that are a failed attempt immediately preceded by another
-- failed attempt at the same camera within 2 minutes -- an early,
-- lightweight signal, cheaper to compute than the full burst-detection
-- query in 03_anomaly_detection.sql.
--
-- Window functions used: LAG() to look at the *previous* row without a
-- self-join. A self-join (recognition_events a JOIN recognition_events b
-- ON a.camera_id = b.camera_id AND b.event_timestamp < a.event_timestamp)
-- would need an extra step to pick only the single closest prior row;
-- LAG() gets it directly, in one pass, ordered per camera.
-- Postgres has no QUALIFY clause (unlike Snowflake/BigQuery/DuckDB), so
-- the window function goes in a CTE and gets filtered in an outer WHERE.
WITH with_prev AS (
    SELECT
        event_id,
        camera_id,
        event_timestamp,
        is_match,
        LAG(is_match) OVER w AS prev_attempt_was_match,
        event_timestamp - LAG(event_timestamp) OVER w AS time_since_prev_attempt
    FROM recognition_events
    WINDOW w AS (PARTITION BY camera_id ORDER BY event_timestamp)
)
SELECT *
FROM with_prev
WHERE NOT is_match
    AND NOT prev_attempt_was_match
    AND time_since_prev_attempt <= INTERVAL '2 minutes'
ORDER BY event_timestamp;
