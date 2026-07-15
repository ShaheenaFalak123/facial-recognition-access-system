-- Daily volume and error-rate trends. FILTER is used instead of
-- CASE/SUM boilerplate for each conditional count -- more readable, and
-- Postgres optimizes it to a single pass over the data just like the
-- CASE-WHEN version would be.
--
-- "False reject" = a genuine attempt that got rejected (the enrolled
-- person locked out of their own door). "False accept" = an impostor
-- attempt that got through. In access control these have very
-- different costs, so tracking them separately (not just overall
-- accuracy) is the point.
SELECT
    date_trunc('day', event_timestamp) AS day,
    count(*) AS total_attempts,
    count(*) FILTER (WHERE is_genuine_attempt) AS genuine_attempts,
    count(*) FILTER (WHERE NOT is_genuine_attempt) AS impostor_attempts,
    count(*) FILTER (WHERE is_genuine_attempt AND NOT is_match) AS false_rejects,
    count(*) FILTER (WHERE NOT is_genuine_attempt AND is_match) AS false_accepts
FROM recognition_events
GROUP BY 1
ORDER BY 1;
