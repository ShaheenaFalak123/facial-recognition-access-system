-- Rolling match rate per enrolled person: for each event, what fraction
-- of the last 10 attempts against that identity were accepted?
--
-- Window function used: AVG() with an explicit ROWS frame. This is
-- different from a GROUP BY aggregate -- every event keeps its own row,
-- with a trailing summary stat computed alongside it, which is what a
-- dashboard needs (a trend line, not one final number).
SELECT
    event_id,
    claimed_person_id,
    event_timestamp,
    is_match,
    ROUND(
        AVG(is_match::int) OVER (
            PARTITION BY claimed_person_id
            ORDER BY event_timestamp
            ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
        )::numeric,
        3
    ) AS rolling_match_rate_last_10
FROM recognition_events
ORDER BY event_timestamp;
