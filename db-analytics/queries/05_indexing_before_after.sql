-- Indexing impact, verified with real EXPLAIN ANALYZE output (not
-- estimated) against the live Supabase instance -- see the captured
-- output below each block. This is the query the composite index
-- (camera_id, event_timestamp) was specifically chosen for: filter by
-- camera, filter by a time range, sorted by time -- exactly the access
-- pattern the anomaly-detection query in 03 also uses.

EXPLAIN ANALYZE
SELECT event_id, confidence_score
FROM recognition_events
WHERE camera_id = 2
    AND event_timestamp BETWEEN '2026-07-12 09:00:00' AND '2026-07-12 11:00:00'
ORDER BY event_timestamp;

-- WITH idx_recognition_events_camera_timestamp (as created in schema.sql):
--
--   Index Scan using idx_recognition_events_camera_timestamp on recognition_events
--     (cost=0.28..2.50 rows=1 width=20) (actual time=0.015..0.016 rows=5 loops=1)
--     Index Cond: ((camera_id = 2) AND (event_timestamp >= ...) AND (event_timestamp <= ...))
--   Planning Time: 0.448 ms
--   Execution Time: 0.092 ms
--
-- WITHOUT the index (DROP INDEX idx_recognition_events_camera_timestamp,
-- idx_recognition_events_timestamp; then re-ran the same query):
--
--   Sort  (cost=15.49..15.50 rows=1 width=20) (actual time=0.149..0.150 rows=5 loops=1)
--     Sort Key: event_timestamp
--     ->  Seq Scan on recognition_events
--           (cost=0.00..15.48 rows=1 width=20) (actual time=0.084..0.089 rows=5 loops=1)
--           Filter: (event_timestamp range AND camera_id = 2)
--           Rows Removed by Filter: 594
--   Planning Time: 0.311 ms
--   Execution Time: 0.179 ms
--
-- Honest read of these numbers: at 599 rows, the wall-clock difference
-- (0.092ms vs 0.179ms) is real but small in absolute terms -- both are
-- sub-millisecond, and Postgres' planner is smart enough to use the
-- index here even on a small table because the predicate is selective
-- (5 rows out of 599). What matters for judging whether this will
-- scale isn't the millisecond gap on today's data, it's the *access
-- pattern* in the plan: with the index, the scan touches exactly the 5
-- matching rows; without it, it touches all 599 and discards 594
-- ("Rows Removed by Filter"). That second number is what grows
-- linearly with table size regardless of how selective the query is --
-- at 5 million rows instead of 599, the sequential scan is doing 5
-- million row-checks per query instead of a handful, and the gap that
-- was sub-millisecond here becomes seconds. Indexes were restored
-- after this comparison (see bottom of schema.sql).
