# Facial Recognition Access System

A face verification service plus the analytics layer that sits on top of the events it produces:

- **`ml-service/`** — face verification API (PCA/Eigenfaces + SVM, scikit-learn) that decides whether a submitted face image matches an enrolled identity. Served via FastAPI, containerized with Docker, tested with pytest, and built/published by a GitHub Actions CI/CD pipeline.
- **`db-analytics/`** — Postgres (Supabase) analytics over the recognition events: rolling match rates, spoofing/anomaly detection with window functions and CTEs, and query optimization with indexing, plus a Streamlit dashboard.

## Results (ml-service)

Task: verify whether a face is the enrolled identity (Arnold Schwarzenegger, LFW) vs. anyone else.
Data: 2,991 LFW images after face detection (41 positives, **~71:1 imbalance**), stratified 80/20 split (`random_state=42`).
Test set: **599 images, 8 positives** — untouched until final evaluation.

| Metric (held-out test set) | SVC default cutoff | Tuned threshold (0.158) |
|---|---|---|
| Recall (positives caught) | 12.5% (1/8) | **87.5% (7/8)** |
| Precision | 50.0% (1/2) | 29.2% (7/24) |
| F1 | 0.20 | **0.44** |
| Accuracy | 98.7% | 97.0% |

Threshold-independent: **ROC-AUC 0.957**, **PR-AUC 0.564** (vs. a 1.3% base rate for a random classifier).
Selected model: PCA (100 components, whitened) + RBF SVM (C=10, gamma=0.01, `class_weight="balanced"`).

**How to read this:**
- Accuracy is misleading here — predicting "not the enrolled person" for everything scores 98.7%. The default cutoff almost does exactly that, catching only 1 of 8 positives.
- The threshold is chosen from **out-of-fold** predictions on the training set (never the test set), using `StratifiedGroupKFold` so augmented copies of the same photo can't leak across folds. That moves the operating point from 12.5% to 87.5% recall without changing the model.
- The trade-off is precision: at this threshold roughly 2 in 3 accepted attempts would need a second check. For access control, missing the real person was treated as the costlier error.
- With only 8 test positives, each positive is worth 12.5 points of recall — these numbers are indicative, not precise. More enrolment photos per identity, or modern face embeddings (e.g. ArcFace/FaceNet), are the obvious next steps.

## Engineering highlights

- **Training/serving skew removed:** training images are routed through the *same* face detection + crop pipeline the API uses at inference (`src/inference.py`). Before this fix, a real photo of the enrolled person scored near-zero confidence, because Eigenfaces is not translation- or scale-invariant.
- **Minority-class augmentation** (flips, small rotations) applied to the training split only, after the train/test split.
- **Model selection on PR-AUC** (`average_precision`) via `GridSearchCV` over RBF and linear SVMs, with every run tracked in **MLflow** (params, metrics, PR curve).
- **CI/CD (GitHub Actions):** `ruff` lint + `pytest` → build Docker image and push to GHCR (`ghcr.io/shaheenafalak123/face-verification-api`) → deploy step to Hugging Face Spaces (runs when the `HF_TOKEN` and `HF_SPACE` repository secrets are set).
- **SQL analytics:** burst detection of ≥3 failed attempts per camera in a trailing 5-minute window (`RANGE` frame over an interval), rolling match rates, and time-bucketed trends. A composite `(camera_id, event_timestamp)` index is verified with `EXPLAIN ANALYZE` (index scan touches 5 rows vs. a sequential scan over all 599).

## Run it

```bash
# Train (downloads LFW via scikit-learn, logs to MLflow)
cd ml-service
pip install -r requirements.txt
python src/train.py

# Serve the API (Docker)
docker build -t face-verification-api .
docker run -p 7860:7860 face-verification-api
# GET /health, POST /predict (multipart image upload)

# Tests
pytest tests/ -v
```

Analytics: apply `db-analytics/schema.sql` and `seed_data.sql` to a Postgres/Supabase database, run the queries in `db-analytics/queries/`, and launch `streamlit run db-analytics/dashboard.py` (needs a `SUPABASE_URL` secret).
