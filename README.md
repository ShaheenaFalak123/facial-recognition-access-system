# Facial Recognition Access System

Two connected pieces, built to be listed as two separate resume projects:

- **`ml-service/`** — face verification API (PCA/Eigenfaces + SVM, scikit-learn) that decides whether a submitted face image matches an enrolled identity. Served via FastAPI, containerized, tested, deployed on Hugging Face Spaces.
- **`db-analytics/`** — Postgres (Supabase) analytics layer over the recognition events the API produces: anomaly/spoofing detection with window functions and CTEs, query optimization with indexing.

Status: work in progress — details filled in as each piece is built.

## Why this exists

Built to demonstrate production ML engineering skills (testing, containerization, CI/CD, deployment, monitoring) and real SQL analytics skills (window functions, CTEs, query optimization) that aren't visible from notebook-only projects.

