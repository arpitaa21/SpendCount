# SpendCount

SpendCount is a student-focused budgeting and cash-flow MVP. It turns manually entered or imported transactions into a current balance, spending picture, conservative month-end estimate, unusual-spend review prompts, and a weekly spending plan. It does not connect to banks or treat planned income as guaranteed.

## What is implemented

- React + Vite responsive application with Overview, Transactions, Budgets, Cash-flow forecast, AI insights, Weekly planner, and Settings.
- FastAPI REST API with registration, login, user-scoped data, transaction CRUD, category budgets, expected income schedules, CSV preview/import, dashboard analytics, forecast and AI endpoints.
- SQLAlchemy schema using PostgreSQL when configured and SQLite by default for local development.
- PBKDF2 password hashes and signed bearer tokens. Financial records are filtered by authenticated user at the API boundary.
- TF-IDF + Logistic Regression expense categoriser, per-user correction records, per-user Isolation Forest amount outlier detection, and moving-average vs. linear-trend cash-flow estimation.
- Synthetic 120-day student demo-data generator with 200+ transactions and seeded unusual expenses.

## Run locally

### Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python seed_demo.py
python run.py
```

API docs: <http://localhost:8000/docs>. Health check: <http://localhost:8000/api/health>.

Demo account: `demo@spendcount.app` / `SpendCount123`.

Set `DATABASE_URL` to `postgresql+psycopg://user:password@localhost:5432/spendcount` for PostgreSQL and install `psycopg[binary]`. SQLite is the default fallback. Use a long random `SECRET_KEY` outside local development.

### Frontend

In a second terminal:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open <http://localhost:5173>. `VITE_API_URL` can point at another API base URL. The browser uses a signed bearer token stored locally; signing out clears it.

### Docker Compose

`docker compose up --build` starts PostgreSQL, the API, and Vite. Seed the demo data in another terminal with `docker compose exec api python seed_demo.py`. The local compose credentials and key are for development only.

## CSV import

The Transactions page offers a downloadable template with the header:

```csv
date,description,amount,type,category
2026-11-01,Campus cafe lunch,185,expense,Food
2026-11-02,Family allowance,8000,income,Family Allowance
```

`POST /api/transactions/import` first parses and previews accepted and rejected rows. To import the accepted rows, send the same CSV with `preview=false`. Invalid dates, amounts, transaction types, categories, and duplicates are returned with row numbers and reasons. Missing categories are predicted from the description; low-confidence predictions are marked `Needs Review`.

## API map

- `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`
- `GET|POST /api/transactions`, `PUT|DELETE /api/transactions/{id}`, `POST /api/transactions/import`, `GET /api/transactions/template`
- `GET /api/dashboard/summary`, `GET /api/dashboard/spending-breakdown`, `GET /api/dashboard/spending-trends`
- `GET|POST /api/budgets`, `PUT|DELETE /api/budgets/{id}`
- `GET|POST /api/income-schedules`, `DELETE /api/income-schedules/{id}`
- `GET /api/forecast/month-end`, `GET /api/forecast/cash-flow`, `POST /api/forecast/recalculate`
- `POST /api/ai/categorise`, `GET /api/ai/anomalies`, `GET /api/ai/insights`, `GET /api/ai/weekly-plan`
- `GET /api/meta/categories`, `GET /api/health`

All personal endpoints require `Authorization: Bearer <token>`. FastAPI Swagger UI documents the request and response shapes.

## ML approach and evaluation

### Categorisation

The classifier uses built-in labeled student expense descriptions, word and bigram TF-IDF features, and Logistic Regression. `backend/app/ml/categorisation.py` trains on first use and stores a `joblib` artifact in the ignored `backend/app/ml/models/` folder. Predictions include the top label and confidence; low-confidence expense predictions return `Needs Review`. When a student chooses a category that differs from the top model label, the app stores a user-scoped correction in `category_corrections` and exposes that account's rows through `GET /api/ai/corrections`; corrections are not pooled across students. The small phrase corpus needs more reviewed labels before it can support reliable personalized retraining.

### Cash-flow forecast

The month-end forecast builds a daily expense series, including zero-spend days. It compares a trailing moving average with a linear trend on a chronological last-week holdout, selects the better approach (with a 10% margin), then projects the selected daily rate. Under seven days of history it falls back to the moving average and lowers confidence. Expected income appears as a separate scenario only. Confidence is a heuristic based on history length and forecast error, not a calibrated probability.

`backend/app/ml/model_evaluation.py` compares a seven/28-day moving average with Random Forest regression on lag features using a chronological 80/20 holdout and reports MAE/RMSE. Run with a Python snippet after supplying a daily series:

```python
from app.ml.model_evaluation import compare_models
print(compare_models(daily_spending))
```

Train the description categoriser and print a stratified cross-validation score for its small built-in phrase corpus with `cd backend; python train_models.py`. On the included synthetic phrases, the current 5-fold accuracy was 0.728 (±0.032); this measures only phrase-corpus consistency.

On the current synthetic daily demo ledger, the chronological 80/20 evaluation reported Random Forest MAE/RMSE ₹820.47/₹1,177.95 and moving-average MAE/RMSE ₹803.52/₹1,101.79. This synthetic result favors the simpler baseline for that sample and does not prove expected accuracy on student data.

The demo data is synthetic and cannot establish real-world accuracy. The runtime currently chooses between moving average and linear trend; Random Forest is provided for offline evaluation and is not deployed as the active forecaster.

### Anomalies and planner

Isolation Forest runs per user over log expense amounts when at least eight expenses exist. Explanations compare flagged amounts to the student's category median where possible. This MVP detects amount outliers; frequency and category-week change alerts are not yet separately modelled. Planner limits reserve a cushion and subtract a simple mandatory-cost estimate from budgets; users should treat them as estimates, not advice.

## Database

SQLAlchemy creates tables for users, transactions, budgets, income schedules, forecasts, and alerts when the API starts. Transactions and schedules have user/date indexes; budgets are unique per user/category. A PostgreSQL baseline migration is included at `backend/migrations/001_initial_schema.sql`; first-run local development also uses `create_all`. For a production release, add a versioned migration runner, account recovery, rate limiting, token rotation, secure cookie storage, audit logging, and calibrated forecast evaluation on consented real data.

## Dataset

`backend/seed_demo.py` generates about four months of realistic student income and spending, seven category budgets, an expected allowance date, and three intentionally unusual expenses. All seeded ledger rows are labeled `source=synthetic`; the CSV export is `dataset/synthetic_transactions.csv`. The dataset is illustrative only.

## Checks

Backend tests: `cd backend; pip install -r requirements.txt; pip install pytest; pytest`. Frontend production build: `cd frontend; npm install; npm run build`.

The tests cover insufficient-history forecast behavior, categoriser confidence, budget cap behavior, and the documented CSV shape. API-level isolation and full workflow tests are a next hardening step; ownership checks are implemented in the API query paths but should be exercised against disposable test databases before deployment.
