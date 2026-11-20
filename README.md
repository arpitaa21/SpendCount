# SpendCount — AI-Powered Student Budget & Cash-Flow Intelligence Platform

SpendCount is a student-first personal finance and cash-flow intelligence platform designed specifically for campus life. It transforms irregular income (family allowances, scholarship disbursements, part-time jobs, stipends) and daily student expenses into a clear financial roadmap. It features genuine machine-learning transaction categorisation with an adaptive feedback loop, time-aware cash-flow runway forecasting, Isolation Forest spending anomaly detection, data-driven financial insights, and a conservative weekly spending planner.

SpendCount does not require bank credentials, OTPs, or UPI access—ensuring zero credential risk while giving students complete control over their financial records.

---

## Key Features

1. **Intelligent Financial Overview**: Real-time snapshot of current balance, month-to-date income and expenses, daily safe-to-spend limit, and dynamic health indicator.
2. **Transaction Management**: Complete CRUD support with instant search, multi-category filtering, transaction type toggles, date ranges, and pagination.
3. **Smart CSV Import & Preview**: Upload CSV bank/UPI statements with client-side template download, server-side format and schema validation, duplicate detection, and instant AI category prediction.
4. **Machine Learning Categorisation**: TF-IDF + Logistic Regression expense classifier predicting categories with confidence scoring and "Needs Review" fallback for low-confidence transactions.
5. **Adaptive User Feedback Loop**: When a student manually corrects an AI category suggestion, SpendCount persists the correction and refines a user-specific model in-memory without cross-account data leakage.
6. **Cash-Flow Runway Forecasting**: Multi-scenario month-end balance prediction comparing moving-average burn rate against linear trend using time-aware chronological holdouts.
7. **Expected Income & Recurring Expenses Modeling**: Separate modeling of unconfirmed income schedules and recurring campus expenses inferred from historical transaction intervals.
8. **Isolation Forest Spending Anomaly Detection**: Outlier detection on log-transformed transaction amounts combined with weekly category spike and frequency change analysis.
9. **Actionable AI Financial Insights**: 100% data-driven insights highlighting top spending categories, peak spending days, weekly pace comparisons, and 10% discretionary savings calculations.
10. **Weekly Spending Planner**: Realistic weekly spending budget reserving upcoming mandatory commitments (rent, utilities) and maintaining a conservative safety cushion.
11. **Category Budget Management**: Monthly category limits with visual progress meters, over-budget warnings, and exhaustion-day projections.
12. **Profile & Account Settings**: Update display name, change password with old-password verification, inspect and manage AI corrections, and view model transparency metrics.
13. **Model Evaluation & Transparency**: Live endpoint and UI comparing Random Forest Regressor vs Moving Average baseline using chronological 80/20 holdouts (MAE / RMSE metrics).
14. **Synthetic Student Demo Sandbox**: Pre-seeded demo account (`demo@spendcount.app`) with 210+ transactions over 120 days and seeded outliers.
15. **Responsive Financial Dashboard**: Modern DM Sans + Manrope UI optimized for mobile, tablet, and desktop screens with dark-mode sidebar branding.

---

## Technology Stack

- **Frontend**: React 18, Vite 5, React Router 6, Recharts (Area, Bar, and Donut charts), Lucide-React icons, Axios, Tailwind CSS & custom responsive CSS.
- **Backend**: Python 3.11+, FastAPI, Starlette, Pydantic v2 with email validation, Uvicorn, Python-Multipart.
- **Database**: SQLAlchemy 2.0 ORM with connection pooling, SQLite for zero-config local development, and PostgreSQL (`psycopg` v3 binary) for production persistence.
- **Machine Learning & Data Science**: Scikit-learn (Logistic Regression, TF-IDF Vectorizer, Isolation Forest, Random Forest Regressor, Pipeline), NumPy, Pandas, Joblib.
- **Testing & QA**: Pytest 8 with AnyIO, SQLite in-memory test fixtures, API isolation tests, CSV parsing validation, ML holdout tests, and end-to-end lifecycle integration test.
- **Deployment**: Vercel Serverless & Static Hosting (`vercel.json`), Docker, Docker Compose, Render / Railway compatible.

---

## Architecture Overview

```
                      ┌───────────────────────────────────────┐
                      │        React + Vite Frontend          │
                      │  (Responsive SPA, Charts, Navigation) │
                      └──────────────────┬────────────────────┘
                                         │ HTTP / JSON
                                         │ Bearer Token
                      ┌──────────────────▼────────────────────┐
                      │          FastAPI REST API             │
                      │  (Auth, Schemas, CORS, API Routes)   │
                      └───────┬──────────────────────┬────────┘
                              │                      │
            ┌─────────────────▼────────┐   ┌─────────▼──────────────────┐
            │   SQLAlchemy 2.0 ORM     │   │     AI / ML Engine         │
            │ SQLite (Dev) / Postgres  │   │  • TF-IDF + Logistic Reg   │
            │ Multi-Tenant Isolation   │   │  • Holdout Runway Forecast │
            └──────────────────────────┘   │  • Isolation Forest Outlier│
                                           │  • Data-driven Insights    │
                                           └────────────────────────────┘
```

---

## Machine Learning Implementation Details

### 1. Expense Categorisation (TF-IDF + Logistic Regression)
- **Base Classifier**: TF-IDF word and character n-grams with balanced Logistic Regression; the checked-in joblib artifact is trained on the labelled phrase corpus and valid expense rows in `dataset/sample_data/student_transactions_6_months.csv`.
- **Evaluation**: `backend/train_models.py` compares the text model with transparent phrase rules using 5-fold `GroupKFold` splits by normalized description, preventing repeated descriptions from appearing in both train and test folds. The supplied synthetic fixture scored 84.85% accuracy / 65.82% macro-F1 for text ML alone and 100% on its 33 unique expense descriptions for rule-plus-ML.
- **Confidence Handling**: Low-confidence or low-margin text predictions return `Needs Review`; UI explanations distinguish keyword matches, model estimates, and the student's own corrections.
- **Online User Feedback**: Corrections stored in `category_corrections` table. The model retrains in-memory with user-specific samples on request, ensuring personalization without cross-student data pooling.

The uploaded ZIP is documented as synthetic and illustrative. Its invalid-import and anomaly fixtures are retained for tests but excluded from training. These small-fixture metrics are not evidence of accuracy on real student transactions. To reproduce training and evaluation, run `python train_models.py` from `backend/`; results are written to `dataset/model_evaluation.json` and the versioned artifact to `backend/app/ml/models/categoriser.joblib`.

### 2. Cash-Flow Runway Forecasting
- **Time-Aware Holdout**: Compares a 14-day trailing moving average against a linear trend model on a chronological 7-day holdout of past daily expenses.
- **Selection Rule**: The linear trend is selected if its holdout Mean Absolute Error (MAE) beats the moving average by at least 10%; otherwise, moving average is used to prevent overfitting.
- **Scenario Separation**: Predicts conservative month-end balance *excluding* unconfirmed expected income, and provides a secondary scenario *including* confirmed expected income.
- **Safe Daily Burn**: Computes the maximum safe daily spend rate preserving a 15% runway buffer through month end.

### 3. Anomaly & Outlier Detection
- **Isolation Forest**: Fits an Isolation Forest model per user over log-transformed expense amounts (`log1p(amount)`) with dynamic contamination factor based on ledger size.
- **Contextual Explanation**: Flags amounts significantly above the student's category median and explains the deviation in plain language.
- **Category Surge & Frequency Rules**: Compares rolling 7-day totals and transaction counts against a 28-day baseline to spot lifestyle drift and sudden weekly spikes.

### 4. Model Evaluation & Transparency
- Accessible via `GET /api/ai/model-evaluation` and the Settings dashboard.
- Reports MAE and RMSE metrics comparing Random Forest Regressor vs Moving Average on an 80/20 chronological split without data leakage.

---

## Local Installation & Setup

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- Git

### 1. Backend Setup

```powershell
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate     # On Linux / macOS

# Install dependencies
pip install -r requirements.txt

# Create environment configuration
Copy-Item .env.example .env

# Seed the demo student account (demo@spendcount.app / SpendCount123)
python seed_demo.py

# Start the development server
python run.py
```

- API Base URL: `http://localhost:8000/api`
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/health`

### 2. Frontend Setup

In a new terminal:

```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create environment configuration
Copy-Item .env.example .env

# Start Vite development server
npm run dev
```

- Frontend App: `http://localhost:5173`

### 3. Docker Compose (Alternative)

To run the entire platform with PostgreSQL and containerized services:

```powershell
docker compose up --build
# In another terminal, seed demo data:
docker compose exec api python seed_demo.py
```

---

## Environment Variables

### Backend (`backend/.env`)

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `sqlite:///./spendcount.db` | Database connection string. Use `postgresql+psycopg://user:pass@host:5432/dbname` for production PostgreSQL. |
| `SECRET_KEY` | `replace-with-a-long-random-secret` | High-entropy secret key for signing user authentication tokens. |
| `ACCESS_TOKEN_MINUTES` | `10080` | Token expiration time in minutes (default is 7 days). |
| `CORS_ORIGINS` | `http://localhost:5173,http://localhost:3000` | Comma-separated list of allowed CORS origins. Preview Vercel URLs are matched automatically. |

### Frontend (`frontend/.env`)

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `VITE_API_URL` | `http://localhost:8000/api` | Backend API base URL for client-side HTTP requests. |
| `VITE_API_BASE_URL` | *(optional)* | Alternative base URL variable supported for Vercel environments. |

---

## API Documentation

### Authentication & Profile
- `POST /api/auth/register` — Create a new student account (name, email, password ≥ 8 chars).
- `POST /api/auth/login` — Sign in and receive bearer token.
- `GET /api/auth/me` — Return current authenticated user profile.
- `PUT /api/auth/profile` — Update student display name.
- `PUT /api/auth/password` — Change password (requires current password verification).

### Transactions & CSV

The downloadable CSV template uses these columns: `date,description,amount,type,category`.
- `GET /api/transactions` — List user transactions with filters (`search`, `category`, `transaction_type`, `start_date`, `end_date`, `limit`, `offset`).
- `POST /api/transactions` — Record income or expense (auto-categorises if category is empty or "Auto").
- `PUT /api/transactions/{id}` — Edit existing transaction.
- `DELETE /api/transactions/{id}` — Delete transaction.
- `POST /api/transactions/import` — CSV batch parsing (`preview=true` for preview report, `preview=false` for commit).
- `GET /api/transactions/template` — Download sample CSV template.

### Budgets & Income Schedules
- `GET /api/budgets` — List monthly category budgets.
- `POST /api/budgets` — Create or update budget limit by category.
- `PUT /api/budgets/{id}` — Update specific budget limit.
- `DELETE /api/budgets/{id}` — Remove budget.
- `GET /api/income-schedules` — List expected income schedules.
- `POST /api/income-schedules` — Add expected income estimate (one-time, weekly, monthly, semester).
- `DELETE /api/income-schedules/{id}` — Remove income schedule.

### Forecasting & Analytics
- `GET /api/dashboard/summary` — Overview metrics: balance, month income/expenses, budget pace, safe daily spend.
- `GET /api/dashboard/spending-breakdown` — Category-wise expense totals for current month.
- `GET /api/dashboard/spending-trends` — Daily income and expense timeline for current month.
- `GET /api/forecast/month-end` — Month-end forecast with uncertainty and daily rate.
- `GET /api/forecast/cash-flow` — Day-by-day projected balance curve through month end.
- `POST /api/forecast/recalculate` — Refresh forecast snapshot.

### AI & Machine Learning
- `POST /api/ai/categorise` — Predict category for transaction description.
- `GET /api/ai/corrections` — List student's recorded manual category corrections.
- `DELETE /api/ai/corrections/{id}` — Remove correction record from personal feedback queue.
- `GET /api/ai/anomalies` — Isolation Forest amount outliers and weekly surge patterns.
- `GET /api/ai/alerts` — Real-time budget pace, cash-flow shortfall, and daily spending alerts.
- `GET /api/ai/insights` — Dynamic text insights, recurring patterns, and 10% savings opportunities.
- `GET /api/ai/weekly-plan` — Conservative weekly budget cap reserving mandatory commitments.
- `GET /api/ai/model-evaluation` — Chronological 80/20 holdout evaluation metrics (Random Forest vs Moving Average).

### System & Health
- `GET /api/health` — Service health check.
- `GET /api/meta/categories` — Supported income and expense categories.

---

## Deployment on Vercel

SpendCount is configured for seamless deployment on Vercel:

### Option A: Complete Full-Stack Deployment on Vercel (Monorepo)

SpendCount includes a preconfigured root `vercel.json` and serverless bridge `api/index.py` that enables Vercel to build the frontend and run the Python FastAPI backend using Vercel Serverless Functions.

1. **Connect Repository**: Import `https://github.com/arpitaa21/SpendCount.git` in your Vercel Dashboard.
2. **Environment Variables**:
   - `DATABASE_URL`: Your managed PostgreSQL connection string (from Neon, Supabase, Render, or AWS RDS).
     *(Note: SQLite is not persistent in serverless environments as AWS Lambda / Vercel functions recycle filesystem state; managed PostgreSQL is required for production data persistence).*
   - `SECRET_KEY`: A secure 64-character random string.
   - `ACCESS_TOKEN_MINUTES`: `10080` (7 days).
3. **Deploy**: Vercel executes `cd frontend && npm install && npm run build`, maps static assets to `/`, and routes `/api/(.*)` to the serverless Python handler `api/index.py`.

### Option B: Frontend on Vercel + Backend on Render / Railway / Cloud Run

For long-running workloads, background training, and heavy ML processing, the recommended production architecture is hosting the FastAPI backend as a persistent service:

1. **Backend**:
   - Deploy `backend/` to Render or Railway using the included `backend/Dockerfile` or standard Python runtime (`uvicorn app.main:app --host 0.0.0.0 --port $PORT`).
   - Add environment variables: `DATABASE_URL` (PostgreSQL), `SECRET_KEY`, `CORS_ORIGINS`.
2. **Frontend on Vercel**:
   - Deploy `frontend/` on Vercel (Root Directory: `frontend`).
   - Build Command: `npm run build`
   - Output Directory: `dist`
   - Environment Variable: `VITE_API_URL=https://your-backend-service.onrender.com/api`
   - Client-side routing is handled by `frontend/vercel.json`.

---

## Running the Automated Test Suite

Run the complete 16-test suite verifying API isolation, CSV parsing, ML forecasting, and end-to-end user lifecycles:

```powershell
& "backend\.venv\Scripts\python.exe" -m pytest
```

```
============================= test session starts =============================
platform win32 -- Python 3.14.6, pytest-8.4.2
collected 16 items

backend\tests\test_api_isolation.py ...                                  [ 18%]
backend\tests\test_csv_logic.py ..                                       [ 31%]
backend\tests\test_e2e_flow.py .                                         [ 37%]
backend\tests\test_extended_api.py .......                               [ 81%]
backend\tests\test_ml.py ...                                             [100%]

============================= 16 passed in 6.25s ==============================
```

---

## Security & Privacy Guidelines

- **No Bank Credentials**: SpendCount never requests bank logins, netbanking passwords, UPI PINs, or card CVVs.
- **Multi-Tenant User Isolation**: Every database query is strictly scoped to the authenticated `user_id`. Users cannot view or modify transactions belonging to other students.
- **Secure Password Hashing**: Passwords are encrypted with PBKDF2-HMAC-SHA256 (240,000 rounds) using a cryptographically random salt.
- **CSV Upload Limits**: Uploads are restricted to UTF-8 `.csv` files under 5 MB with strict validation to prevent resource exhaustion.
- **Secret Protection**: `.env` files, SQLite databases (`*.db`), virtual environments, and build artifacts are strictly excluded by `.gitignore`.

---

## Limitations & Financial Disclaimer

- **Planning Estimates Only**: SpendCount provides cash-flow estimates based solely on entered data. Forecasts and recommendations are planning tools and do not constitute certified financial or investment advice.
- **Unconfirmed Income**: Future expected income is kept separate from conservative runway estimates to prevent over-optimistic budgeting.
- **Synthetic Training Data**: The baseline phrase corpus and demo transactions are synthetic benchmarks designed for reproducibility; real-world accuracy improves as users log and correct transactions over time.
