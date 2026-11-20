from datetime import date, datetime, timedelta
from calendar import monthrange
from statistics import median
from io import StringIO
from types import SimpleNamespace
import csv
import os
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import select, func, case, text
from .database import Base, engine, get_db
from .models import User, Transaction, Budget, IncomeSchedule, Forecast, Alert, PlannedCommitment, PlanningSetting, CategoryCorrection
from .schemas import (
    RegisterIn,
    LoginIn,
    ProfileUpdateIn,
    PasswordUpdateIn,
    TransactionIn,
    TransactionOut,
    BudgetIn,
    ScheduleIn,
    CommitmentIn,
    PlanningSettingsIn,
    EXPENSE_CATEGORIES,
    INCOME_CATEGORIES,
)
from .security import hash_password, verify_password, create_token, current_user
from .ml.categorisation import predict_category, predict_income_category
from .ml.forecasting import forecast
from .ml.anomaly_detection import find_anomalies
from .ml.recommendations import weekly_plan

Base.metadata.create_all(bind=engine)
# create_all does not alter existing tables, so keep the small additive schema
# changes safe for existing SQLite and PostgreSQL deployments.
with engine.begin() as _connection:
    if engine.dialect.name == "postgresql":
        _connection.execute(text("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS alert_key VARCHAR(120) NOT NULL DEFAULT 'legacy'"))
        _connection.execute(text("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS snoozed_until DATE"))
        _connection.execute(text("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE"))
    else:
        _columns = {column["name"] for column in __import__("sqlalchemy").inspect(_connection).get_columns("alerts")}
        for _column, _definition in (("alert_key", "VARCHAR(120) NOT NULL DEFAULT 'legacy'"), ("snoozed_until", "DATE"), ("is_active", "BOOLEAN NOT NULL DEFAULT TRUE")):
            if _column not in _columns:
                _connection.execute(text(f"ALTER TABLE alerts ADD COLUMN {_column} {_definition}"))
    try:
        _connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_alert_user_key ON alerts (user_id, alert_key) WHERE alert_key <> 'legacy'"))
    except Exception:
        pass
app = FastAPI(title="SpendCount API", version="1.0.0", description="Student cash-flow intelligence and budgeting API")

cors_origins_env = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173")
allowed_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def tx_query(db, user_id): return db.query(Transaction).filter(Transaction.user_id == user_id)

def predict_user_category(db, user_id, description):
    feedback = db.query(CategoryCorrection).filter(CategoryCorrection.user_id == user_id).order_by(CategoryCorrection.created_at.desc()).limit(500).all()
    examples = tuple((row.description, row.corrected_category) for row in feedback)
    return predict_category(description, examples)

def month_bounds(d=None):
    if d is None:
        d = date.today()
    return d.replace(day=1), d.replace(day=monthrange(d.year, d.month)[1])

def running_balance(items, as_of=None):
    as_of = as_of or date.today()
    return sum(t.amount if t.transaction_type == "income" else -t.amount for t in items if t.date <= as_of)

def balance_for(user_id, db, as_of=None):
    query = db.query(func.coalesce(func.sum(case((Transaction.transaction_type == "income", Transaction.amount), else_=-Transaction.amount)), 0.0)).filter(Transaction.user_id == user_id)
    if as_of is not None:
        query = query.filter(Transaction.date <= as_of)
    return float(query.scalar() or 0)

def advance_income_date(schedule, occurrence):
    frequency = schedule.recurring_frequency
    if frequency == "weekly":
        return occurrence + timedelta(days=7)
    months = 6 if frequency == "semester" else 1
    month_index = occurrence.year * 12 + occurrence.month - 1 + months
    year, month = divmod(month_index, 12)
    month += 1
    return date(year, month, min(occurrence.day, monthrange(year, month)[1]))

def income_events(schedules, today, end):
    events = []
    for schedule in schedules:
        occurrence = schedule.expected_date
        if occurrence < today:
            if schedule.recurring_frequency == "one-time":
                continue
            while occurrence < today:
                occurrence = advance_income_date(schedule, occurrence)
        while occurrence <= end:
            events.append({"date": occurrence.isoformat(), "amount": round(schedule.expected_amount, 2), "source": schedule.income_source})
            if schedule.recurring_frequency == "one-time":
                break
            occurrence = advance_income_date(schedule, occurrence)
    return sorted(events, key=lambda event: event["date"])

def recurring_expense_events(items, today, end):
    groups = {}
    for item in items:
        if item.transaction_type != "expense":
            continue
        key = (item.description.casefold().strip(), item.category)
        groups.setdefault(key, []).append(item)
    events = []
    observed = {}
    for (_, category), rows in groups.items():
        rows.sort(key=lambda row: row.date)
        if len(rows) < 2:
            continue
        intervals = [(right.date - left.date).days for left, right in zip(rows, rows[1:])]
        interval = median(intervals)
        if 5 <= interval <= 9:
            step = 7
        elif 25 <= interval <= 35:
            step = 30
        else:
            continue
        average_amount = sum(row.amount for row in rows) / len(rows)
        for row in rows:
            observed[row.id] = min(row.amount, average_amount)
        occurrence = rows[-1].date + timedelta(days=step)
        while occurrence <= today:
            occurrence += timedelta(days=step)
        while occurrence <= end:
            events.append({"date": occurrence.isoformat(), "amount": round(average_amount, 2), "description": rows[-1].description, "category": category})
            occurrence += timedelta(days=step)
    return sorted(events, key=lambda event: event["date"]), observed

def forecast_for(user_id, db):
    today = date.today()
    start, end = month_bounds(today)
    days = (end - today).days
    history_start = today - timedelta(days=179)
    daily_rows = db.query(Transaction.date, func.sum(Transaction.amount)).filter(Transaction.user_id == user_id, Transaction.transaction_type == "expense", Transaction.date >= today - timedelta(days=89), Transaction.date <= today).group_by(Transaction.date).all()
    all_dates = [today - timedelta(days=i) for i in range(89, -1, -1)]
    by_day = {d: 0.0 for d in all_dates}
    for tx_date, amount in daily_rows:
        by_day[tx_date] = float(amount or 0)
    expenses = tx_query(db, user_id).filter(Transaction.transaction_type == "expense", Transaction.date >= history_start, Transaction.date <= end).order_by(Transaction.date).all()
    items = expenses
    schedules = db.query(IncomeSchedule).filter(IncomeSchedule.user_id == user_id).order_by(IncomeSchedule.expected_date).all()
    expected_events = income_events(schedules, today, end)
    upcoming = None
    next_schedules = []
    for schedule in schedules:
        occurrence = schedule.expected_date
        if occurrence < today:
            if schedule.recurring_frequency == "one-time":
                continue
            while occurrence < today:
                occurrence = advance_income_date(schedule, occurrence)
        next_schedules.append((occurrence, schedule))
    if next_schedules:
        next_date, schedule = min(next_schedules, key=lambda item: item[0])
        upcoming = SimpleNamespace(expected_date=next_date, expected_amount=schedule.expected_amount, income_source=schedule.income_source)
    recurring_events, recurring_observed = recurring_expense_events(items, today, end)
    for tx in expenses:
        if tx.id in recurring_observed and tx.date in by_day:
            by_day[tx.date] = max(0, by_day[tx.date] - recurring_observed[tx.id])
    daily = list(by_day.values())
    current = balance_for(user_id, db, today)
    next_days = (upcoming.expected_date - today).days if upcoming else None
    expected = sum(event["amount"] for event in expected_events)
    result = forecast(daily, current, days, next_days, expected)
    recurring_total = sum(event["amount"] for event in recurring_events)
    result["expected_expenses"] = round(result["expected_expenses"] + recurring_total, 2)
    result["predicted_end_balance"] = round(result["predicted_end_balance"] - recurring_total, 2)
    result["predicted_with_expected_income"] = round(result["predicted_end_balance"] + expected, 2)
    result["daily_safe_limit"] = round(max(0, (current - recurring_total - result["daily_spend_rate"] * days * 0.55) / max(days, 1)), 2)
    result["expected_income_events"] = expected_events
    result["expected_recurring_expense_events"] = recurring_events
    return result, items, upcoming

@app.get("/api/health")
def health():
    try:
        with engine.connect() as connection:
            connection.execute(select(1))
    except Exception as exc:
        raise HTTPException(503, "Database unavailable") from exc
    return {"status": "ok", "service": "SpendCount API", "version": "1.0.0", "database": engine.dialect.name}

@app.post("/api/auth/register")
def register(data: RegisterIn, db: Session = Depends(get_db)):
    email = data.email.lower()
    if db.query(User).filter(func.lower(User.email) == email).first():
        raise HTTPException(409, "An account with this email already exists")
    user = User(name=data.name.strip(), email=email, hashed_password=hash_password(data.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"access_token": create_token(user.id), "token_type": "bearer", "user": {"id": user.id, "name": user.name, "email": user.email}}

@app.post("/api/auth/login")
def login(data: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(func.lower(User.email) == data.email.lower()).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(401, "Email or password is incorrect")
    return {"access_token": create_token(user.id), "token_type": "bearer", "user": {"id": user.id, "name": user.name, "email": user.email}}

@app.get("/api/auth/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "name": user.name, "email": user.email}

@app.put("/api/auth/profile")
def update_profile(data: ProfileUpdateIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    user.name = data.name.strip()
    db.commit()
    db.refresh(user)
    return {"id": user.id, "name": user.name, "email": user.email}

@app.put("/api/auth/password")
def update_password(data: PasswordUpdateIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    if not verify_password(data.old_password, user.hashed_password):
        raise HTTPException(400, "Current password is incorrect")
    user.hashed_password = hash_password(data.new_password)
    db.commit()
    return {"message": "Password updated successfully"}

@app.get("/api/transactions", response_model=list[TransactionOut])
def list_transactions(start_date: date | None = None, end_date: date | None = None, category: str | None = None, transaction_type: str | None = None, search: str | None = None, limit: int = Query(100, ge=1, le=500), offset: int = 0, db: Session = Depends(get_db), user: User = Depends(current_user)):
    q = tx_query(db, user.id)
    if start_date:
        q = q.filter(Transaction.date >= start_date)
    if end_date:
        q = q.filter(Transaction.date <= end_date)
    if category:
        q = q.filter(Transaction.category == category)
    if transaction_type:
        q = q.filter(Transaction.transaction_type == transaction_type)
    if search:
        q = q.filter(Transaction.description.ilike(f"%{search[:100]}%"))
    return q.order_by(Transaction.date.desc(), Transaction.id.desc()).offset(offset).limit(limit).all()

@app.post("/api/transactions", response_model=TransactionOut, status_code=201)
def create_transaction(data: TransactionIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    typ = data.transaction_type.lower()
    if typ not in ("income", "expense"):
        raise HTTPException(422, "Transaction type must be income or expense")
    allowed = INCOME_CATEGORIES if typ == "income" else EXPENSE_CATEGORIES
    category = data.category
    suggestion = None
    if category in ("", "Needs Review", "Auto"):
        suggestion = predict_income_category(data.description) if typ == "income" else predict_user_category(db, user.id, data.description)
        category = suggestion["category"]
    if category not in allowed and category != "Needs Review":
        raise HTTPException(422, "Category does not match transaction type")
    tx = Transaction(user_id=user.id, date=data.date, description=data.description.strip(), amount=data.amount, transaction_type=typ, category=category, payment_method=data.payment_method, notes=data.notes, source=data.source)
    db.add(tx)
    db.commit()
    db.refresh(tx)
    if typ == "expense" and category != "Needs Review":
        prediction = suggestion or predict_user_category(db, user.id, data.description)
        if category != prediction["predicted_category"]:
            db.add(CategoryCorrection(user_id=user.id, transaction_id=tx.id, description=tx.description, predicted_category=prediction["predicted_category"], corrected_category=category, confidence=prediction["confidence"]))
            db.commit()
    return tx

@app.put("/api/transactions/{tx_id}", response_model=TransactionOut)
def update_transaction(tx_id: int, data: TransactionIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    tx = tx_query(db, user.id).filter(Transaction.id == tx_id).first()
    if not tx:
        raise HTTPException(404, "Transaction not found")
    values = data.model_dump()
    typ = values["transaction_type"].lower()
    if typ not in ("income", "expense"):
        raise HTTPException(422, "Transaction type must be income or expense")
    allowed = INCOME_CATEGORIES if typ == "income" else EXPENSE_CATEGORIES
    if values["category"] not in allowed and values["category"] != "Needs Review":
        raise HTTPException(422, "Category does not match transaction type")
    previous_category = tx.category
    values["transaction_type"] = typ
    values["description"] = values["description"].strip()
    for k, v in values.items():
        setattr(tx, k, v)
    db.commit()
    db.refresh(tx)
    if tx.transaction_type == "expense" and tx.category not in ("Needs Review", previous_category):
        prediction = predict_user_category(db, user.id, tx.description)
        if tx.category != prediction["predicted_category"]:
            db.add(CategoryCorrection(user_id=user.id, transaction_id=tx.id, description=tx.description, predicted_category=prediction["predicted_category"], corrected_category=tx.category, confidence=prediction["confidence"]))
            db.commit()
    return tx

@app.delete("/api/transactions/{tx_id}", status_code=204)
def delete_transaction(tx_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    tx = tx_query(db, user.id).filter(Transaction.id == tx_id).first()
    if not tx:
        raise HTTPException(404, "Transaction not found")
    db.delete(tx)
    db.commit()

@app.post("/api/transactions/import")
async def import_csv(file: UploadFile = File(...), preview: bool = True, db: Session = Depends(get_db), user: User = Depends(current_user)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Upload a valid .csv file")
    try:
        raw_bytes = await file.read()
    except Exception:
        raise HTTPException(400, "Could not read uploaded file")
    # File upload size validation (Max 5 MB)
    if len(raw_bytes) > 5 * 1024 * 1024:
        raise HTTPException(413, "CSV file exceeds maximum allowed size of 5 MB")
    try:
        content = raw_bytes.decode("utf-8-sig")
    except Exception:
        raise HTTPException(400, "Could not decode CSV. Please save the file as UTF-8 encoded and try again.")
    if not content.strip():
        raise HTTPException(422, "Uploaded CSV file is empty")
    return parse_csv_content(content, preview, db, user)

def parse_csv_content(content: str, preview: bool, db: Session, user: User):
    reader = csv.DictReader(StringIO(content))
    required = {"date", "description", "amount", "type"}
    normalized_headers = [x.strip().lower() if x else "" for x in (reader.fieldnames or [])]
    fields = set(normalized_headers)
    if not required.issubset(fields):
        raise HTTPException(422, "CSV needs date, description, amount, and type columns")
    reader.fieldnames = normalized_headers
    existing = {(t.date.isoformat(), t.description.casefold(), round(t.amount, 2), t.transaction_type) for t in tx_query(db, user.id).all()}
    accepted = []
    rejected = []
    seen = set()
    for line, row in enumerate(reader, start=2):
        row = {str(k).strip().lower(): (v or "").strip() for k, v in row.items() if k}
        try:
            dt = date.fromisoformat(row["date"][:10])
            desc = row["description"][:250]
            amt = float(row["amount"].replace(",", ""))
            typ = row["type"].lower()
            if not desc:
                raise ValueError("description is required")
            if amt <= 0:
                raise ValueError("amount must be greater than zero")
            if typ not in ("income", "expense"):
                raise ValueError("type must be income or expense")
            cat = row.get("category", "")
            if not cat:
                pred = predict_income_category(desc) if typ == "income" else predict_user_category(db, user.id, desc)
                cat = pred["category"]
            allowed = INCOME_CATEGORIES if typ == "income" else EXPENSE_CATEGORIES
            if cat not in allowed and cat != "Needs Review":
                raise ValueError(f"unknown {typ} category: {cat}")
            key = (dt.isoformat(), desc.casefold(), round(amt, 2), typ)
            if key in existing or key in seen:
                raise ValueError("duplicate transaction")
            seen.add(key)
            accepted.append({"date": dt.isoformat(), "description": desc, "amount": amt, "transaction_type": typ, "category": cat, "source": "csv"})
        except Exception as e:
            rejected.append({"row": line, "reason": str(e), "data": row})
    if not preview:
        for item in accepted:
            row_data = dict(item)
            row_data["date"] = date.fromisoformat(row_data["date"])
            db.add(Transaction(user_id=user.id, **row_data))
        db.commit()
    return {
        "preview": preview,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
        "message": "Review the preview, then upload again with preview=false to import valid rows." if preview else f"Imported {len(accepted)} rows; {len(rejected)} rows need attention.",
    }

@app.get("/api/transactions/template")
def csv_template():
    return {
        "filename": "spendcount-template.csv",
        "content": "date,description,amount,type,category\n2026-11-01,Campus cafe lunch,185,expense,Food\n2026-11-02,Family allowance,8000,income,Family Allowance\n",
    }

@app.get("/api/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db), user: User = Depends(current_user)):
    today = date.today()
    start, end = month_bounds(today)
    row = db.query(
        func.coalesce(func.sum(case((Transaction.transaction_type == "income", Transaction.amount), else_=0)), 0),
        func.coalesce(func.sum(case((Transaction.transaction_type == "expense", Transaction.amount), else_=0)), 0),
    ).filter(Transaction.user_id == user.id, Transaction.date >= start, Transaction.date <= today).one()
    income, spent = float(row[0]), float(row[1])
    fc, _, upcoming = forecast_for(user.id, db)
    budgets = db.query(Budget).filter(Budget.user_id == user.id).all()
    limit = sum(b.monthly_limit for b in budgets)
    current = balance_for(user.id, db, today)
    remaining_days = (end - today).days
    health = "On track" if fc["predicted_end_balance"] > 0 and (not limit or spent < limit * 0.8) else "Needs attention"
    return {
        "balance": round(current, 2),
        "month_income": round(income, 2),
        "month_expenses": round(spent, 2),
        "remaining_budget": round(max(0, limit - spent), 2) if limit else None,
        "monthly_budget": round(limit, 2) if limit else None,
        "predicted_end_balance": fc["predicted_end_balance"],
        "predicted_with_expected_income": fc["predicted_with_expected_income"],
        "days_until_income": (upcoming.expected_date - today).days if upcoming else None,
        "next_income_date": upcoming.expected_date.isoformat() if upcoming else None,
        "daily_safe_limit": fc["daily_safe_limit"],
        "health": health,
        "forecast_confidence": fc["confidence"],
        "remaining_days_in_month": remaining_days,
        "has_budget": bool(budgets),
    }

@app.get("/api/dashboard/spending-breakdown")
def spending_breakdown(db: Session = Depends(get_db), user: User = Depends(current_user)):
    start, _ = month_bounds()
    rows = db.query(Transaction.category, func.sum(Transaction.amount).label("amount")).filter(Transaction.user_id == user.id, Transaction.date >= start, Transaction.transaction_type == "expense").group_by(Transaction.category).order_by(func.sum(Transaction.amount).desc()).all()
    return [{"category": category, "amount": round(float(amount), 2)} for category, amount in rows]

@app.get("/api/dashboard/spending-trends")
def spending_trends(db: Session = Depends(get_db), user: User = Depends(current_user)):
    today = date.today()
    first = today.replace(day=1)
    items = db.query(Transaction.date, Transaction.transaction_type, func.sum(Transaction.amount)).filter(Transaction.user_id == user.id, Transaction.date >= first, Transaction.date <= today).group_by(Transaction.date, Transaction.transaction_type).all()
    daily = {}
    for i in range((today - first).days + 1):
        daily[(first + timedelta(days=i)).isoformat()] = {"date": (first + timedelta(days=i)).isoformat(), "income": 0, "expenses": 0}
    for tx_date, tx_type, amount in items:
        if tx_date.isoformat() in daily:
            daily[tx_date.isoformat()]["income" if tx_type == "income" else "expenses"] += float(amount or 0)
    return list(daily.values())

@app.get("/api/budgets")
def get_budgets(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return db.query(Budget).filter(Budget.user_id == user.id).order_by(Budget.category).all()

@app.get("/api/budgets/usage")
def budget_usage(db: Session = Depends(get_db), user: User = Depends(current_user)):
    start, today = month_bounds()
    rows = db.query(Transaction.category, func.sum(Transaction.amount)).filter(Transaction.user_id == user.id, Transaction.transaction_type == "expense", Transaction.date >= start, Transaction.date <= today).group_by(Transaction.category).all()
    return {category: round(float(amount or 0), 2) for category, amount in rows}

@app.post("/api/budgets", status_code=201)
def create_budget(data: BudgetIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    if data.category not in EXPENSE_CATEGORIES:
        raise HTTPException(422, "Choose a valid expense category")
    item = db.query(Budget).filter(Budget.user_id == user.id, Budget.category == data.category).first()
    if item:
        item.monthly_limit = data.monthly_limit
    else:
        item = Budget(user_id=user.id, **data.model_dump())
        db.add(item)
    db.commit()
    db.refresh(item)
    return item

@app.put("/api/budgets/{budget_id}")
def update_budget(budget_id: int, data: BudgetIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = db.query(Budget).filter(Budget.id == budget_id, Budget.user_id == user.id).first()
    if not item:
        raise HTTPException(404, "Budget not found")
    item.category = data.category
    item.monthly_limit = data.monthly_limit
    db.commit()
    db.refresh(item)
    return item

@app.delete("/api/budgets/{budget_id}", status_code=204)
def delete_budget(budget_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = db.query(Budget).filter(Budget.id == budget_id, Budget.user_id == user.id).first()
    if not item:
        raise HTTPException(404, "Budget not found")
    db.delete(item)
    db.commit()

@app.get("/api/income-schedules")
def get_schedules(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return db.query(IncomeSchedule).filter(IncomeSchedule.user_id == user.id).order_by(IncomeSchedule.expected_date).all()

@app.post("/api/income-schedules", status_code=201)
def add_schedule(data: ScheduleIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = IncomeSchedule(user_id=user.id, **data.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

@app.delete("/api/income-schedules/{schedule_id}", status_code=204)
def delete_schedule(schedule_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = db.query(IncomeSchedule).filter(IncomeSchedule.id == schedule_id, IncomeSchedule.user_id == user.id).first()
    if not item:
        raise HTTPException(404, "Schedule not found")
    db.delete(item)
    db.commit()

@app.get("/api/planning/commitments")
def list_commitments(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return db.query(PlannedCommitment).filter(PlannedCommitment.user_id == user.id).order_by(PlannedCommitment.due_date).all()

@app.post("/api/planning/commitments", status_code=201)
def add_commitment(data: CommitmentIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = PlannedCommitment(user_id=user.id, **data.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

@app.delete("/api/planning/commitments/{commitment_id}", status_code=204)
def delete_commitment(commitment_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = db.query(PlannedCommitment).filter(PlannedCommitment.id == commitment_id, PlannedCommitment.user_id == user.id).first()
    if not item:
        raise HTTPException(404, "Commitment not found")
    db.delete(item)
    db.commit()

@app.get("/api/planning/settings")
def get_planning_settings(db: Session = Depends(get_db), user: User = Depends(current_user)):
    row = db.query(PlanningSetting).filter(PlanningSetting.user_id == user.id).first()
    return {"emergency_buffer_target": row.emergency_buffer_target if row else 0}

@app.put("/api/planning/settings")
def update_planning_settings(data: PlanningSettingsIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    row = db.query(PlanningSetting).filter(PlanningSetting.user_id == user.id).first()
    if row is None:
        row = PlanningSetting(user_id=user.id, emergency_buffer_target=data.emergency_buffer_target)
        db.add(row)
    else:
        row.emergency_buffer_target = data.emergency_buffer_target
    db.commit()
    return {"emergency_buffer_target": row.emergency_buffer_target}

@app.get("/api/forecast/month-end")
def month_end(db: Session = Depends(get_db), user: User = Depends(current_user)):
    result, _, _ = forecast_for(user.id, db)
    return result

@app.get("/api/forecast/cash-flow")
def cash_flow(db: Session = Depends(get_db), user: User = Depends(current_user)):
    result, items, _ = forecast_for(user.id, db)
    today = date.today()
    balance = balance_for(user.id, db, today)
    daily = result["daily_spend_rate"]
    points = []
    for n in range(0, result["days_remaining"] + 1, max(1, result["days_remaining"] // 12)):
        point_date = today + timedelta(days=n)
        expenses = sum(event["amount"] for event in result["expected_recurring_expense_events"] if today < date.fromisoformat(event["date"]) <= point_date)
        income = sum(event["amount"] for event in result["expected_income_events"] if today < date.fromisoformat(event["date"]) <= point_date)
        projected = balance - daily * n - expenses
        points.append({"date": point_date.isoformat(), "projected_balance": round(projected, 2), "with_expected_income": round(projected + income, 2)})
    return {"forecast": result, "points": points}

@app.post("/api/forecast/recalculate")
def recalculate(db: Session = Depends(get_db), user: User = Depends(current_user)):
    result, _, _ = forecast_for(user.id, db)
    row = Forecast(user_id=user.id, forecast_date=date.today(), predicted_balance=result["predicted_end_balance"], expected_expenses=result["expected_expenses"], confidence_or_uncertainty=result["uncertainty"], model_version=result["model_version"])
    db.add(row)
    db.commit()
    return result

@app.post("/api/ai/categorise")
def categorise(payload: dict, db: Session = Depends(get_db), user: User = Depends(current_user)):
    description = str(payload.get("description", "")).strip()
    if not description:
        raise HTTPException(422, "Description is required")
    return predict_income_category(description[:250]) if payload.get("transaction_type") == "income" else predict_user_category(db, user.id, description[:250])

@app.get("/api/ai/corrections")
def category_corrections(db: Session = Depends(get_db), user: User = Depends(current_user)):
    rows = db.query(CategoryCorrection).filter(CategoryCorrection.user_id == user.id).order_by(CategoryCorrection.created_at.desc()).limit(500).all()
    return [{"id": x.id, "description": x.description, "predicted_category": x.predicted_category, "corrected_category": x.corrected_category, "confidence": x.confidence, "created_at": x.created_at.isoformat()} for x in rows]

@app.delete("/api/ai/corrections/{correction_id}", status_code=204)
def delete_correction(correction_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    row = db.query(CategoryCorrection).filter(CategoryCorrection.id == correction_id, CategoryCorrection.user_id == user.id).first()
    if not row:
        raise HTTPException(404, "Correction not found")
    db.delete(row)
    db.commit()

@app.get("/api/ai/anomalies")
def anomalies(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return find_anomalies(tx_query(db, user.id).all())

@app.get("/api/ai/alerts")
def personalized_alerts(db: Session = Depends(get_db), user: User = Depends(current_user)):
    today = date.today()
    start, end = month_bounds(today)
    spent_rows = db.query(Transaction.category, func.sum(Transaction.amount)).filter(Transaction.user_id == user.id, Transaction.transaction_type == "expense", Transaction.date >= start, Transaction.date <= today).group_by(Transaction.category).all()
    spent_by_category = {category: float(amount or 0) for category, amount in spent_rows}
    days_elapsed = today.day
    days_left = (end - today).days
    alerts = []
    for budget in db.query(Budget).filter(Budget.user_id == user.id).all():
        spent = spent_by_category.get(budget.category, 0)
        pace = spent / max(1, days_elapsed)
        projected = pace * end.day
        if spent >= budget.monthly_limit:
            alerts.append({"alert_id": f"budget-over-{budget.category}", "title": f"{budget.category} budget reached", "description": f"You have spent ₹{spent:,.0f} of your ₹{budget.monthly_limit:,.0f} monthly limit.", "severity": "warning", "amount": round(spent, 2)})
        elif projected > budget.monthly_limit and days_left > 0:
            per_day = max(0.01, pace)
            days_to_limit = max(1, int((budget.monthly_limit - spent) / per_day))
            alerts.append({"alert_id": f"budget-pace-{budget.category}", "title": f"{budget.category} budget may run over", "description": f"At your current pace, the ₹{budget.monthly_limit:,.0f} limit could be reached in about {days_to_limit} days.", "severity": "warning", "amount": round(projected, 2)})
    fc, _, upcoming = forecast_for(user.id, db)
    if fc["predicted_end_balance"] < 0:
        alerts.append({"alert_id": "cash-flow-shortfall", "title": "Cash-flow pinch ahead", "description": f"Your current spending pace projects a ₹{abs(fc['predicted_end_balance']):,.0f} shortfall by month end, before unconfirmed income.", "severity": "critical", "amount": fc["predicted_end_balance"]})
    if upcoming:
        days = (upcoming.expected_date - today).days
        if days <= 14:
            alerts.append({"alert_id": "next-income", "title": "Next expected income", "description": f"{upcoming.income_source} of ₹{upcoming.expected_amount:,.0f} is expected in {days} days. Treat it as an estimate until it arrives.", "severity": "info", "amount": upcoming.expected_amount})
    if fc["daily_spend_rate"] > fc["daily_safe_limit"] > 0:
        alerts.append({"alert_id": "daily-spending-pace", "title": "Spending pace is above your daily guide", "description": f"Your recent average is ₹{fc['daily_spend_rate']:,.0f} a day, above the estimated safe limit of ₹{fc['daily_safe_limit']:,.0f}.", "severity": "warning", "amount": fc["daily_spend_rate"]})
    priority = {"critical": 0, "warning": 1, "info": 2}
    alerts = sorted(alerts, key=lambda alert: (priority[alert["severity"]], alert["alert_id"]))[:6]
    db.query(Alert).filter(Alert.user_id == user.id).update({Alert.is_active: False}, synchronize_session=False)
    results = []
    for data in alerts:
        key = data["alert_id"]
        record = db.query(Alert).filter(Alert.user_id == user.id, Alert.alert_key == key).first()
        if record is None:
            record = Alert(user_id=user.id, alert_key=key, alert_type=key.split("-")[0], **{k: data[k] for k in ("title", "description", "severity")})
            db.add(record)
        else:
            record.title, record.description, record.severity = data["title"], data["description"], data["severity"]
            record.is_active = True
        if record.snoozed_until and record.snoozed_until < today:
            record.snoozed_until = None
        db.flush()
        if not record.snoozed_until or record.snoozed_until < today:
            results.append({**data, "id": record.id, "is_read": record.is_read, "snoozed_until": record.snoozed_until.isoformat() if record.snoozed_until else None})
    db.commit()
    return results

@app.get("/api/alerts")
def list_persisted_alerts(db: Session = Depends(get_db), user: User = Depends(current_user)):
    personalized_alerts(db, user)
    rows = db.query(Alert).filter(Alert.user_id == user.id, Alert.is_active.is_(True), (Alert.snoozed_until.is_(None) | (Alert.snoozed_until < date.today()))).order_by(Alert.is_read, Alert.created_at.desc()).all()
    return [{"id": row.id, "alert_id": row.alert_key, "title": row.title, "description": row.description, "severity": row.severity, "is_read": row.is_read, "snoozed_until": row.snoozed_until.isoformat() if row.snoozed_until else None, "created_at": row.created_at.isoformat()} for row in rows]

@app.patch("/api/alerts/{alert_id}/read")
def mark_alert_read(alert_id: int, db: Session = Depends(get_db), user: User = Depends(current_user)):
    row = db.query(Alert).filter(Alert.id == alert_id, Alert.user_id == user.id).first()
    if not row:
        raise HTTPException(404, "Alert not found")
    row.is_read = True
    db.commit()
    return {"id": row.id, "is_read": row.is_read}

@app.post("/api/alerts/{alert_id}/snooze")
def snooze_alert(alert_id: int, days: int = Query(7, ge=1, le=30), db: Session = Depends(get_db), user: User = Depends(current_user)):
    row = db.query(Alert).filter(Alert.id == alert_id, Alert.user_id == user.id).first()
    if not row:
        raise HTTPException(404, "Alert not found")
    row.snoozed_until = date.today() + timedelta(days=days)
    row.is_read = True
    db.commit()
    return {"id": row.id, "snoozed_until": row.snoozed_until.isoformat()}

@app.get("/api/ai/insights")
def insights(db: Session = Depends(get_db), user: User = Depends(current_user)):
    expenses_query = tx_query(db, user.id).filter(Transaction.transaction_type == "expense")
    expense_count = expenses_query.count()
    if not expense_count:
        return {"empty": True, "insights": []}
    today = date.today()
    start, end = month_bounds(today)
    totals = {category: float(amount or 0) for category, amount in db.query(Transaction.category, func.sum(Transaction.amount)).filter(Transaction.user_id == user.id, Transaction.transaction_type == "expense", Transaction.date >= start, Transaction.date <= today).group_by(Transaction.category).all()}
    daytotals = {tx_date: float(amount or 0) for tx_date, amount in db.query(Transaction.date, func.sum(Transaction.amount)).filter(Transaction.user_id == user.id, Transaction.transaction_type == "expense", Transaction.date >= start, Transaction.date <= today).group_by(Transaction.date).all()}
    top = max(totals, key=totals.get) if totals else None
    last7 = float(expenses_query.filter(Transaction.date >= today - timedelta(days=6), Transaction.date <= today).with_entities(func.coalesce(func.sum(Transaction.amount), 0)).scalar() or 0)
    prev7 = float(expenses_query.filter(Transaction.date >= today - timedelta(days=13), Transaction.date < today - timedelta(days=6)).with_entities(func.coalesce(func.sum(Transaction.amount), 0)).scalar() or 0)
    fc, _, _ = forecast_for(user.id, db)
    insights_list = []
    if top:
        insights_list.append({"title": f"{top} leads this month", "detail": f"₹{totals[top]:,.0f} so far, the largest category in your current-month spending."})
    if daytotals:
        d = max(daytotals, key=daytotals.get)
        insights_list.append({"title": "Your highest-spend day", "detail": f"{d.strftime('%b %d')} had ₹{daytotals[d]:,.0f} in expenses."})
    insights_list.append({
        "title": "Weekly spending pace",
        "detail": f"You spent ₹{last7:,.0f} in the last 7 days, {('up' if last7 >= prev7 else 'down')} {abs((last7 - prev7) / prev7 * 100):.0f}% vs. the previous week." if prev7 else f"₹{last7:,.0f} spent in the last 7 days; more history is needed for a comparison.",
    })
    if fc["predicted_end_balance"] < 0:
        insights_list.append({"title": "Cash-flow pinch ahead", "detail": f"Current spending pace projects a ₹{abs(fc['predicted_end_balance']):,.0f} shortfall by month end, excluding unconfirmed income."})
    recurring_rows = db.query(func.min(Transaction.description), func.count(Transaction.id), func.sum(Transaction.amount)).filter(Transaction.user_id == user.id, Transaction.transaction_type == "expense").group_by(func.lower(func.trim(Transaction.description))).having(func.count(Transaction.id) >= 3).order_by(func.sum(Transaction.amount).desc()).limit(5).all()
    repeats = [{"description": description, "count": int(count), "total": round(float(total), 2)} for description, count, total in recurring_rows]
    budgets = {b.category: b.monthly_limit for b in db.query(Budget).filter(Budget.user_id == user.id).all()}
    elapsed = max(1, today.day)
    month_days = end.day
    budget_risks = []
    for category, limit in budgets.items():
        spent = totals.get(category, 0)
        projected = spent / elapsed * month_days
        if spent >= limit * 0.8 or projected > limit:
            budget_risks.append({"category": category, "spent": round(spent, 2), "limit": round(limit, 2), "projected": round(projected, 2), "overrun": round(max(0, projected - limit), 2)})
    budget_risks.sort(key=lambda item: item["overrun"], reverse=True)
    discretionary = [(category, amount) for category, amount in totals.items() if category in ("Food", "Shopping", "Entertainment")]
    savings = [{"category": category, "potential_savings": round(amount * 0.1, 2), "reason": f"A 10% reduction in this month's {category.lower()} spending would free about ₹{amount * 0.1:,.0f}."} for category, amount in sorted(discretionary, key=lambda item: -item[1])[:3]]
    weekly_spending = []
    for weeks_ago in range(3, -1, -1):
        week_end = today - timedelta(days=weeks_ago * 7)
        week_start = week_end - timedelta(days=6)
        amount = float(expenses_query.filter(Transaction.date >= week_start, Transaction.date <= week_end).with_entities(func.coalesce(func.sum(Transaction.amount), 0)).scalar() or 0)
        weekly_spending.append({"week": "This week" if weeks_ago == 0 else f"{weeks_ago} wk ago", "amount": round(amount, 2)})
    return {
        "empty": False,
        "highest_category": top,
        "most_expensive_day": max(daytotals, key=daytotals.get).isoformat() if daytotals else None,
        "average_daily_spending": round(float(expenses_query.with_entities(func.coalesce(func.sum(Transaction.amount), 0)).scalar() or 0) / max(1, (today - expenses_query.with_entities(func.min(Transaction.date)).scalar()).days + 1), 2),
        "week_over_week_change_pct": round((last7 - prev7) / prev7 * 100, 1) if prev7 else None,
        "recurring_expenses": repeats[:5],
        "weekly_spending": weekly_spending,
        "budget_risks": budget_risks,
        "savings_opportunities": savings,
        "insights": insights_list,
    }

@app.get("/api/ai/weekly-plan")
def get_weekly_plan(db: Session = Depends(get_db), user: User = Depends(current_user), delay_income_days: int = 0):
    if not 0 <= delay_income_days <= 30:
        raise HTTPException(422, "Income delay must be between 0 and 30 days")
    fc, items, upcoming = forecast_for(user.id, db)
    today = date.today()
    balance = balance_for(user.id, db, today)
    budgets = db.query(Budget).filter(Budget.user_id == user.id).all()
    first, _ = month_bounds(today)
    spent_rows = db.query(Transaction.category, func.sum(Transaction.amount)).filter(Transaction.user_id == user.id, Transaction.transaction_type == "expense", Transaction.date >= first, Transaction.date <= today).group_by(Transaction.category).all()
    spent = {category: float(amount or 0) for category, amount in spent_rows}
    days = (upcoming.expected_date - today).days if upcoming else min(7, month_bounds(today)[1].day - today.day)
    days = max(0, days) + delay_income_days
    mandatory = sum(b.monthly_limit * 0.08 for b in budgets if b.category in ("Rent and Hostel", "Subscriptions"))
    commitment_rows = db.query(PlannedCommitment).filter(PlannedCommitment.user_id == user.id).all()
    commitments = []
    commitment_total = 0.0
    for item in commitment_rows:
        occurrence = item.due_date
        if occurrence < today and item.recurring_frequency == "one-time":
            continue
        while occurrence < today:
            occurrence = occurrence + timedelta(days=7) if item.recurring_frequency == "weekly" else advance_income_date(SimpleNamespace(recurring_frequency="monthly"), occurrence)
        while occurrence <= today + timedelta(days=30):
            commitments.append({"id": item.id, "title": item.title, "category": item.category, "amount": item.amount, "due_date": occurrence.isoformat()})
            if occurrence <= today + timedelta(days=7):
                commitment_total += item.amount
            if item.recurring_frequency == "weekly": occurrence += timedelta(days=7)
            elif item.recurring_frequency == "monthly": occurrence = advance_income_date(SimpleNamespace(recurring_frequency="monthly"), occurrence)
            else: break
    setting = db.query(PlanningSetting).filter(PlanningSetting.user_id == user.id).first()
    buffer_target = setting.emergency_buffer_target if setting else 0
    plan = weekly_plan(balance, days, fc["daily_spend_rate"], {b.category: b.monthly_limit for b in budgets}, spent, mandatory + commitment_total + buffer_target / 4)
    if delay_income_days:
        scenario_factor = 7 / (7 + delay_income_days * 0.5)
        plan["weekly_budget"] = round(plan["weekly_budget"] * scenario_factor, 2)
        plan["daily_limit"] = round(plan["weekly_budget"] / min(max(1, days), 7), 2)
        for suggestion in plan["suggestions"]:
            suggestion["weekly_target"] = round(suggestion["weekly_target"] * scenario_factor, 2)
        plan["explanation"] += f" This late-income scenario trims the flexible cap by {round((1 - scenario_factor) * 100)}% to extend your runway."
    plan.update({"delay_income_days": delay_income_days, "income_delay_scenario": "Current schedule" if not delay_income_days else f"If expected income arrives {delay_income_days} days late", "commitments_next_week": commitments, "commitments_reserved": round(commitment_total, 2), "protected_upcoming_expenses": round(mandatory + commitment_total, 2), "emergency_buffer_target": round(buffer_target, 2), "emergency_buffer_gap": round(max(0, buffer_target - max(0, balance)), 2), "current_balance": round(balance, 2)})
    return plan

@app.get("/api/ai/model-evaluation")
def model_evaluation(db: Session = Depends(get_db), user: User = Depends(current_user)):
    from .ml.model_evaluation import compare_models
    items = tx_query(db, user.id).order_by(Transaction.date).all()
    expenses = [t for t in items if t.transaction_type == "expense"]
    if len(expenses) < 30:
        try:
            from pathlib import Path
            synth_path = Path(__file__).resolve().parents[2] / "dataset" / "synthetic_transactions.csv"
            if synth_path.exists():
                with open(synth_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    by_day = {}
                    for row in reader:
                        if row.get("type") == "expense":
                            d = row.get("date")
                            by_day[d] = by_day.get(d, 0.0) + float(row.get("amount", 0))
                    daily_series = [v for k, v in sorted(by_day.items())]
                    res = compare_models(daily_series)
                    res["data_source"] = "synthetic_benchmark"
                    return res
        except Exception:
            pass
    by_day = {}
    for t in expenses:
        by_day[t.date.isoformat()] = by_day.get(t.date.isoformat(), 0.0) + t.amount
    daily_series = [v for k, v in sorted(by_day.items())]
    res = compare_models(daily_series)
    res["data_source"] = "user_transactions"
    return res

@app.get("/api/meta/categories")
def categories():
    return {"expense": EXPENSE_CATEGORIES, "income": INCOME_CATEGORIES}
