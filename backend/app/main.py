from datetime import date, datetime, timedelta
from calendar import monthrange
from statistics import median
from io import StringIO
from types import SimpleNamespace
import csv
import os
import re
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query, Body
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
    DocumentImportIn,
    ExpenseDiaryIn,
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
        _connection.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_data_url TEXT"))
        _connection.execute(text("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS alert_key VARCHAR(120) NOT NULL DEFAULT 'legacy'"))
        _connection.execute(text("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS snoozed_until DATE"))
        _connection.execute(text("ALTER TABLE alerts ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE"))
    else:
        _user_columns = {column["name"] for column in __import__("sqlalchemy").inspect(_connection).get_columns("users")}
        if "avatar_data_url" not in _user_columns:
            _connection.execute(text("ALTER TABLE users ADD COLUMN avatar_data_url TEXT"))
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

def expense_categories_for_user(db, user_id):
    budget_categories = db.query(Budget.category).filter(Budget.user_id == user_id).distinct().all()
    transaction_categories = db.query(Transaction.category).filter(Transaction.user_id == user_id, Transaction.transaction_type == "expense").distinct().all()
    custom = {name for (name,) in budget_categories + transaction_categories if name and name not in EXPENSE_CATEGORIES}
    return [*EXPENSE_CATEGORIES, *sorted(custom, key=str.casefold)]

def valid_transaction_category(db, user_id, transaction_type, category):
    if category == "Needs Review":
        return True
    if transaction_type == "income":
        return category in INCOME_CATEGORIES
    return category in expense_categories_for_user(db, user_id)

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
    return {"access_token": create_token(user.id), "token_type": "bearer", "user": {"id": user.id, "name": user.name, "email": user.email, "avatar_data_url": user.avatar_data_url}}

@app.post("/api/auth/login")
def login(data: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(func.lower(User.email) == data.email.lower()).first()
    if not user or not verify_password(data.password, user.hashed_password):
        raise HTTPException(401, "Email or password is incorrect")
    return {"access_token": create_token(user.id), "token_type": "bearer", "user": {"id": user.id, "name": user.name, "email": user.email, "avatar_data_url": user.avatar_data_url}}

@app.get("/api/auth/me")
def me(user: User = Depends(current_user)):
    return {"id": user.id, "name": user.name, "email": user.email, "avatar_data_url": user.avatar_data_url}

@app.put("/api/auth/profile")
def update_profile(data: ProfileUpdateIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    user.name = data.name.strip()
    if data.avatar_data_url is not None:
        if not data.avatar_data_url.startswith(("data:image/jpeg;base64,", "data:image/png;base64,", "data:image/webp;base64,")):
            raise HTTPException(422, "Profile photo must be a JPEG, PNG, or WebP image")
        user.avatar_data_url = data.avatar_data_url
    db.commit()
    db.refresh(user)
    return {"id": user.id, "name": user.name, "email": user.email, "avatar_data_url": user.avatar_data_url}

@app.delete("/api/auth/profile/avatar")
def remove_profile_avatar(db: Session = Depends(get_db), user: User = Depends(current_user)):
    user.avatar_data_url = None
    db.commit()
    db.refresh(user)
    return {"id": user.id, "name": user.name, "email": user.email, "avatar_data_url": user.avatar_data_url}

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
    category = data.category
    suggestion = None
    if category in ("", "Needs Review", "Auto"):
        suggestion = predict_income_category(data.description) if typ == "income" else predict_user_category(db, user.id, data.description)
        category = suggestion["category"]
    if not valid_transaction_category(db, user.id, typ, category):
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
    if not valid_transaction_category(db, user.id, typ, values["category"]):
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
            if not valid_transaction_category(db, user.id, typ, cat):
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

def parse_document_date(value):
    value = re.sub(r"[,·•]", " ", value).strip()
    formats = ("%d %b %Y", "%d %B %Y", "%d %b", "%d %B", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d/%m", "%d-%m")
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt).date()
            if "%Y" not in fmt:
                parsed = parsed.replace(year=date.today().year)
            return parsed
        except ValueError:
            pass
    return None

@app.post("/api/transactions/import-document-text")
def import_document_text(data: dict = Body(...), db: Session = Depends(get_db), user: User = Depends(current_user)):
    content = data.get("text", "")
    preview = bool(data.get("preview", True))
    if not isinstance(content, str) or not content.strip():
        raise HTTPException(422, "No readable text was found. Try a clearer image or text-based PDF.")
    if len(content) > 1_000_000:
        raise HTTPException(413, "Extracted document text is too large")
    existing = {(t.date.isoformat(), t.description.casefold(), round(t.amount, 2), t.transaction_type) for t in tx_query(db, user.id).all()}
    accepted, rejected, seen, handled_lines, card_occurrences = [], [], set(), set(), {}
    lines = [" ".join(line.split()) for line in content.splitlines()]
    date_re = re.compile(r"(?<!\d)(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}(?:[-/]\d{2,4})?|\d{1,2}[-\s]+[A-Za-z]{3,9}(?:[-\s]+\d{2,4})?)(?!\d)", re.I)
    card_re = re.compile(r"\b(?:payment\s+to\s+(?:you|[A-Za-z][A-Za-z0-9 .'-]{1,50})|payment\s+from\s+[A-Za-z][A-Za-z0-9 .'-]{1,50}|payment\s+received|money\s+received|received\s+from\s+[A-Za-z][A-Za-z0-9 .'-]{1,50}|sent\s+to\s+[A-Za-z][A-Za-z0-9 .'-]{1,50})\b", re.I)
    # Tesseract's sparse-text mode detects amounts and dates on these UPI cards,
    # but its plain text mode can reorder those lines away from their headings.
    paid_lines = [i for i, line in enumerate(lines) if re.search(r"\bpaid\b", line, re.I)]
    has_payment_heading = any(re.search(r"\bpayment\s+(?:to|from)\b|\b(?:money\s+received|payment\s+received)\b", line, re.I) for line in lines)
    if len(paid_lines) >= 2 and has_payment_heading:
        boundaries = [-1, *paid_lines]
        amount_re = re.compile(r"(?:₹|rs\.?|inr|\br)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", re.I)
        for boundary_index in range(1, len(boundaries)):
            start, end = boundaries[boundary_index - 1] + 1, boundaries[boundary_index]
            block = lines[start:end + 1]
            handled_lines.update(range(start, end + 1))
            joined = " ".join(block)
            amount_match = next((match for line in block if (match := amount_re.search(line))), None)
            amount = float(amount_match.group(1).replace(",", "")) if amount_match else None
            date_match = next((match for line in block if (match := date_re.search(line))), None)
            parsed_date = parse_document_date(date_match.group(1)) if date_match else None
            heading = next((line.strip() for line in block if re.search(r"\bpayment\s+(?:to|from)\b|\b(?:money\s+received|payment\s+received)\b", line, re.I)), "")
            if re.fullmatch(r"payment\s+to", heading, re.I):
                heading_index = block.index(heading)
                recipient = next((line.strip() for line in block[heading_index + 1:] if re.fullmatch(r"[A-Za-z][A-Za-z0-9 .'-]{1,50}", line.strip())), "")
                if recipient:
                    heading = f"{heading} {recipient}"
            heading_lower = heading.casefold()
            typ = "income" if ("to you" in heading_lower or "received" in heading_lower or "payment from" in heading_lower) else "expense"
            if "to you" in heading_lower or "received" in heading_lower or "payment from" in heading_lower:
                description = "Payment received"
            elif heading:
                description = heading[:250]
            else:
                description = "Imported payment (review details)"
            review_reason = "Some details could not be read. Fill in the date and amount before importing." if amount is None or amount <= 0 or parsed_date is None else None
            occurrence_key = (parsed_date.isoformat() if parsed_date else "", description.casefold(), round(amount or 0, 2), typ)
            card_occurrences[occurrence_key] = card_occurrences.get(occurrence_key, 0) + 1
            if card_occurrences[occurrence_key] > 1:
                description = f"{description[:230]} ({card_occurrences[occurrence_key]})"
            accepted.append({"date": parsed_date.isoformat() if parsed_date else "", "description": description, "amount": amount or "", "transaction_type": typ, "category": "", "payment_method": "UPI", "notes": None, "source": "document", "review_reason": review_reason})
        if accepted:
            return {"preview": True, "accepted_count": len(accepted), "rejected_count": len(rejected), "accepted": accepted, "rejected": rejected, "message": "Review each payment and fill any unreadable fields before importing."}
    anchors = [i for i, line in enumerate(lines) if card_re.search(line)]
    for position, start in enumerate(anchors):
        end = anchors[position + 1] if position + 1 < len(anchors) else min(len(lines), start + 12)
        block = lines[start:end]
        handled_lines.update(range(start, end))
        title = block[0]
        heading = card_re.search(title).group(0)
        heading_lower = heading.casefold()
        typ = "income" if ("to you" in heading_lower or "received" in heading_lower or "payment from" in heading_lower) else "expense"
        amount = None
        for item in block:
            amount_match = re.search(r"(?:₹|rs\.?|inr)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", item, re.I)
            if amount_match:
                amount = float(amount_match.group(1).replace(",", ""))
                break
        date_match = next((date_re.search(item) for item in block if date_re.search(item)), None)
        parsed_date = parse_document_date(date_match.group(1)) if date_match else None
        if amount is None or amount <= 0 or parsed_date is None:
            rejected.append({"row": start + 1, "reason": "Could not confidently read both amount and date from this payment card. Check the screenshot and enter it manually if needed.", "data": {"text": " ".join(block)[:250]}})
            continue
        if "to you" in heading_lower or "received" in heading_lower or "payment from" in heading_lower:
            description = "Payment received"
        else:
            description = heading[:250]
        occurrence_key = (parsed_date.isoformat(), description.casefold(), round(amount, 2), typ)
        card_occurrences[occurrence_key] = card_occurrences.get(occurrence_key, 0) + 1
        if card_occurrences[occurrence_key] > 1:
            description = f"{description[:230]} ({card_occurrences[occurrence_key]})"
        key = (parsed_date.isoformat(), description.casefold(), round(amount, 2), typ)
        if key in existing or key in seen:
            rejected.append({"row": start + 1, "reason": "Possible duplicate transaction", "data": {"text": " ".join(block)[:250]}})
            continue
        seen.add(key)
        accepted.append({"date": parsed_date.isoformat(), "description": description, "amount": amount, "transaction_type": typ, "category": "", "payment_method": "UPI", "notes": None, "source": "document"})

    credit_words = re.compile(r"\b(cr|credit|credited|deposit|deposited|salary|scholarship|allowance|refund|received)\b", re.I)
    debit_words = re.compile(r"\b(dr|debit|debited|withdrawal|withdrawn|purchase|paid|payment|spent)\b", re.I)
    money_re = re.compile(r"(?<![A-Za-z0-9/#:])[-+]?\s*(?:₹|INR\s*)?\d[\d,]*(?:\.\d{1,2})?(?![A-Za-z0-9/#:])", re.I)
    for line_no, line in enumerate(lines, 1):
        if line_no - 1 in handled_lines or not line:
            continue
        dm = date_re.search(line)
        if not dm:
            continue
        parsed_date = parse_document_date(dm.group(1))
        if not parsed_date:
            rejected.append({"row": line_no, "reason": "Could not read this transaction date", "data": {"text": line[:250]}})
            continue
        remainder = line[:dm.start()] + " " + line[dm.end():]
        values = money_re.findall(remainder)
        parsed_values = []
        for value in values:
            cleaned = re.sub(r"[^0-9.+-]", "", value.replace(",", ""))
            try:
                number = abs(float(cleaned))
                if number > 0:
                    parsed_values.append(number)
            except ValueError:
                pass
        if not parsed_values:
            rejected.append({"row": line_no, "reason": "No transaction amount could be read", "data": {"text": line[:250]}})
            continue
        amount = parsed_values[0]
        typ = "income" if credit_words.search(remainder) else "expense"
        if debit_words.search(remainder) or re.search(r"-\s*(?:₹|INR\s*)?\d", remainder):
            typ = "expense"
        cleaned_description = remainder
        for value in values:
            cleaned_description = cleaned_description.replace(value, " ", 1)
        cleaned_description = re.sub(r"\b(cr|dr|credit|credited|debit|debited|withdrawal|withdrawn|deposit|deposited)\b", " ", cleaned_description, flags=re.I)
        desc = re.sub(r"\s+", " ", cleaned_description).strip(" -|,;:")[:250]
        if not desc or len(parsed_values) > 4:
            rejected.append({"row": line_no, "reason": "Transaction details are ambiguous; review this line manually", "data": {"text": line[:250]}})
            continue
        predicted = predict_income_category(desc) if typ == "income" else predict_user_category(db, user.id, desc)
        category = predicted["category"] if isinstance(predicted, dict) else predicted
        key = (parsed_date.isoformat(), desc.casefold(), round(amount, 2), typ)
        if key in existing or key in seen:
            rejected.append({"row": line_no, "reason": "Duplicate transaction", "data": {"text": line[:250]}})
            continue
        seen.add(key)
        accepted.append({"date": parsed_date.isoformat(), "description": desc, "amount": amount, "transaction_type": typ, "category": category, "payment_method": None, "notes": None, "source": "document"})
    if not accepted and not rejected:
        raise HTTPException(422, "No transaction rows with readable dates were found in this document.")
    return {"preview": True, "accepted_count": len(accepted), "rejected_count": len(rejected), "accepted": accepted, "rejected": rejected, "message": "Confirm each date, amount, transaction type, description, and category before importing."}

@app.post("/api/transactions/import-reviewed")
def import_reviewed_transactions(data: DocumentImportIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    existing = {(t.date.isoformat(), t.description.casefold(), round(t.amount, 2), t.transaction_type) for t in tx_query(db, user.id).all()}
    accepted, rejected = [], []
    for row_no, item in enumerate(data.transactions, 1):
        category = item.category.strip()
        description = item.description.strip()
        key = (item.date.isoformat(), description.casefold(), round(item.amount, 2), item.transaction_type)
        if not description or category in ("", "Needs Review") or not valid_transaction_category(db, user.id, item.transaction_type, category):
            rejected.append({"row": row_no, "reason": "Choose a valid category and description for this transaction."})
        elif key in existing:
            rejected.append({"row": row_no, "reason": "Duplicate transaction."})
        else:
            existing.add(key)
            accepted.append(Transaction(user_id=user.id, date=item.date, description=description, amount=item.amount, transaction_type=item.transaction_type, category=category, payment_method=item.payment_method, notes=item.notes, source="document"))
    db.add_all(accepted)
    db.commit()
    return {"accepted_count": len(accepted), "rejected_count": len(rejected), "rejected": rejected, "message": f"Imported {len(accepted)} transactions; {len(rejected)} rows need attention." if rejected else f"Imported {len(accepted)} transactions."}

@app.post("/api/expense-diary")
def add_expense_diary_entries(data: ExpenseDiaryIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    existing = {(t.date.isoformat(), t.description.casefold(), round(t.amount, 2), t.transaction_type) for t in tx_query(db, user.id).all()}
    accepted, rejected = [], []
    for row_no, item in enumerate(data.entries, 1):
        description = item.description.strip()
        key = (data.date.isoformat(), description.casefold(), round(item.amount, 2), "expense")
        if key in existing:
            rejected.append({"row": row_no, "description": description, "reason": "This diary entry already exists for that day."})
            continue
        predicted = predict_user_category(db, user.id, description)
        category = predicted.get("category", "Other") if isinstance(predicted, dict) else predicted
        if category not in EXPENSE_CATEGORIES:
            category = "Other"
        accepted.append(Transaction(user_id=user.id, date=data.date, description=description, amount=item.amount, transaction_type="expense", category=category, source="expense_diary", notes="Added from expense diary"))
        existing.add(key)
    db.add_all(accepted)
    db.commit()
    return {"accepted_count": len(accepted), "rejected_count": len(rejected), "rejected": rejected, "message": f"Added {len(accepted)} expenses; {len(rejected)} duplicates skipped." if rejected else f"Added {len(accepted)} expenses."}

@app.get("/api/expense-diary")
def get_expense_diary(month: str = Query(..., pattern=r"^\d{4}-\d{2}$"), db: Session = Depends(get_db), user: User = Depends(current_user)):
    try:
        year, month_number = map(int, month.split("-"))
        first = date(year, month_number, 1)
    except ValueError:
        raise HTTPException(422, "Month must be in YYYY-MM format")
    last = date(year, month_number, monthrange(year, month_number)[1])
    # The diary is the readable daily view of every saved expense, regardless of
    # whether it came from a manual entry, statement import, or diary quick-add.
    entries = tx_query(db, user.id).filter(Transaction.transaction_type == "expense", Transaction.date >= first, Transaction.date <= last).order_by(Transaction.date.desc(), Transaction.id.desc()).all()
    daily_totals = {}
    for item in entries:
        day = item.date.isoformat()
        daily_totals[day] = daily_totals.get(day, 0) + float(item.amount)
    weekly_totals = []
    week_start = first
    while week_start <= last:
        week_end = min(last, week_start + timedelta(days=6 - week_start.weekday()))
        week_amount = sum(value for day, value in daily_totals.items() if week_start.isoformat() <= day <= week_end.isoformat())
        weekly_totals.append({"start_date": week_start.isoformat(), "end_date": week_end.isoformat(), "amount": round(week_amount, 2)})
        week_start = week_end + timedelta(days=1)
    return {"month": month, "monthly_total": round(sum(float(item.amount) for item in entries), 2), "entry_count": len(entries), "daily_totals": daily_totals, "weekly_totals": weekly_totals, "entries": entries}

@app.get("/api/transactions/template")
def csv_template():
    return {
        "filename": "spendcount-template.csv",
        "content": "date,description,amount,type,category\n2026-11-01,Campus cafe lunch,185,expense,Food\n2026-11-02,Family allowance,8000,income,Family Allowance\n",
    }

@app.get("/api/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return dashboard_summary_data(db, user)

def dashboard_summary_data(db: Session, user: User, forecast_context=None):
    today = date.today()
    start, end = month_bounds(today)
    row = db.query(
        func.coalesce(func.sum(case((Transaction.transaction_type == "income", Transaction.amount), else_=0)), 0),
        func.coalesce(func.sum(case((Transaction.transaction_type == "expense", Transaction.amount), else_=0)), 0),
    ).filter(Transaction.user_id == user.id, Transaction.date >= start, Transaction.date <= today).one()
    income, spent = float(row[0]), float(row[1])
    fc, _, upcoming = forecast_context or forecast_for(user.id, db)
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
    category = data.category.strip()
    if not category:
        raise HTTPException(422, "Enter a category name")
    item = db.query(Budget).filter(Budget.user_id == user.id, Budget.category == category).first()
    if item:
        item.monthly_limit = data.monthly_limit
    else:
        item = Budget(user_id=user.id, category=category, monthly_limit=data.monthly_limit)
        db.add(item)
    db.commit()
    db.refresh(item)
    return item

@app.put("/api/budgets/{budget_id}")
def update_budget(budget_id: int, data: BudgetIn, db: Session = Depends(get_db), user: User = Depends(current_user)):
    item = db.query(Budget).filter(Budget.id == budget_id, Budget.user_id == user.id).first()
    if not item:
        raise HTTPException(404, "Budget not found")
    category = data.category.strip()
    if not category:
        raise HTTPException(422, "Enter a category name")
    item.category = category
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
    return build_personalized_alerts(db, user)

def build_personalized_alerts(db: Session, user: User, forecast_context=None):
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
    fc, _, upcoming = forecast_context or forecast_for(user.id, db)
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

@app.get("/api/dashboard/overview")
def dashboard_overview(db: Session = Depends(get_db), user: User = Depends(current_user)):
    forecast_context = forecast_for(user.id, db)
    return {
        "summary": dashboard_summary_data(db, user, forecast_context),
        "trends": spending_trends(db, user),
        "breakdown": spending_breakdown(db, user),
        "transactions": list_transactions(db=db, user=user, limit=6, offset=0),
        "alerts": build_personalized_alerts(db, user, forecast_context),
    }

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
def categories(db: Session = Depends(get_db), user: User = Depends(current_user)):
    return {"expense": expense_categories_for_user(db, user.id), "income": INCOME_CATEGORIES}
