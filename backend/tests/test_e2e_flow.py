from datetime import date, timedelta
from io import BytesIO
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.database import Base
from app.models import User, Transaction, Budget, IncomeSchedule
from app.schemas import RegisterIn, LoginIn, ProfileUpdateIn, PasswordUpdateIn, TransactionIn, BudgetIn, ScheduleIn
from app.main import (
    register,
    login,
    update_profile,
    update_password,
    create_transaction,
    list_transactions,
    create_budget,
    dashboard_summary,
    spending_breakdown,
    month_end,
    cash_flow,
    recalculate,
    anomalies,
    personalized_alerts,
    insights,
    get_weekly_plan,
    parse_csv_content,
    categories,
    health,
)
from app.security import verify_password

def test_complete_end_to_end_student_lifecycle():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    db = sessionmaker(bind=engine)()

    # Step 1: Health check
    h = health()
    assert h["status"] == "ok"

    # Step 2: Register user
    reg = register(RegisterIn(name="EndToEnd Student", email="e2e@spendcount.app", password="OriginalPassword123"), db)
    user_id = reg["user"]["id"]
    token = reg["access_token"]
    assert token is not None

    user = db.get(User, user_id)
    assert user is not None

    # Step 3: Login
    log = login(LoginIn(email="e2e@spendcount.app", password="OriginalPassword123"), db)
    assert log["access_token"] is not None

    # Step 4: Add income transactions
    today = date.today()
    tx_inc1 = create_transaction(TransactionIn(date=today - timedelta(days=20), description="Family allowance from parents", amount=12000.0, transaction_type="income", category="Auto"), db, user)
    assert tx_inc1.category == "Family Allowance"

    tx_inc2 = create_transaction(TransactionIn(date=today - timedelta(days=10), description="Campus research internship stipend", amount=5000.0, transaction_type="income", category="Auto"), db, user)
    assert tx_inc2.category == "Internship"

    # Step 5: Add expense transactions across categories
    create_transaction(TransactionIn(date=today - timedelta(days=15), description="Hostel room fee monthly", amount=5500.0, transaction_type="expense", category="Auto"), db, user)
    create_transaction(TransactionIn(date=today - timedelta(days=8), description="Textbook and stationery store", amount=850.0, transaction_type="expense", category="Auto"), db, user)
    create_transaction(TransactionIn(date=today - timedelta(days=5), description="Campus cafe lunch with friends", amount=240.0, transaction_type="expense", category="Auto"), db, user)
    create_transaction(TransactionIn(date=today - timedelta(days=3), description="Metro card top up recharge", amount=300.0, transaction_type="expense", category="Auto"), db, user)
    create_transaction(TransactionIn(date=today - timedelta(days=1), description="Coffee cafe breakfast", amount=120.0, transaction_type="expense", category="Auto"), db, user)

    # Step 6: CSV batch upload
    csv_text = (
        "date,description,amount,type,category\n"
        f"{today - timedelta(days=4)},Groceries market supermarket,650,expense,Food\n"
        f"{today - timedelta(days=2)},Pharmacy medicine prescription,220,expense,Healthcare\n"
        f"{today},Dining hall thali meal,140,expense,Food\n"
    )
    csv_res = parse_csv_content(csv_text, preview=False, db=db, user=user)
    assert csv_res["accepted_count"] == 3
    assert csv_res["rejected_count"] == 0

    # Step 7: Check transactions list
    tx_list = list_transactions(None, None, None, None, None, 100, 0, db, user)
    assert len(tx_list) == 10

    # Step 8: Set budgets and expected income
    b1 = create_budget(BudgetIn(category="Food", monthly_limit=5000.0), db, user)
    b2 = create_budget(BudgetIn(category="Rent and Hostel", monthly_limit=6000.0), db, user)
    assert b1.monthly_limit == 5000.0

    sch = IncomeSchedule(user_id=user.id, income_source="Family transfer", expected_amount=10000.0, expected_date=today + timedelta(days=7), recurring_frequency="monthly")
    db.add(sch)
    db.commit()

    # Step 9: Verify dashboard metrics and breakdown
    summary = dashboard_summary(db, user)
    assert summary["balance"] > 0
    assert summary["month_income"] >= 0
    assert summary["month_expenses"] >= 0
    assert summary["has_budget"] is True
    assert summary["health"] in ("On track", "Needs attention")

    breakdown = spending_breakdown(db, user)
    assert len(breakdown) > 0

    # Step 10: Cash-flow forecast & recalculation
    fc = month_end(db, user)
    assert "predicted_end_balance" in fc
    assert "daily_safe_limit" in fc

    recalc_res = recalculate(db, user)
    assert recalc_res["predicted_end_balance"] == fc["predicted_end_balance"]

    cf_curve = cash_flow(db, user)
    assert len(cf_curve["points"]) > 0

    # Step 11: Anomaly detection
    # Add a massive unusual transaction to verify Isolation Forest outlier detection
    outlier = Transaction(user_id=user.id, date=today, description="Urgent emergency purchase", amount=9500.0, transaction_type="expense", category="Shopping")
    db.add(outlier)
    db.commit()

    detected_anomalies = anomalies(db, user)
    assert isinstance(detected_anomalies, list)

    # Step 12: AI Insights & Weekly plan
    ins = insights(db, user)
    assert ins["empty"] is False
    assert len(ins["insights"]) > 0

    plan = get_weekly_plan(db, user)
    assert plan["weekly_budget"] >= 0
    assert len(plan["suggestions"]) > 0

    # Step 13: Profile update and password change
    update_profile(ProfileUpdateIn(name="EndToEnd Graduated Student"), db, user)
    assert user.name == "EndToEnd Graduated Student"

    update_password(PasswordUpdateIn(old_password="OriginalPassword123", new_password="NewSecureStudentPassword999"), db, user)
    assert verify_password("NewSecureStudentPassword999", user.hashed_password)

    # Step 14: Verify re-login with new password
    new_login = login(LoginIn(email="e2e@spendcount.app", password="NewSecureStudentPassword999"), db)
    assert new_login["access_token"] is not None

    db.close()
    engine.dispose()
