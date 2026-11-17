from datetime import date, timedelta
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi import HTTPException
from app.database import Base
from app.models import User, Transaction, Budget, IncomeSchedule, CategoryCorrection
from app.schemas import RegisterIn, LoginIn, ProfileUpdateIn, PasswordUpdateIn, TransactionIn, BudgetIn, ScheduleIn
from app.main import (
    register,
    login,
    update_profile,
    update_password,
    create_transaction,
    list_transactions,
    update_transaction,
    delete_transaction,
    create_budget,
    get_budgets,
    delete_budget,
    add_schedule,
    get_schedules,
    delete_schedule,
    dashboard_summary,
    spending_breakdown,
    spending_trends,
    month_end,
    cash_flow,
    recalculate,
    categorise,
    category_corrections,
    delete_correction,
    anomalies,
    personalized_alerts,
    insights,
    get_weekly_plan,
    parse_csv_content,
    model_evaluation,
)
from app.security import hash_password, verify_password

@pytest.fixture
def db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()

@pytest.fixture
def test_user(db):
    user = User(name="Test Student", email="student@spendcount.app", hashed_password=hash_password("SecurePass123"))
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def test_user_registration_and_login(db):
    reg_data = RegisterIn(name="New Student", email="new@spendcount.app", password="Password1234")
    reg_res = register(reg_data, db)
    assert reg_res["user"]["email"] == "new@spendcount.app"
    assert "access_token" in reg_res

    # Duplicate registration should raise 409
    with pytest.raises(HTTPException) as exc:
        register(reg_data, db)
    assert exc.value.status_code == 409

    # Login with valid password
    login_res = login(LoginIn(email="new@spendcount.app", password="Password1234"), db)
    assert login_res["access_token"] is not None

    # Login with invalid password
    with pytest.raises(HTTPException) as exc:
        login(LoginIn(email="new@spendcount.app", password="WrongPassword"), db)
    assert exc.value.status_code == 401

def test_profile_and_password_update(db, test_user):
    # Update profile name
    updated = update_profile(ProfileUpdateIn(name="Updated Student Name"), db, test_user)
    assert updated["name"] == "Updated Student Name"
    assert test_user.name == "Updated Student Name"

    # Update password with wrong old password
    with pytest.raises(HTTPException) as exc:
        update_password(PasswordUpdateIn(old_password="WrongOldPassword", new_password="NewSecurePass456"), db, test_user)
    assert exc.value.status_code == 400

    # Update password with correct old password
    res = update_password(PasswordUpdateIn(old_password="SecurePass123", new_password="NewSecurePass456"), db, test_user)
    assert res["message"] == "Password updated successfully"
    assert verify_password("NewSecurePass456", test_user.hashed_password)

def test_transaction_crud_and_validation(db, test_user):
    # Create expense transaction
    tx_in = TransactionIn(date=date.today(), description="Textbooks for semester", amount=1200.0, transaction_type="expense", category="Academics")
    tx = create_transaction(tx_in, db, test_user)
    assert tx.id is not None
    assert tx.amount == 1200.0

    # Auto-categorise when category is Auto or empty
    tx_auto = create_transaction(TransactionIn(date=date.today(), description="Coffee and muffin at campus cafe", amount=140.0, transaction_type="expense", category="Auto"), db, test_user)
    assert tx_auto.category == "Food"

    # List transactions
    items = list_transactions(None, None, None, None, None, 100, 0, db, test_user)
    assert len(items) == 2

    # Update transaction
    tx_update = TransactionIn(date=date.today(), description="Used textbooks", amount=900.0, transaction_type="expense", category="Academics")
    updated_tx = update_transaction(tx.id, tx_update, db, test_user)
    assert updated_tx.amount == 900.0
    assert updated_tx.description == "Used textbooks"

    # Delete transaction
    delete_transaction(tx.id, db, test_user)
    remaining = list_transactions(None, None, None, None, None, 100, 0, db, test_user)
    assert len(remaining) == 1

def test_budget_and_schedule_flow(db, test_user):
    # Create budget
    b = create_budget(BudgetIn(category="Food", monthly_limit=6000.0), db, test_user)
    assert b.category == "Food"
    assert b.monthly_limit == 6000.0

    # Update budget limit for same category
    b2 = create_budget(BudgetIn(category="Food", monthly_limit=7500.0), db, test_user)
    assert b2.monthly_limit == 7500.0
    assert len(get_budgets(db, test_user)) == 1

    # Create schedule
    s = add_schedule(ScheduleIn(income_source="Family allowance", expected_amount=8000.0, expected_date=date.today() + timedelta(days=5), recurring_frequency="monthly"), db, test_user)
    assert s.id is not None
    assert len(get_schedules(db, test_user)) == 1

    # Delete schedule
    delete_schedule(s.id, db, test_user)
    assert len(get_schedules(db, test_user)) == 0

    # Delete budget
    delete_budget(b.id, db, test_user)
    assert len(get_budgets(db, test_user)) == 0

def test_ai_insights_alerts_and_forecasting(db, test_user):
    # Add income and expense history
    today = date.today()
    db.add(Transaction(user_id=test_user.id, date=today - timedelta(days=10), description="Parent transfer", amount=15000.0, transaction_type="income", category="Family Allowance"))
    db.add(Transaction(user_id=test_user.id, date=today - timedelta(days=5), description="Hostel room fee", amount=6000.0, transaction_type="expense", category="Rent and Hostel"))
    db.add(Transaction(user_id=test_user.id, date=today - timedelta(days=2), description="Grocery store order", amount=1200.0, transaction_type="expense", category="Food"))
    db.add(Transaction(user_id=test_user.id, date=today, description="Campus cafe lunch", amount=180.0, transaction_type="expense", category="Food"))
    db.commit()

    # Dashboard summary
    summary = dashboard_summary(db, test_user)
    assert summary["balance"] == 7620.0
    assert summary["month_income"] == 15000.0
    assert summary["month_expenses"] == 7380.0

    # Breakdown and trends
    breakdown = spending_breakdown(db, test_user)
    assert len(breakdown) >= 2
    trends = spending_trends(db, test_user)
    assert len(trends) > 0

    # Cash-flow forecast
    fc = month_end(db, test_user)
    assert "predicted_end_balance" in fc
    assert "daily_spend_rate" in fc

    cf = cash_flow(db, test_user)
    assert "forecast" in cf
    assert len(cf["points"]) > 0

    # AI Insights
    ins = insights(db, test_user)
    assert ins["empty"] is False
    assert len(ins["insights"]) > 0
    assert ins["highest_category"] in ("Rent and Hostel", "Food")

    # Alerts
    alerts_list = personalized_alerts(db, test_user)
    assert isinstance(alerts_list, list)

    # Weekly plan
    plan = get_weekly_plan(db, test_user)
    assert "weekly_budget" in plan
    assert "daily_limit" in plan

def test_category_corrections_flow(db, test_user):
    # Manual correction adds CategoryCorrection
    tx = create_transaction(TransactionIn(date=date.today(), description="Campus cafeteria lunch", amount=120.0, transaction_type="expense", category="Academics"), db, test_user)
    corrections = category_corrections(db, test_user)
    assert len(corrections) >= 1
    corr_id = corrections[0]["id"]

    # Delete correction
    delete_correction(corr_id, db, test_user)
    assert len(category_corrections(db, test_user)) == 0

def test_model_evaluation_endpoint(db, test_user):
    res = model_evaluation(db, test_user)
    assert "split" in res or "status" in res
