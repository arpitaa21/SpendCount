from datetime import date
import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.database import Base
from app.models import User, Transaction, CategoryCorrection
from app.schemas import TransactionIn
from app.main import list_transactions, delete_transaction, create_transaction
from fastapi import HTTPException

engine=create_engine("sqlite://",connect_args={"check_same_thread":False},poolclass=StaticPool)
Base.metadata.create_all(bind=engine)
TestingSession=sessionmaker(bind=engine,autoflush=False,autocommit=False)

def test_user_transactions_are_isolated():
    db=TestingSession()
    try:
        user_a=User(name="Student A",email="a@example.com",hashed_password="hash")
        user_b=User(name="Student B",email="b@example.com",hashed_password="hash")
        db.add_all([user_a,user_b]);db.commit();db.refresh(user_a);db.refresh(user_b)
        tx=Transaction(user_id=user_a.id,date=date.today(),description="Campus lunch",amount=150,transaction_type="expense",category="Food")
        db.add(tx);db.commit();db.refresh(tx)
        rows_a=list_transactions(None,None,None,None,None,100,0,db,user_a)
        rows_b=list_transactions(None,None,None,None,None,100,0,db,user_b)
        assert [row.id for row in rows_a]==[tx.id]
        assert rows_b==[]
        with pytest.raises(HTTPException) as error: delete_transaction(tx.id,db,user_b)
        assert error.value.status_code==404
    finally: db.close()

def test_invalid_amount_is_rejected():
    with pytest.raises(ValidationError):
        TransactionIn(date=date.today(),description="Invalid",amount=-1,transaction_type="expense",category="Food")

def test_category_override_is_saved_to_the_user_feedback_queue():
    db=TestingSession()
    try:
        user=User(name="Review Student",email="review@example.com",hashed_password="hash");db.add(user);db.commit();db.refresh(user)
        payload=TransactionIn(date=date.today(),description="Campus cafe lunch",amount=150,transaction_type="expense",category="Transport")
        create_transaction(payload,db,user)
        corrections=db.query(CategoryCorrection).filter_by(user_id=user.id).all()
        assert len(corrections)==1
        assert corrections[0].corrected_category=="Transport"
        assert corrections[0].predicted_category!="Transport"
    finally: db.close()
