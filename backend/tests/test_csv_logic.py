from pathlib import Path
from datetime import date
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.database import Base
from app.models import User, Transaction
from app.main import parse_csv_content

def test_template_columns_are_documented():
    readme=(Path(__file__).parents[2]/"README.md").read_text(encoding="utf-8")
    assert "date,description,amount,type,category" in readme

def test_csv_preview_reports_invalid_rows_and_duplicates():
    engine=create_engine("sqlite://",connect_args={"check_same_thread":False},poolclass=StaticPool)
    Base.metadata.create_all(bind=engine);db=sessionmaker(bind=engine)()
    user=User(name="CSV Student",email="csv@example.com",hashed_password="hash");db.add(user);db.commit();db.refresh(user)
    db.add(Transaction(user_id=user.id,date=date(2026,11,1),description="Already logged",amount=80,transaction_type="expense",category="Food"));db.commit()
    content=("date,description,amount,type,category\n"
      "2026-11-02,Campus cafe lunch,120,expense,Food\n"
      "2026-11-03,Invalid date,50,expense,Food\n"
      "2026-11-04,Negative amount,-2,expense,Food\n"
      "2026-11-01,Already logged,80,expense,Food\n"
      "2026-11-02,Campus cafe lunch,120,expense,Food\n")
    result=parse_csv_content(content,True,db,user)
    assert result["accepted_count"]==1, result
    assert result["rejected_count"]==4
    assert any("invalid isoformat" in row["reason"].lower() for row in result["rejected"])
    assert any("duplicate" in row["reason"].lower() for row in result["rejected"])
    assert db.query(Transaction).filter_by(user_id=user.id).count()==1
    db.close();engine.dispose()
