"""Create a demo account and 4 months of clearly synthetic student transactions."""
import csv, random, sys
from pathlib import Path
from datetime import date, timedelta
sys.path.insert(0, str(Path(__file__).parent))
from app.database import Base, engine, SessionLocal
from app.models import User, Transaction, Budget, IncomeSchedule
from app.security import hash_password

random.seed(42)
Base.metadata.create_all(bind=engine)
db=SessionLocal()
user=db.query(User).filter_by(email="demo@spendcount.app").first()
if not user:
    user=User(name="Aarav Mehta",email="demo@spendcount.app",hashed_password=hash_password("SpendCount123"));db.add(user);db.commit();db.refresh(user)
if db.query(Transaction).filter_by(user_id=user.id).count()==0:
    today=date.today(); start=today-timedelta(days=119); rows=[]
    for i in range(120):
        day=start+timedelta(days=i)
        # irregular family transfers plus occasional scholarship and internship pay
        if i in (4,36,70,105): rows.append((day,"Family allowance",random.randint(7000,10500),"income","Family Allowance"))
        if i in (18,79): rows.append((day,"Scholarship disbursement",12000,"income","Scholarship"))
        if i in (27,58,91): rows.append((day,"Campus internship stipend",8500,"income","Internship"))
        for _ in range(random.choices([0,1,2,3],[.1,.37,.4,.13])[0]):
            cat=random.choices(["Food","Transport","Rent and Hostel","Shopping","Subscriptions","Academics","Entertainment","Healthcare","Personal Care","Other"],[.34,.15,.08,.10,.05,.08,.09,.03,.04,.04])[0]
            descriptions={"Food":["Campus cafe lunch","Grocery market","Coffee and snacks","Mess dinner"],"Transport":["Metro recharge","Auto ride","Bus ticket"],"Rent and Hostel":["Hostel fee","Electricity bill"],"Shopping":["Online shopping","Campus essentials"],"Subscriptions":["Spotify premium","Cloud storage"],"Academics":["Textbook and stationery","Printing notes"],"Entertainment":["Movie ticket","Weekend outing"],"Healthcare":["Pharmacy medicine"],"Personal Care":["Laundry","Haircut salon"],"Other":["Bank charge","Miscellaneous payment"]}
            amount=random.randint(80,650) if cat not in ("Rent and Hostel","Shopping","Academics") else random.randint(300,1800)
            rows.append((day,random.choice(descriptions[cat]),amount,"expense",cat))
    # Intentionally unusual synthetic outliers.
    rows += [(today-timedelta(days=12),"Unusually large cafe group order",3200,"expense","Food"),(today-timedelta(days=33),"Urgent textbook bundle",4900,"expense","Academics"),(today-timedelta(days=51),"Late night cab ride",1850,"expense","Transport")]
    db.add_all([Transaction(user_id=user.id,date=d,description=desc,amount=amt,transaction_type=typ,category=cat,source="synthetic") for d,desc,amt,typ,cat in rows]);db.commit()
# Keep the demo comfortably above the requested model-development sample size,
# including after a previous seed run with a different random generator version.
count=db.query(Transaction).filter_by(user_id=user.id).count()
if count<210:
    additions=[]
    for offset in range(45):
        if count+len(additions)>=210: break
        day=date.today()-timedelta(days=offset*2+1)
        additions.append(Transaction(user_id=user.id,date=day,description="Synthetic campus meal",amount=random.randint(120,340),transaction_type="expense",category="Food",source="synthetic"))
    db.add_all(additions);db.commit()
if db.query(Budget).filter_by(user_id=user.id).count()==0:
    db.add_all([Budget(user_id=user.id,category=c,monthly_limit=v) for c,v in [("Food",7000),("Transport",2600),("Shopping",3000),("Academics",2500),("Entertainment",2200),("Rent and Hostel",8000),("Subscriptions",900)]]);db.commit()
if db.query(IncomeSchedule).filter_by(user_id=user.id).count()==0:
    db.add(IncomeSchedule(user_id=user.id,income_source="Family allowance",expected_amount=8500,expected_date=date.today()+timedelta(days=8),recurring_frequency="monthly"));db.commit()
out=Path(__file__).parent.parent/"dataset"/"synthetic_transactions.csv";out.parent.mkdir(exist_ok=True)
with out.open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f);w.writerow(["date","description","amount","type","category","source"])
    for t in db.query(Transaction).filter_by(user_id=user.id).order_by(Transaction.date):w.writerow([t.date,t.description,t.amount,t.transaction_type,t.category,"synthetic"])
print(f"Seeded {db.query(Transaction).filter_by(user_id=user.id).count()} synthetic transactions for demo@spendcount.app")
print("Demo password: SpendCount123")
db.close()
