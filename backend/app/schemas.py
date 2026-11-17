from datetime import date
from pydantic import BaseModel, ConfigDict, EmailStr, Field

EXPENSE_CATEGORIES = ["Food", "Transport", "Rent and Hostel", "Shopping", "Subscriptions", "Academics", "Entertainment", "Healthcare", "Personal Care", "Other"]
INCOME_CATEGORIES = ["Family Allowance", "Scholarship", "Part-time Job", "Internship", "Freelancing", "Other Income"]

class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
class LoginIn(BaseModel):
    email: EmailStr
    password: str
class TransactionIn(BaseModel):
    date: date
    description: str = Field(min_length=1, max_length=250)
    amount: float = Field(gt=0, le=100000000)
    transaction_type: str
    category: str = Field(min_length=1, max_length=40)
    payment_method: str | None = None
    notes: str | None = None
    source: str = "manual"
class TransactionOut(TransactionIn):
    id: int
    model_config = ConfigDict(from_attributes=True)
class BudgetIn(BaseModel):
    category: str
    monthly_limit: float = Field(gt=0, le=100000000)
class ScheduleIn(BaseModel):
    income_source: str
    expected_amount: float = Field(gt=0)
    expected_date: date
    recurring_frequency: str = "one-time"

