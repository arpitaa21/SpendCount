from datetime import date
from pydantic import BaseModel, ConfigDict, EmailStr, Field
from typing import Literal

EXPENSE_CATEGORIES = ["Food", "Transport", "Rent and Hostel", "Shopping", "Subscriptions", "Academics", "Entertainment", "Healthcare", "Personal Care", "Other"]
INCOME_CATEGORIES = ["Family Allowance", "Scholarship", "Part-time Job", "Internship", "Freelancing", "Other Income"]

class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class ProfileUpdateIn(BaseModel):
    name: str = Field(min_length=2, max_length=100)
    avatar_data_url: str | None = Field(default=None, max_length=1500000)

class PasswordUpdateIn(BaseModel):
    old_password: str
    new_password: str = Field(min_length=8, max_length=128)

class TransactionIn(BaseModel):
    date: date
    description: str = Field(min_length=1, max_length=250)
    amount: float = Field(gt=0, le=100000000)
    transaction_type: Literal["income", "expense"]
    category: str = Field(min_length=1, max_length=40)
    payment_method: str | None = None
    notes: str | None = None
    source: str = "manual"

class TransactionOut(TransactionIn):
    id: int
    model_config = ConfigDict(from_attributes=True)

class BudgetIn(BaseModel):
    category: str = Field(min_length=1, max_length=40)
    monthly_limit: float = Field(gt=0, le=100000000)

class DocumentImportIn(BaseModel):
    transactions: list[TransactionIn] = Field(min_length=1, max_length=500)

class ExpenseDiaryEntryIn(BaseModel):
    description: str = Field(min_length=1, max_length=250)
    amount: float = Field(gt=0, le=100000000)

class ExpenseDiaryIn(BaseModel):
    date: date
    entries: list[ExpenseDiaryEntryIn] = Field(min_length=1, max_length=50)

class ScheduleIn(BaseModel):
    income_source: str
    expected_amount: float = Field(gt=0)
    expected_date: date
    recurring_frequency: Literal["one-time", "weekly", "monthly", "semester"] = "one-time"

class CommitmentIn(BaseModel):
    title: str = Field(min_length=2, max_length=120)
    category: str = Field(default="Other", max_length=40)
    amount: float = Field(gt=0, le=100000000)
    due_date: date
    recurring_frequency: Literal["one-time", "weekly", "monthly"] = "one-time"

class PlanningSettingsIn(BaseModel):
    emergency_buffer_target: float = Field(ge=0, le=100000000)
