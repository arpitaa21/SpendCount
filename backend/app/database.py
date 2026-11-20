import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool, NullPool

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
DATABASE_URL = (
    os.getenv("NEON_DATABASE_URL")
    or os.getenv("NEON_POSTGRES_URL")
    or os.getenv("DATABASE_URL")
    or "sqlite:///./spendcount.db"
)

# Normalize PostgreSQL schemes for psycopg
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# In serverless environments (e.g. Vercel), only /tmp is writable
is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
if is_serverless and DATABASE_URL.startswith("sqlite:///./"):
    DATABASE_URL = "sqlite:////tmp/spendcount.db"

connect_args = {}
engine_kwargs = {"pool_pre_ping": True}

if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False
    engine_kwargs["connect_args"] = connect_args
else:
    # Use NullPool in serverless environments to avoid idle connection accumulation
    if is_serverless:
        engine_kwargs["poolclass"] = NullPool
    engine_kwargs["connect_args"] = connect_args

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
