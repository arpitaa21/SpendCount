import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.pool import StaticPool, NullPool

backend_dir = Path(__file__).resolve().parents[1]
load_dotenv(backend_dir / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")

# Normalize PostgreSQL schemes for psycopg
if DATABASE_URL:
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
    elif DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# In serverless environments (e.g. Vercel), only /tmp is writable
is_serverless = bool(os.getenv("VERCEL") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

if is_serverless:
    if not DATABASE_URL or DATABASE_URL.startswith("sqlite:///"):
        DATABASE_URL = "sqlite:////tmp/spendcount.db"
else:
    if not DATABASE_URL:
        DATABASE_URL = f"sqlite:///{(backend_dir / 'spendcount.db').as_posix()}"
    elif DATABASE_URL.startswith("sqlite:///./"):
        rel_path = DATABASE_URL[len("sqlite:///./"):]
        DATABASE_URL = f"sqlite:///{(backend_dir / rel_path).as_posix()}"

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
