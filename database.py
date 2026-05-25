from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

if not DATABASE_URL:
    DATABASE_URL = "sqlite:///./jobprepmate.db"

try:
    if DATABASE_URL.startswith("sqlite"):
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
    else:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
        )
except ImportError as e:
    missing = str(e)
    if "psycopg2" in missing or "asyncpg" in missing:
        import logging
        logging.getLogger("database").warning(
            "Database driver missing (%s); falling back to SQLite at jobprepmate.db",
            missing,
        )
        DATABASE_URL = "sqlite:///./jobprepmate.db"
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
    else:
        raise

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

Base = declarative_base()