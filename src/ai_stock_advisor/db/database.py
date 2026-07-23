"""
Database Configuration & ORM Schemas.
Establishes SQLAlchemy sessions and maps SQL schemas.
Defaults to SQLite storage located at data/advisor.db.
"""
from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Generator
from sqlalchemy import Column, DateTime, Integer, String, Float, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from config.settings import settings

logger = logging.getLogger("ai_stock_advisor.db")

# Setup SQLite DB Path in the workspace data folder
db_dir = Path(settings.BASE_DIR) / "data"
db_dir.mkdir(parents=True, exist_ok=True)
db_url = f"sqlite:///{db_dir}/advisor.db"

logger.info("Initializing SQLAlchemy database engine at %s", db_url)

# check_same_thread=False is safe for SQLite in multi-thread FastAPI contexts
engine = create_engine(db_url, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class User(Base):
    """
    User Table Schema.
    Stores login username and hashed password credentials.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DailyScan(Base):
    """
    DailyScan Table Schema.
    Persists morning F&O opportunity scan details.
    """
    __tablename__ = "daily_scans"

    id = Column(Integer, primary_key=True, index=True)
    scan_date = Column(DateTime, default=datetime.utcnow, index=True)
    ticker = Column(String, index=True, nullable=False)
    indicators = Column(String, nullable=False)  # JSON-serialized indicator dictionary
    news = Column(String, nullable=False)        # JSON-serialized news sentiment dictionary
    options = Column(String, nullable=False)     # JSON-serialized option chain metrics dictionary
    score = Column(Float, nullable=False, index=True)
    recommendation = Column(String, nullable=False)  # JSON-serialized detailed suggestions (SL, Targets, position size, etc.)


def get_db() -> Generator[Any, None, None]:
    """FastAPI Session injection dependency."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Executes DDL schema creation statements."""
    logger.info("Creating SQLAlchemy database schemas...")
    Base.metadata.create_all(bind=engine)
