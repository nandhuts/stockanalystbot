"""
FastAPI REST API router entrypoint.
Exposes endpoints for user registration, token generation, index scanners,
ranked bullish pick lists, and option chain analytics.
"""
import sys
from pathlib import Path

# Dynamically inject root and src paths to ensure standard resolution
root_dir = Path(__file__).resolve().parents[3]
src_dir = root_dir / "src"
for p in [str(root_dir), str(src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from datetime import datetime
import logging
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from config.settings import settings
from ai_stock_advisor.db.database import get_db, init_db, User
from ai_stock_advisor.api.auth import (
    create_access_token,
    get_current_user,
    get_password_hash,
    verify_password,
)
from ai_stock_advisor.core.indicators import TechnicalIndicatorEngine
from ai_stock_advisor.core.scanner import StockScanner
from ai_stock_advisor.core.ranker import StockRanker
from ai_stock_advisor.core.options import OptionAnalyzer
from ai_stock_advisor.services.market_data.client import MarketDataClient

logger = logging.getLogger("ai_stock_advisor.api")

app = FastAPI(
    title="AI Stock Advisor REST API",
    description="Production-ready derivatives trading scanner and market analyzer endpoints.",
    version="1.0.0",
)

# CORS configurations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Pydantic Schemas
# ==============================================================================

class UserCreate(BaseModel):
    """Payload to register a new user."""
    username: str = Field(..., description="Unique alphanumeric username")
    password: str = Field(..., min_length=6, description="Password (min 6 characters)")


class UserResponse(BaseModel):
    """Public user details profile schema."""
    id: int
    username: str
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    """Authentication token response payload schema."""
    access_token: str
    token_type: str


# ==============================================================================
# Startup Hook
# ==============================================================================

@app.on_event("startup")
def on_startup() -> None:
    """Trigger DB creation on start."""
    init_db()


# ==============================================================================
# Auth API Routes
# ==============================================================================

@app.post("/api/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)) -> User:
    """Registers a new user inside the SQLite/PostgreSQL database."""
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
        
    hashed_pwd = get_password_hash(user_data.password)
    new_user = User(username=user_data.username, hashed_password=hashed_pwd)
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    logger.info("Successfully registered user '%s'", new_user.username)
    return new_user


@app.post("/api/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> Dict[str, str]:
    """Validates login credentials to return a signed JWT access token."""
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    logger.info("Issued JWT access token for user '%s'", user.username)
    return {"access_token": access_token, "token_type": "bearer"}


# ==============================================================================
# Stock Advisor Data API Routes (Protected)
# ==============================================================================

@app.get("/api/scanner", response_model=List[Dict[str, Any]])
def get_latest_market_scan(
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Loads and returns Nifty 50 scanner results records from CSV logs."""
    results_file = Path(settings.BASE_DIR) / "data" / "scan_results.csv"
    if not results_file.exists():
        # Fallback to run fresh scan if file is missing
        try:
            client = MarketDataClient()
            engine = TechnicalIndicatorEngine()
            scanner = StockScanner(client, engine)
            df = scanner.scan(force_refresh=True)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Scanner execution failed: {exc}"
            )
    else:
        df = pd.read_csv(results_file)

    return df.to_dict(orient="records")


@app.get("/api/rankings", response_model=List[Dict[str, Any]])
def get_bullish_stock_rankings(
    current_user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """Loads and returns Top 20 stock rankings records sorted by probability."""
    results_file = Path(settings.BASE_DIR) / "data" / "rankings_results.csv"
    if not results_file.exists():
        try:
            client = MarketDataClient()
            engine = TechnicalIndicatorEngine()
            scanner = StockScanner(client, engine)
            ranker = StockRanker(scanner)
            df = ranker.rank_stocks(force_refresh=True)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Ranking execution failed: {exc}"
            )
    else:
        df = pd.read_csv(results_file)

    return df.to_dict(orient="records")


@app.get("/api/options/{ticker}", response_model=Dict[str, Any])
def get_option_chain_sentiment_report(
    ticker: str,
    current_user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Runs option chain PCR, Max Pain, and volatility-adjusted SL/Target recommendations."""
    try:
        client = MarketDataClient()
        engine = TechnicalIndicatorEngine()
        opt_analyzer = OptionAnalyzer(client, engine)
        report = opt_analyzer.analyze_options(ticker)
        return report
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Option analysis failed for ticker '{ticker}': {exc}"
        )
