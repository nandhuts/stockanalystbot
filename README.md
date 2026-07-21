# AI Stock Advisor 📈🤖

A production-ready, modular AI-powered Stock Advisor and Derivatives Analyzer built on Python 3.12. This project forms the architectural foundation for a multi-agent financial analyst, offering index scanners, option chain analysis, news sentiment classification, machine learning probability models, backtesting engines, and a conversational chat advisor.

---

## Directory Architecture & Component Explanations

The project follows a standard `src`-layout structure (PEP 517/518 compliance) which decouples source code from configuration scripts, preventing import leakage and test double conflicts.

Below is an overview of the key directories:

### 📂 Root Directory Configurations
- **`.github/workflows/`**: Continuous Integration workflows. Runs Ruff lints, Mypy type-checks, and automated Pytest unit testing on commits and PRs.
- **`config/`**: Central configs. Contains [settings.py](file:///C:/Users/anand/.gemini/antigravity/scratch/ai_stock_advisor/config/settings.py) for variables validation and [logging_config.py](file:///C:/Users/anand/.gemini/antigravity/scratch/ai_stock_advisor/config/logging_config.py) for log management.
- **`data/`**: Stores data caches, model files, SQLite databases, and scan reports.
- **`tests/`**: Contains the full unit testing suite.

### 📂 Source Code (`src/ai_stock_advisor/`)
The primary source code is modularized into dedicated domain layers:
- **`core/`**: Custom mathematical engines:
  - `indicators.py`: Calculations for EMAs, RSI, MACD, Bollinger Bands, ATR, VWAP, and ADX.
  - `scanner.py`: Trend ratings scan orchestrator.
  - `ranker.py`: Quantitative multi-factor probability ranker.
  - `options.py`: Option chain Put-Call Ratio (PCR) and Max Pain calculators.
  - `backtester.py`: 5-year strategy simulation backtest engine.
- **`services/`**: Integration layers with third-party APIs:
  - `llm/`: Handlers for OpenAI completions, structured technical report analysis, news sentiment summaries, options trading recommendations, and chat advisor.
  - `market_data/`: Client classes for querying stock prices via `yfinance`.
- **`ml/`**: Machine Learning pipeline for training Random Forest, XGBoost, and LightGBM models.
- **`db/`**: SQLite database layer managing SQLAlchemy models.
- **`api/`**: FastAPI REST API endpoints protected via JWT Bearer tokens.
- **`telegram/`**: Active Telegram Bot client exposing advisor commands.
- **`dashboard/`**: Streamlit dashboard UI app.

---

## REST API & JWT Authentication Workflow

FastAPI exposes a secure, token-protected REST API on port `8000`:

1. **Register User**:
   - `POST /api/register`
   - Payload: `{"username": "your_user", "password": "your_secure_password"}`
2. **Retrieve Token**:
   - `POST /api/token`
   - Form Data: `username`, `password`
   - Response returns a Bearer `access_token`.
3. **Query Protected Data**:
   Include the token in HTTP headers: `Authorization: Bearer <your_access_token>`.
   - `GET /api/scanner`: Retrieve Nifty 50 scan scores.
   - `GET /api/rankings`: Retrieve top 20 bullish stock picks.
   - `GET /api/options/{ticker}`: Get PCR, Max Pain, and volatility SL/Targets.

---

## Containerization & Local Setup

### 🐳 Docker & Docker Compose (Recommended)
Run the entire production stack (FastAPI API and Streamlit Dashboard) inside Docker:

```bash
# Build images and launch multi-containers
docker-compose up --build
```
- **API Swagger Documentation**: Open `http://localhost:8000/docs`
- **Streamlit Advisor Dashboard**: Open `http://localhost:8501`

### 💻 Developer Local Environment Setup

1. **Repository Setup**:
   ```bash
   cp .env.example .env
   ```
   Add your `OPENAI_API_KEY` to the `.env` file to sign tokens and run AI models.

2. **Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1   # Windows PowerShell
   source .venv/Scripts/activate # Git Bash / Linux
   ```

3. **Install Packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Services Manually**:
   - **FastAPI API**:
     ```bash
     uvicorn ai_stock_advisor.api.main:app --reload --port 8000
     ```
   - **Streamlit Dashboard**:
     ```bash
     streamlit run src/ai_stock_advisor/dashboard/app.py
     ```
   - **Telegram Bot**:
     ```bash
     python src/ai_stock_advisor/telegram/bot.py
     ```

5. **Execute Test Suite**:
   ```bash
   # Run automated test scripts
   pytest tests/ --cov=src --cov-report=term
   ```
