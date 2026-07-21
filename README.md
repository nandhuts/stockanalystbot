# AI Stock Advisor 📈🤖

A production-ready, modular AI-powered Stock Advisor layout built on Python 3.12. This project forms the architectural foundation for a multi-agent financial analyst and advisory application.

---

## Directory Architecture & Component Explanations

The project follows a standard `src`-layout structure (PEP 517/518 compliance) which decouples source code from configuration scripts, preventing import leakage and test double conflicts.

Below is an overview of the key directories:

### 📂 Root Directory Configurations
- **`.github/workflows/`**: Continuous Integration workflows. Configured for lint checks, formatting validations, and automated unit testing on code commits.
- **`config/`**: Central application configurations. Contains [settings.py](file:///C:/Users/anand/.gemini/antigravity/scratch/ai_stock_advisor/config/settings.py) for validating variables using `pydantic-settings` and [logging_config.py](file:///C:/Users/anand/.gemini/antigravity/scratch/ai_stock_advisor/config/logging_config.py) for managing rotational file and console logs.
- **`data/`**: Ignored placeholder directory for storing offline data caches, model snapshots, or stock databases.
- **`docs/`**: Holds developer documentation, architecture diagrams, and service definitions.
- **`logs/`**: Ignored folder containing the runtime execution output log file (`app.log`).
- **`tests/`**: Contains the full test suite matching the application design:
  - `conftest.py`: Shared testing hooks, fixtures, and environment mocks.
  - `unit/`: Direct testing of standalone functions, exceptions, and properties.
  - `integration/`: End-to-end service and model communication testing.

### 📂 Source Code (`src/ai_stock_advisor/`)
The primary source code is modularized into dedicated domain layers:
- **`core/`**: Core utilities, base classes, security policies, and application-wide domain error classes ([exceptions.py](file:///C:/Users/anand/.gemini/antigravity/scratch/ai_stock_advisor/src/ai_stock_advisor/core/exceptions.py)).
- **`agents/`**: Domain reasoning actors (e.g. market analyst, advisor, portfolio planner). Contains logic definitions and prompts.
- **`services/`**: Integration layers with third-party APIs:
  - `llm/`: Handlers for sending prompt context to providers (e.g. Gemini, OpenAI).
  - `market_data/`: Client classes for querying stock prices and statement details.
- **`models/`**: Domain entities and parsing schemas using Pydantic.
- **`utils/`**: Shared functions for date formatting, calculation helpers, and math.

---

## Developer Getting Started Guide

### Prerequisites
- Python 3.12 (Verify version with `python --version`)
- Git installed locally

### 1. Repository Setup & Environment Variables
If not already done, configure the environment variables:
```bash
# Clone or locate the directory
cd C:\Users\anand\.gemini\antigravity\scratch\ai_stock_advisor

# Copy the environment variable configuration template
cp .env.example .env
```
Open `.env` and fill in your actual API keys (`GEMINI_API_KEY` or `OPENAI_API_KEY`).

### 2. Setting Up Virtual Environment
Create and activate the virtual environment:
```powershell
# Create the virtual environment
python -m venv .venv

# Activate on Windows PowerShell
.venv\Scripts\Activate.ps1

# Or activate on Git Bash / Linux
source .venv/Scripts/activate
```

### 3. Installing Dependencies
Install production and developer packages:
```bash
pip install -r requirements-dev.txt
```

### 4. Running the Application
Run the entry point main script (optionally passing a stock ticker):
```bash
python src/ai_stock_advisor/main.py GOOG
```

### 5. Running Tests and Checks
Validate configuration parsing and formatting rules:
```bash
# Run test suite
pytest

# Format & Linting checks
ruff check .
ruff format . --check

# Strict Type validation
mypy src tests
```
