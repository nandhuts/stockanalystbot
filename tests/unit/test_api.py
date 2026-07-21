from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pandas as pd
import pytest

from ai_stock_advisor.api.main import app
from ai_stock_advisor.db.database import get_db, User
from ai_stock_advisor.api.auth import create_access_token


@pytest.fixture
def test_client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def mock_db() -> MagicMock:
    """Mock database SessionLocal instance."""
    return MagicMock()


def test_register_user_success(test_client: TestClient, mock_db: MagicMock) -> None:
    """Ensure register user hashes password and writes record to database."""
    # Override get_db injection dependency
    app.dependency_overrides[get_db] = lambda: mock_db
    
    mock_db.query().filter().first.return_value = None  # User doesn't exist yet
    
    payload = {"username": "test_trader", "password": "securepassword123"}
    response = test_client.post("/api/register", json=payload)
    
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "test_trader"
    assert "id" in data
    
    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()
    
    # Cleanup overrides
    app.dependency_overrides.clear()


def test_token_login_success(test_client: TestClient, mock_db: MagicMock) -> None:
    """Ensure login with correct credentials issues bearer JWT token."""
    app.dependency_overrides[get_db] = lambda: mock_db
    
    from ai_stock_advisor.api.auth import get_password_hash
    mock_user = User(id=1, username="test_trader", hashed_password=get_password_hash("securepassword123"))
    mock_db.query().filter().first.return_value = mock_user

    payload = {"username": "test_trader", "password": "securepassword123"}
    response = test_client.post("/api/token", data=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["token_type"] == "bearer"
    assert "access_token" in data
    
    app.dependency_overrides.clear()


def test_get_rankings_protected_endpoint(test_client: TestClient, mock_db: MagicMock) -> None:
    """Ensure protected endpoint returns 401 on missing JWT and loads data on valid JWT."""
    # 1. Access without token should be blocked
    response = test_client.get("/api/rankings")
    assert response.status_code == 401
    
    # 2. Access with valid token should load rankings
    app.dependency_overrides[get_db] = lambda: mock_db
    
    mock_user = User(id=1, username="test_trader", hashed_password="hashed_pwd")
    mock_db.query().filter().first.return_value = mock_user
    
    # Generate valid test token
    token = create_access_token(data={"sub": "test_trader"})
    headers = {"Authorization": f"Bearer {token}"}
    
    # Mock rankings dataframe load
    mock_df = pd.DataFrame([{"Ticker": "RELIANCE.NS", "Probability_Score": 85.0}])
    with patch("pathlib.Path.exists", return_value=True):
        with patch("pandas.read_csv", return_value=mock_df):
            response = test_client.get("/api/rankings", headers=headers)
            
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 1
            assert data[0]["Ticker"] == "RELIANCE.NS"
            
    app.dependency_overrides.clear()
