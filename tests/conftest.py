import os
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

# 1. Environment variables set once for the entire session
os.environ.setdefault("OPENAI_API_KEY", "sk-mock-testing-key-for-test-suite")
os.environ.setdefault("HF_TOKEN", "hf-mock-testing-key-for-test-suite")
os.environ.setdefault("TASK_3_ENABLED", "False")

from src.api import app

@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client

@pytest.fixture(scope="session")
def real_held_out_test_slice():
    """Loads a slice of the real committed test parquet/data files if available."""
    processed_dir = Path("data/processed")
    x_test_path = processed_dir / "X_test.parquet"
    y_test_path = processed_dir / "y_test.parquet"

    # Support both parquet or csv if committed differently
    if not x_test_path.exists():
        x_candidates = list(processed_dir.glob("X_test.*"))
        y_candidates = list(processed_dir.glob("y_test.*"))
        if x_candidates and y_candidates:
            x_test_path = x_candidates[0]
            y_test_path = y_candidates[0]

    if x_test_path.exists() and y_test_path.exists():
        if x_test_path.suffix == ".parquet":
            x_df = pd.read_parquet(x_test_path)
            y_df = pd.read_parquet(y_test_path)
        else:
            x_df = pd.read_csv(x_test_path)
            y_df = pd.read_csv(y_test_path)
        # Take a clean 200-row slice
        slice_len = min(200, len(x_df))
        return {
            "X": x_df.iloc[:slice_len],
            "y": y_df.iloc[:slice_len].values.ravel()
        }
    return None

@pytest.fixture
def mock_hf_success_response(mocker):
    """Mocks Hugging Face 200 OK image byte response."""
    dummy_png_bytes = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
        b'\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00'
        b'\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4'
        b'\x00\x00\x00\x00IEND\xaeB`\x82'
    )
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = dummy_png_bytes
    return mocker.patch("requests.post", return_value=mock_resp)