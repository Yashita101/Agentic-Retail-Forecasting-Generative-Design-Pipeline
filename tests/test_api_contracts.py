import pytest

# 1. Top Styles Endpoint Contract
def test_get_top_styles_contract(client):
    """Assert status 200, valid JSON schema, and boundary limit execution."""
    res_default = client.get("/styles/top")
    assert res_default.status_code == 200
    assert "styles" in res_default.json()
    
    response = client.get("/styles/top?limit=3")
    assert response.status_code == 200
    data = response.json()
    assert "styles" in data
    assert isinstance(data["styles"], list)
    assert len(data["styles"]) <= 3
    
    # Limit check: requesting 1 returns exactly 1
    res_one = client.get("/styles/top?limit=1")
    assert len(res_one.json()["styles"]) == 1

    # Contract schema validation
    required_keys = {"style_id", "prediction_score", "predicted_volume", "category", "sales_history"}
    for item in data["styles"]:
        assert required_keys.issubset(item.keys())
        assert isinstance(item["style_id"], str)
        assert isinstance(item["predicted_volume"], int)
        assert 0.0 <= item["prediction_score"] <= 1.0

# 2. Single Style Lookups
def test_get_style_invalid_id_returns_404(client):
    """Negative test: Non-existent style_id returns HTTP 404."""
    response = client.get("/styles/non_existent_style_999999")
    assert response.status_code == 404
    assert "detail" in response.json()

# 3. User Feedback Workflow (Contract, Bad Enum, Missing Fields)
def test_feedback_endpoint_contract(client):
    """Verifies feedback submission accepts valid types and rejects malformed payloads."""
    # Valid submission
    valid = {"style_id": "test_id_101", "feedback_type": "like", "comments": "Good fit"}
    res_valid = client.post("/styles/feedback", json=valid)
    assert res_valid.status_code == 200
    assert res_valid.json()["status"] == "success"

    # Invalid enum value
    invalid_enum = {"style_id": "test_id_101", "feedback_type": "neutral"}
    assert client.post("/styles/feedback", json=invalid_enum).status_code == 400

    # Missing mandatory field
    missing_field = {"style_id": "test_id_101"}
    assert client.post("/styles/feedback", json=missing_field).status_code == 422

# 4. Visual Assets Endpoints
def test_presentation_board_endpoint(client):
    """Asserts presentation board returns 200 with image/png header (or 404 if file absent)."""
    response = client.get("/presentation/board")
    assert response.status_code in (200, 404)
    if response.status_code == 200:
        assert response.headers["content-type"].startswith("image/")

def test_style_image_invalid_type_returns_400(client):
    """Asserts invalid image type param or non-existent style"""
    response = client.get("/styles/0108775015/image/invalid_type")
    assert response.status_code == 400
    assert response.json()["detail"] == "Type must be 'original' or 'concept'"

    response = client.get("/styles/unknown_style_999999/image/original")
    assert response.status_code == 404