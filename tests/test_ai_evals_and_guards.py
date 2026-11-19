import time
import pytest
from src.skills import forecast_seasonal_styles
from src.concept_generator import build_sdxl_prompt

# 1. Pipeline Grounding
def test_pipeline_grounding_forecast_to_prompt(mocker):
    """Pipeline Grounding: Asserts attributes chosen by the forecaster are retained in prompt construction."""
    import pandas as pd
    dummy_df = pd.DataFrame({
        "article_id": ["0101"],
        "product_type_name": ["Trousers"],
        "colour_group_name": ["Black"],
        "index_name": ["Divided"],
        "predicted_30d_sales_volume": [500.0]
    })
    mocker.patch("src.skills.forecaster.predict_top_performers", return_value=dummy_df)
    
    top_items = forecast_seasonal_styles.invoke({"season": "Summer"})
    chosen = top_items[0]

    prompt, _ = build_sdxl_prompt(
        product_type=chosen["product_type_name"],
        color=chosen["colour_group_name"],
        season="Summer"
    )

    assert "black" in prompt.lower()
    assert "trousers" in prompt.lower()

# 2. Latency Smoke Check
def test_top_styles_api_latency_smoke_check(client):
    """Smoke check: Asserts in-memory route execution completes within reasonable budget."""
    start = time.perf_counter()
    res = client.get("/styles/top?limit=3")
    duration = time.perf_counter() - start

    assert res.status_code == 200
    assert duration < 2.0, f"Call exceeded 2.0s smoke threshold: took {duration:.3f}s"