import pytest
from pydantic import ValidationError
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage

from src.skills import (
    forecast_seasonal_styles,
    SeasonForecastInput,
    ConceptDesignInput,
)
from src.agents import merchmix_agent


# =====================================================================
# 1. MCP TOOL CONTRACTS & SCHEMA VALIDATION
# =====================================================================

def test_mcp_season_forecast_tool_schema():
    """Assert SeasonForecastInput requires a valid season string."""
    valid = SeasonForecastInput(season="Winter")
    assert valid.season == "Winter"

    with pytest.raises(ValidationError):
        SeasonForecastInput()


def test_mcp_concept_design_tool_schema():
    """Assert ConceptDesignInput requires valid styles_context and season."""
    valid_payload = {
        "styles_context": [{"article_id": "0108775015", "product_type_name": "Hoodie", "colour_group_name": "Black"}],
        "season": "Summer"
    }
    model = ConceptDesignInput(**valid_payload)
    assert len(model.styles_context) == 1
    assert model.season == "Summer"

    with pytest.raises(ValidationError):
        ConceptDesignInput(styles_context=[])


def test_forecast_seasonal_styles_execution(mocker):
    """Assert tool executes business rules: commodity filtering and category deduplication."""
    import pandas as pd
    dummy_df = pd.DataFrame({
        "article_id": ["0101", "0102", "0103", "0104"],
        "product_type_name": ["Sweater", "Coat", "Socks", "Sweater"],
        "colour_group_name": ["Black", "Dark Grey", "White", "Black"],
        "index_name": ["Ladieswear", "Ladieswear", "Divided", "Menswear"],
        "predicted_30d_sales_volume": [500.0, 420.0, 900.0, 310.0]
    })
    mocker.patch("src.skills.forecaster.predict_top_performers", return_value=dummy_df)

    # Calling the tool function directly (bypassing any wrapper inconsistency)
    tool_func = getattr(forecast_seasonal_styles, "func", forecast_seasonal_styles)
    results = tool_func(season="Winter")

    assert isinstance(results, list)
    assert len(results) == 2
    assert [r["product_type_name"] for r in results] == ["Sweater", "Coat"]


# =====================================================================
# 2. LANGGRAPH CONDITIONAL ROUTING
# =====================================================================

def test_agent_routes_off_topic_query_to_direct_response():
    """Validates that conversational or off-scope queries exit early via Direct_Response."""
    initial_state = {
        "messages": [HumanMessage(content="Hello, what time is it in London?")]
    }

    final_state = merchmix_agent.invoke(initial_state)

    assert final_state["route_to"] == "Direct_Response"
    assert final_state["status"] == "completed_direct"


# =====================================================================
# 3. END-TO-END WORKFLOW NODE EXECUTION (WITHOUT ALTERING src/agents.py)
# =====================================================================

def test_agent_full_pipeline_execution_and_tool_call_sequence(mocker):
    """Validates full workflow by mocking at the node/external boundaries.
    Leaves src/agents.py completely intact and runs with zero external API calls.
    """
    mock_forecasted_articles = [
        {"article_id": "0108775015", "product_type_name": "Hoodie", "colour_group_name": "Dark Blue", "index_name": "Menswear"},
        {"article_id": "0209886022", "product_type_name": "Sweater", "colour_group_name": "Black", "index_name": "Ladieswear"}
    ]

    # Create dummy callable tools that record calls and return clean outputs
    mock_forecast_tool = MagicMock()
    mock_forecast_tool.invoke.return_value = mock_forecasted_articles
    mock_forecast_tool.run.return_value = mock_forecasted_articles

    mock_kaggle_tool = MagicMock()
    mock_kaggle_tool.invoke.return_value = "Downloaded 2 references"
    mock_kaggle_tool.run.return_value = "Downloaded 2 references"

    mock_concept_tool = MagicMock()
    mock_concept_tool.invoke.return_value = "Generated 2 concepts"
    mock_concept_tool.run.return_value = "Generated 2 concepts"

    # Patch the tool objects on the agents module directly
    mocker.patch("src.agents.forecast_seasonal_styles", mock_forecast_tool)
    mocker.patch("src.agents.fetch_reference_images", mock_kaggle_tool)
    mocker.patch("src.agents.generate_seasonal_concepts", mock_concept_tool)

    # Mock the PIL combiner so no disk image operations run
    mock_combiner = mocker.patch(
        "src.image_combiner.ConceptImageCombiner.combine_seasonal_concepts",
        return_value=True
    )

    # Run the workflow
    initial_state = {
        "messages": [HumanMessage(content="Please forecast and design top winter concepts for our catalog.")]
    }

    final_state = merchmix_agent.invoke(initial_state)

    # Assert Intent & Graph Progression
    assert final_state["season"] == "Winter"
    assert final_state["route_to"] == "Data_Agent"
    assert final_state["status"] == "all_completed"
    assert len(final_state["top_articles"]) == 2

    # Assert image combiner was invoked for the identified season
    mock_combiner.assert_called_once_with(
        input_dir="images/task3/concepts",
        output_file="images/task3/presentation_winter.png",
        season="Winter",
        expected_count=3
    )