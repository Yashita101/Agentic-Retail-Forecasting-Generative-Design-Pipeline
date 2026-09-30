"""
src/agents.py
LangGraph Orchestrator with Dedicated MCP Nodes and Conditional Edges.
"""
from typing import TypedDict, Sequence, Annotated
import operator
from pathlib import Path
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from src.skills import forecast_seasonal_styles, fetch_reference_images, generate_seasonal_concepts
from src.image_combiner import ConceptImageCombiner


class WorkflowState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    season: str
    top_articles: list[dict]
    route_to: str
    status: str


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def orchestrator_node(state: WorkflowState):
    """Analyzes intent, extracts season, and decides routing."""
    user_query = state["messages"][0].content.lower()
    
    if "winter" in user_query:
        season = "Winter"
    elif "summer" in user_query:
        season = "Summer"
    elif "autumn" in user_query or "fall" in user_query:
        season = "Autumn"
    else:
        season = "Spring"
    
    # Intent classification
    is_valid = any(kw in user_query for kw in ["generate", "design", "analyze", "forecast", "concept"])
    route = "Data_Agent" if is_valid else "Direct_Response"

    return {
        "season": season,
        "route_to": route,
        "status": "initialized",
        "messages": [AIMessage(content=f"Orchestrator: Identified target season '{season}'. Routing to '{route}'.")]
    }


def router_logic(state: WorkflowState):
    """Conditional edge branch selector."""
    return state.get("route_to", "Direct_Response")


def direct_response_node(state: WorkflowState):
    """Fallback handler for off-topic queries."""
    return {
        "status": "completed_direct",
        "messages": [AIMessage(content="Orchestrator: Query does not match merchandise planning scope.")]
    }


def data_sub_agent_node(state: WorkflowState):
    """Coordinates data extraction and sets up MCP execution."""
    season = state["season"]
    return {
        "status": "data_agent_ready",
        "messages": [AIMessage(content=f"Data Sub-Agent: Initializing MCP tools for {season} demand extraction.")]
    }


def mcp_seasonal_forecast_node(state: WorkflowState):
    """Executes the Seasonal Forecasting MCP Tool."""
    season = state["season"]
    top_styles = forecast_seasonal_styles.invoke({"season": season})
    return {
        "top_articles": top_styles,
        "status": "forecast_completed",
        "messages": [AIMessage(content=f"MCP Forecast: Identified {len(top_styles)} volume drivers for {season}.")]
    }


def mcp_kaggle_bridge_node(state: WorkflowState):
    """Executes the Kaggle Reference Download MCP Tool."""
    articles = state.get("top_articles", [])
    article_ids = [str(item["article_id"]) for item in articles]
    fetch_result = fetch_reference_images.invoke({"article_ids": article_ids})
    return {
        "status": "kaggle_fetch_completed",
        "messages": [AIMessage(content=f"MCP Kaggle: {fetch_result}")]
    }


def design_sub_agent_node(state: WorkflowState):
    """Coordinates generative briefing."""
    season = state["season"]
    return {
        "status": "design_agent_ready",
        "messages": [AIMessage(content=f"Design Sub-Agent: Preparing prompt synthesis for {season} collection.")]
    }


def mcp_concept_generator_node(state: WorkflowState):
    """Executes the Generative Diffusion MCP Tool."""
    season = state["season"]
    articles = state.get("top_articles", [])
    
    design_result = generate_seasonal_concepts.invoke({
        "styles_context": articles,
        "season": season
    })
    return {
        "status": "concepts_completed",
        "messages": [AIMessage(content=f"MCP Concept Generator: {design_result}")]
    }


def presentation_assembly_node(state: WorkflowState):
    """Stitches final concepts into a cohesive presentation board."""
    season = state["season"]
    combiner = ConceptImageCombiner()
    output_path = f"images/task3/presentation_{season.lower()}.png"
    
    result = combiner.combine_seasonal_concepts(
        input_dir="images/task3/concepts",
        output_file=output_path,
        season=season,
        expected_count=3
    )
    
    status_msg = f"Saved to {output_path}" if result else "Stitching skipped or incomplete."
    return {
        "status": "all_completed",
        "messages": [AIMessage(content=f"Presentation Assembly: {status_msg}")]
    }


# ==========================================
# Graph Definition with Explicit Tool Nodes
# ==========================================
workflow = StateGraph(WorkflowState)

workflow.add_node("Orchestrator", orchestrator_node)
workflow.add_node("Direct_Response", direct_response_node)
workflow.add_node("Data_Agent", data_sub_agent_node)
workflow.add_node("MCP_Forecast", mcp_seasonal_forecast_node)
workflow.add_node("MCP_Kaggle", mcp_kaggle_bridge_node)
workflow.add_node("Design_Agent", design_sub_agent_node)
workflow.add_node("MCP_ConceptGen", mcp_concept_generator_node)
workflow.add_node("Assemble_Presentation", presentation_assembly_node)

workflow.set_entry_point("Orchestrator")

# Conditional Edge (Decision Diamond)
workflow.add_conditional_edges(
    "Orchestrator",
    router_logic,
    {
        "Data_Agent": "Data_Agent",
        "Direct_Response": "Direct_Response"
    }
)

# Deterministic Flow
workflow.add_edge("Data_Agent", "MCP_Forecast")
workflow.add_edge("MCP_Forecast", "MCP_Kaggle")
workflow.add_edge("MCP_Kaggle", "Design_Agent")
workflow.add_edge("Design_Agent", "MCP_ConceptGen")
workflow.add_edge("MCP_ConceptGen", "Assemble_Presentation")
workflow.add_edge("Assemble_Presentation", END)
workflow.add_edge("Direct_Response", END)

merchmix_agent = workflow.compile()