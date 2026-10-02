"""
run_agent.py
Interactive CLI for Task 3 Agentic Workflow.
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage
from src.agents import merchmix_agent

# ==========================================
# Task Orchestration Configurations
# ==========================================
# TURN THIS TO "True" when ready to call Hugging Face API for Task 3
os.environ["TASK_3_ENABLED"] = "False"

def display_graph():
    """Renders the state node graph."""
    print("\n=== AGENT STATE GRAPH ===")
    try:
        print(merchmix_agent.get_graph().draw_ascii())
    except Exception as e:
        print(f"Could not render ASCII graph: {e}")

    try:
        png_bytes = merchmix_agent.get_graph().draw_mermaid_png()
        with open("agent_architecture_graph.png", "wb") as f:
            f.write(png_bytes)
        print("\n✅ Agent graph saved visually as 'agent_architecture_graph.png'")
    except Exception:
        print("\n(Note: Mermaid PNG export requires internet connection, skipping PNG render.)")


def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY missing in .env. LangGraph orchestrator requires this.")
        sys.exit(1)

    print("=== TASK 3: AGENTIC WORKFLOW ===")
    display_graph()

    print("\n--- Select a Target Season for the Next Collection ---")
    print("1. Winter (Focus: Heavyweight, Layers, Dark colors)")
    print("2. Summer (Focus: Breathable, Shorts, Light colors)")
    print("3. Autumn / Fall (Focus: Earthy tones, Knitwear, Outerwear)")
    print("4. Spring (Focus: Pastels, Lightweight tailoring)")

    choice = input("\nEnter 1, 2, 3, or 4: ").strip()

    seasons = {
        "1": "Winter",
        "2": "Summer",
        "3": "Autumn",
        "4": "Spring"
    }

    # Strict Validation: Exit if choice is not 1, 2, 3, or 4
    if choice not in seasons:
        print(f"\n❌ Invalid selection '{choice}'. Please enter only 1, 2, 3, or 4.")
        sys.exit(1)

    target_season = seasons[choice]

    user_query = (
        f"Please analyze the transactions, find the top styles, "
        f"and generate next-season concepts specifically optimized for {target_season}."
    )
    print(f"\n[ORCHESTRATOR TRIGGERED]: {user_query}")
    print("=" * 60)

    initial_state = {"messages": [HumanMessage(content=user_query)]}

    # Stream execution across nodes
    for output in merchmix_agent.stream(initial_state):
        for node_name, state_update in output.items():
            if "messages" in state_update:
                print(f"[{node_name}] {state_update['messages'][-1].content}")

    print("\n✅ Task 3 workflow execution finished.")


if __name__ == "__main__":
    main()
