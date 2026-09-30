# Agentic Retail Forecasting & Design Pipeline

## Overview
This project has an automated pipeline for the retail fashion industry. It uses machine learning to predict top-selling items from historical transaction data, and an AI agent workflow to generate visual prototypes for the next season.

## System Architecture

![Agent Architecture](https://github.com/user-attachments/assets/7749bc21-2e16-42a4-a0e7-7a9fd95e28b1)

The project consists of three main components:

### 1. Predictive Analytics
An XGBoost regression model trained on a chronological 80/20 split. It predicts 30-day forward sales volume to capture actual behavioral demand. It filters out low-margin commodities (like basic socks) to prioritize high-value apparel.

### 2. Generative Concept Design
This module takes the metadata of the top forecasted items and updates them with seasonal design changes (e.g., adding cargo pockets or changing fabric weight). It uses `stabilityai/stable-diffusion-xl-base-1.0` to generate 1024x1024 photorealistic images. The prompts strictly enforce 3D hollow garment cavities and block human models.

### 3. Agent Orchestration
A LangGraph state-graph workflow manages the system using specialized tools:
*   **Orchestrator:** Parses user intent and routes the workflow.
*   **Data Agent:** Runs the `forecast_seasonal_styles` tool to apply seasonal rules and isolate top anchor pieces from the XGBoost predictions.
*   **Design Agent:** Runs the `generate_seasonal_concepts` tool to craft the final text-to-image prompts and trigger the Hugging Face API.

## Repository Structure
```text
.
├── run_agent.py                # LangGraph conversational orchestrator
├── run_pipeline.py             # Sequential pipeline testing script
├── requirements.txt            # Environment dependencies
├── src/
│   ├── agents.py               # LangGraph state and node definitions
│   ├── concept_generator.py    # SDXL Text-to-Image API engine
│   ├── forecaster.py           # ML sales prediction and evaluation
│   ├── image_combiner.py       # Presentation assembly utilities
│   └── skills.py               # MCP-compatible tools (@tool)
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <your-github-repo-url>
   cd MerchMix
   ```
   
2. **Install dependencies:
(Requires Python 3.12+)
```bash
  pip install -r requirements.txt
```

3. **Configure Environment Variables:
Create a .env file in the root directory:
```Code snippet
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```

4. **Usage
 - Interactive Agent Workflow:
Run the LangGraph orchestrator. Select a target season, and the agents will automatically filter the data, engineer the prompts, and generate the images.
```bash
  python run_agent.py
```

- Sequential Pipeline
Run the standard data science pipeline to evaluate XGBoost metrics (MAE, R²), view the dataframe predictions, and execute the design generation in sequence.
```
