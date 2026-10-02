# Agentic Retail Forecasting & Design Pipeline

## Overview
This project is an automated pipeline for the retail fashion industry. It uses machine learning to predict top-selling items from historical transaction data, and an AI agent workflow to generate visual prototypes for the next season. 

## System Architecture

![Agent Architecture](https://github.com/user-attachments/assets/7749bc21-2e16-42a4-a0e7-7a9fd95e28b1)

*(Add a flowchart or diagram here showing the end-to-end flow from Kaggle Data -> XGBoost -> FastAPI -> Streamlit)*

The project consists of four main components:

### 1. Predictive Analytics
An XGBoost regression model trained on a chronological 80/20 split. It predicts 30-day forward sales volume to capture actual behavioral demand. It explicitly filters out low-margin commodities (like basic socks) to prioritize high-value apparel anchors.

### 2. Generative Concept Design
This module takes the metadata of the top forecasted items and updates them with seasonal design changes. It uses `stabilityai/stable-diffusion-xl-base-1.0` to generate 1024x1024 photorealistic images. The prompts strictly enforce 3D hollow garment cavities and block human models.

### 3. Agent Orchestration
A LangGraph state-graph workflow manages the system using specialized tools:
*   **Orchestrator:** Parses user intent and routes the workflow.
*   **Data Agent:** Runs the `forecast_seasonal_styles` tool to apply seasonal rules and isolate top anchor pieces.
*   **Design Agent:** Runs the `generate_seasonal_concepts` tool to craft final text-to-image prompts and trigger the Hugging Face API.

### 4. Full-Stack Dashboard
A FastAPI backend serves the ML predictions and routes the generated images. A Streamlit frontend consumes this API to provide a merchant-facing dashboard where stakeholders can view ranked assortments, compare historical baselines to AI concepts, and log feedback.

**Kaggle Dataset** ➔ **XGBoost ML Model** ➔ **FastAPI Backend** ➔ **Streamlit Frontend** ➔ **Business Stakeholder**

![Dashboard Detail View](https://github.com/user-attachments/assets/decd82e4-694b-4bf9-a005-174f0b785649)

## Repository Structure
```text
.
├── run_agent.py                # LangGraph conversational orchestrator
├── run_pipeline.py             # Sequential pipeline testing script
├── frontend_app.py             # Streamlit user interface
├── requirements.txt            # Environment dependencies
├── notebooks/                  # Jupyter notebooks for EDA and model training
├── data/                       # Raw and processed datasets (use .gitkeep)
├── models/                     # Saved XGBoost models (use .gitkeep)
└── src/
    ├── api.py                  # FastAPI backend server
    ├── agents.py               # LangGraph state and node definitions
    ├── concept_generator.py    # SDXL Text-to-Image API engine
    ├── forecaster.py           # ML sales prediction and evaluation
    ├── image_combiner.py       # Presentation assembly utilities
    └── skills.py               # MCP-compatible tools (@tool)


## Setup & Installation

### 1. Clone the repository

```bash
git clone <your-github-repo-url>
cd MerchMix

```

### 2. Install dependencies

(Requires Python 3.12+)

```bash
pip install -r requirements.txt

```

### 3. Configure Environment Variables

Create a `.env` file in the root directory:

```text
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here

```

### 4. Data Preparation & Model Training

*Git only tracks the folder structure (`.gitkeep`). You must download the raw data and generate the models locally.*

1. Download the [H&M Personalized Fashion Recommendations dataset](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations/data) from Kaggle.
2. Place `transactions_train.csv` and `articles.csv` into the `data/raw/` directory.
3. Run the notebook inside `notebooks/` to process the data, train the XGBoost model, and populate the `models/` and `data/processed/` folders.

## Usage


### A: Sequential Pipeline

Run the standard data science pipeline to evaluate XGBoost metrics (MAE, R²), view the dataframe predictions in the console, and execute the design generation in sequence. *(Note: Set `TASK_2_ENABLED = False` in the file to run a dry-run without consuming API tokens).

```bash
python run_pipeline.py

```

### B: Interactive Agent Workflow

Run the LangGraph orchestrator. Select a target season, and the agents will automatically filter data, engineer prompts, and generate images. *(Note: Set `TASK_3_ENABLED = False` in the file to run a dry-run without consuming API tokens).*

```bash
python run_agent.py

```

### C: Full-Stack Dashboard

To view the results in the interactive merchant UI, you need to run both the backend and frontend simultaneously.

**1. Start the FastAPI Backend:**
Open a terminal and run:

```bash
uvicorn src.api:app --reload

```

*(The API will be available at http://localhost:8000. You can view the docs at http://localhost:8000/docs)*

**2. Start the Streamlit Frontend:**
Open a second terminal window and run:

```bash
streamlit run frontend_app.py

```
