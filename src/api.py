import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.model_service import ModelService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MerchMix Prediction & Design API", version="1.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = ModelService()

class ProductInfo(BaseModel):
    article_id: str
    category: str
    color: str
    demographic: str
    pattern: str

class HistoricalPerformance(BaseModel):
    sales_last_30d: int
    sales_velocity_status: str

class ModelExplanation(BaseModel):
    primary_driver: str
    confidence_logic: str
    reasoning: str

class StyleTopResponse(BaseModel):
    style_id: str
    prediction_score: float
    predicted_volume: int
    category: str
    sales_history: int

class StyleDetailResponse(BaseModel):
    style_id: str
    prediction_score: float
    predicted_volume: int
    product_information: ProductInfo
    historical_performance: HistoricalPerformance
    model_explanation: ModelExplanation

class FeedbackRequest(BaseModel):
    style_id: str
    feedback_type: str 
    comments: str = ""

@app.get("/styles/top", response_model=dict, tags=["Predictions"])
def get_top_styles(limit: int = 3):
    return {"styles": service.get_top_styles(limit=limit)}

@app.get("/styles/{style_id}", response_model=StyleDetailResponse, tags=["Predictions"])
def get_style(style_id: str):
    details = service.get_style_details(style_id)
    if not details:
        raise HTTPException(status_code=404, detail="Style ID not found.")
    return details

@app.post("/styles/feedback", tags=["User Interaction"])
def submit_feedback(feedback: FeedbackRequest):
    if feedback.feedback_type not in ['like', 'dislike']:
        raise HTTPException(status_code=400, detail="feedback_type must be 'like' or 'dislike'")
    return {"status": "success", "message": "Feedback successfully logged."}

@app.get("/styles/{style_id}/image/{image_type}", tags=["Visual Assets"])
def get_style_image(style_id: str, image_type: str):
    """Serves images. Concepts use readable format: demographic_color_category.png"""
    if image_type not in ['original', 'concept']:
        raise HTTPException(status_code=400, detail="Type must be 'original' or 'concept'")
    
    details = service.get_style_details(style_id)
    if not details:
        raise HTTPException(status_code=404, detail="Style not found")
        
    file_path = None
    
    if image_type == 'original':
        article_id = details['product_information']['article_id']
        temp_path = f"images/task2/originals/{article_id}.jpg"
        if os.path.exists(temp_path):
            file_path = temp_path
    
    elif image_type == 'concept':
        raw_demo = details['product_information']['demographic']
        raw_color = details['product_information']['color']
        raw_type = details['product_information']['category']
        
        demo_clean = str(raw_demo).lower().replace(',', '').replace('/', '_').replace(' ', '_')
        color_clean = str(raw_color).lower().replace(',', '').replace('/', '_').replace(' ', '_')
        type_clean = str(raw_type).lower().replace(',', '').replace('/', '_').replace(' ', '_')
        
        concept_filename = f"{demo_clean}_{color_clean}_{type_clean}.png"
        temp_path = f"images/task2/concepts/{concept_filename}"
        
        if os.path.exists(temp_path):
            file_path = temp_path

    if not file_path:
        raise HTTPException(status_code=404, detail=f"Image for {style_id} not found locally.")
        
    return FileResponse(file_path)

@app.get("/presentation/board", tags=["Visual Assets"])
def get_presentation_board():
    file_path = "images/task2/final_concepts_presentation.png"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Combined presentation board not found.")
    return FileResponse(file_path)