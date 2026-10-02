import os
import subprocess
import pandas as pd
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from src.forecaster import SalesForecaster
from src.concept_generator import HFDiffusionConceptGenerator
from pathlib import Path

# --- GRACEFUL INITIALIZATION & DRY RUN SAFETY NET ---
TASK_3_ENABLED = os.environ.get("TASK_3_ENABLED", "False").lower() == "true"

# Initialize core engines for Agent Use
forecaster = SalesForecaster(models_dir="models")

if TASK_3_ENABLED:
    try:
        generator = HFDiffusionConceptGenerator()
        AGENT_GENERATION_ENABLED = True
    except ValueError as e:
        print("\n" + "="*60)
        print("⚠️  [ENVIRONMENT CONFIGURATION MISSING]")
        print(f"Details: {e}")
        print("Action : Please add your 'HF_TOKEN' to the .env file to enable image generation.")
        print("Status : The Agent will continue in DRY RUN mode (simulating API calls).")
        print("="*60 + "\n")
        generator = None
        AGENT_GENERATION_ENABLED = False
    except ImportError as e:
        print(f"\n⚠ [DEPENDENCY MISSING]: {e}\nAgent will simulate generation.")
        generator = None
        AGENT_GENERATION_ENABLED = False
else:
    # Explicitly bypass initialization if the user sets it to False
    generator = None
    AGENT_GENERATION_ENABLED = False

# ==========================================
# MCP SCHEMAS (Strict typing for Agent tools)
# ==========================================
class SeasonForecastInput(BaseModel):
    season: str = Field(description="The target season to filter predictions for, e.g., 'Winter', 'Summer', 'Spring'.")

class FetchImageInput(BaseModel):
    article_ids: list[str] = Field(description="List of 10-digit H&M article IDs to download from Kaggle.")

class ConceptDesignInput(BaseModel):
    styles_context: list[dict] = Field(description="List of dictionaries containing 'article_id', 'product_type_name', and 'colour_group_name'.")
    season: str = Field(description="The target season for design adaptation.")

# ==========================================
# SKILL 1: Seasonal Forecaster (Dataset-Mapped)
# ==========================================
@tool("forecast_seasonal_styles", args_schema=SeasonForecastInput)
def forecast_seasonal_styles(season: str) -> list[dict]:
    """Forecasts top performing styles from the entire catalog and applies a seasonal filter."""
    print(f"\n[Data Agent] 📊 Forecasting entire catalog for {season}...")
    
    all_preds = forecaster.predict_top_performers(processed_data_dir="data/processed", limit_distinct=1000)

    # 1. Standardized 4-season taxonomy mapped to the H&M categories
    SEASON_RULES = {
        "winter": {
            "sweater": 3, "cardigan": 3, "coat": 3, "hoodie": 2, "jacket": 2, 
            "outdoor waistcoat": 2, "outdoor trousers": 2, "leggings/tights": 1,
            "black": 1, "dark grey": 1, "dark blue": 1
        },
        "spring": {
            "blouse": 3, "shirt": 3, "blazer": 2, "jumpsuit/playsuit": 2, 
            "cardigan": 2, "skirt": 2, "dress": 2, "trousers": 1, "top": 1,
            "light blue": 1, "light pink": 1, "off white": 1, "beige": 1
        },
        "summer": {
            "vest top": 3, "shorts": 3, "t-shirt": 2, "top": 2, "polo shirt": 2,
            "dress": 2, "skirt": 2, "swimwear set": 2, "swimsuit": 2,
            "white": 1, "off white": 1, "yellow": 1, "orange": 1
        },
        "autumn": {
            "jacket": 3, "coat": 3, "sweater": 3, "tailored waistcoat": 2, 
            "blazer": 2, "trousers": 2, "cardigan": 1,
            "greenish khaki": 1, "brown": 1, "dark brown": 1, "dark grey": 1, "mustard": 1
        }
    }

    norm_season = season.strip().lower()
    if norm_season == "fall":
        norm_season = "autumn"
    
    season_weights = SEASON_RULES.get(norm_season, {})

    def score_row(row):
        score = 0
        p_type = str(row["product_type_name"]).strip().lower()
        color = str(row["colour_group_name"]).strip().lower()

        for kw, weight in season_weights.items():
            if kw in p_type or kw in color:
                score += weight
        return score

    all_preds["season_score"] = all_preds.apply(score_row, axis=1)

    # 2. Exclude basic intimates
    excluded_items = {
        'accessories set', 'alice band', 'baby bib', 'bag', 'backpack', 'bracelet', 
        'bra', 'bra extender', 'bumbag', 'chem. cosmetics', 'clothing mist', 'cushion', 
        'dog wear', 'earring', 'earrings', 'eyeglasses', 'fine cosmetics', 'giftbox', 
        'hair string', 'hair ties', 'hair/alice band', 'hairband', 'hairclip', 
        'keychain', 'marker pen', 'mobile case', 'necklace', 'nipple covers', 
        'other accessories', 'ring', 'sewing kit', 'side table', 'soft toys', 
        'stain remover spray', 'sunglasses', 'towel', 'umbrella', 'wallet', 
        'washing bag', 'watch', 'waterbottle', 'wood balls', 'zipper head',
        'socks', 'underdress', 'underwear tights', 'underwear body', 
        'underwear bottom', 'underwear corset', 'underwear set', 'kids underwear top',
        'pyjama bottom', 'pyjama jumpsuit/playsuit', 'pyjama set', 'night gown', 'robe'
    }
    
    design_candidates = all_preds[~all_preds["product_type_name"].str.lower().isin(excluded_items)]

    # 3. Enforce distinct categories for the top 3
    top_seasonal = (
        design_candidates.sort_values(
            by=["season_score", "predicted_30d_sales_volume"], 
            ascending=[False, False]
        )
        .drop_duplicates(subset=["product_type_name"])
        .head(3)
    )

    return top_seasonal[["article_id", "product_type_name", "colour_group_name", "index_name"]].to_dict(orient="records")

# ==========================================
# SKILL 2: Kaggle CLI Secure Bridge
# ==========================================
@tool("fetch_reference_images", args_schema=FetchImageInput)
def fetch_reference_images(article_ids: list[str]) -> str:
    """Uses Kaggle CLI to securely download reference images into an isolated agent folder."""
    print(f"\n[Data Agent] 📥 Fetching original references for: {article_ids}...")
    
    output_dir = "images/task3/originals"
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded = []
    for article in article_ids:
        clean_id = str(article).zfill(10)
        folder = clean_id[:3]
        file_path = f"images/{folder}/{clean_id}.jpg"
        
        command = [
            "kaggle", "competitions", "download", "-c", "h-and-m-personalized-fashion-recommendations", 
            "-f", file_path, "-p", output_dir
        ]
        
        try:
            subprocess.run(command, check=True) 
            downloaded.append(clean_id)
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Failed to fetch {clean_id}. Please ensure you have accepted the competition rules on Kaggle's website.")
            
    return f"Successfully downloaded original references to {output_dir}/: {', '.join(downloaded)}"

# ==========================================
# SKILL 3: Autonomous Concept Generator
# ==========================================
@tool("generate_seasonal_concepts", args_schema=ConceptDesignInput)
def generate_seasonal_concepts(styles_context: list[dict], season: str) -> str:
    """Generates next-season AI fashion concepts using strong structural prompts."""
    
    global AGENT_GENERATION_ENABLED, generator
    
    print(f"\n[Design Agent] 🎨 Formulating {season} concept prompts...")
    
    season_modifiers = {
        "winter": "Heavyweight insulating fabrics, structured winter-ready construction.",
        "spring": "Lightweight breathable fabrics, clean transitional tailoring.",
        "summer": "Airy open-weave fabric, fluid relaxed hot-weather drape.",
        "autumn": "Mid-weight textured twill, rich earthy layered styling."
    }
    modifier = season_modifiers.get(season.lower(), "High-quality construction.")
    
    if generator:
        generator._output_dir = Path("images/task3/concepts")
        concepts_dir = Path(generator._output_dir)
    else:
        concepts_dir = Path("images/task3/concepts")
        
    originals_dir = Path("images/task3/originals")
    
    saved_paths = []
    for item in styles_context:
        p_type = item.get("product_type_name", "Garment")
        color = item.get("colour_group_name", "Color")
        demo = item.get("index_name", "General") 
        article_id = str(item.get("article_id", "0000000000")).zfill(10)
        
        ptype_lower = p_type.lower()
        
        if any(w in ptype_lower for w in ["top", "shirt", "blouse", "sweater", "cardigan", "hoodie", "jacket", "waistcoat", "vest", "t-shirt"]):
            cut = "featuring a realistic hollow 3D neck cavity showing the inside back of the collar, isolated upper-body garment"
        elif any(w in ptype_lower for w in ["trouser", "short", "legging", "skirt"]):
            cut = "featuring a realistic hollow 3D open waistband cavity, isolated lower-body garment"
        elif any(w in ptype_lower for w in ["dress", "jumpsuit"]):
            cut = "featuring a realistic hollow 3D neck cavity, full-length isolated garment"
        else:
            cut = "featuring a hollow 3D opening, isolated single garment"
            
        prompt = (
            f"Commercial e-commerce product catalog shot of exactly ONE single isolated {color} {p_type}. "
            f"Strictly displayed on an invisible ghost mannequin, {cut}, centered front-facing view. "
            f"Basic commercial design modification with subtle refined stitching, preserving the original {p_type} silhouette exactly. "
            f"{modifier} High fashion commercial studio lighting, neutral light-grey backdrop, 8k resolution, photorealistic clothing item. "
            f"THIS IS A 3D GARMENT, NOT A FLAT PATTERN."
        )
        
        neg_prompt = (
            "ABSOLUTELY NO HUMAN MODEL, no face, no skin, no hands, no arms, no legs, no physical body inside the garment, "
            "no mannequin head, no hanger. fabric swatch, seamless pattern, wallpaper, flat 2d design, background texture, "
            "macro shot, close up, sketch, drawing, illustration, cartoon, 3d render, multiple views, multiple items, "
            "cropped, blurry, low quality, collage, storyboard."
        )
        
        demo_clean = str(demo).lower().replace(',', '').replace('/', '_').replace(' ', '_')
        color_clean = str(color).lower().replace(',', '').replace('/', '_').replace(' ', '_')
        type_clean = str(p_type).lower().replace(',', '').replace('/', '_').replace(' ', '_')
        
        stakeholder_filename = f"{demo_clean}_{color_clean}_{type_clean}"
        
        target_original_path = originals_dir / f"{article_id}.jpg"
        target_concept_path = concepts_dir / f"{stakeholder_filename}.png"
        
        print(f"\n--- DRY RUN: PROMPT FOR '{color} {p_type}' ---")
        print(f"  > Reference Original: {target_original_path.as_posix()}")
        print(f"  > Target Concept:     {target_concept_path.as_posix()}")
        print(f"Positive: {prompt}")
        print(f"Negative: {neg_prompt}")
        
        if AGENT_GENERATION_ENABLED and generator:
            try:
                out_path = generator.generate_concept(prompt, neg_prompt, stakeholder_filename) 
                saved_paths.append(str(out_path))
            except Exception as e:
                print(f"  ❌ Error generating {p_type}: {e}")
        else:
            saved_paths.append(target_concept_path.as_posix())
            
    status_msg = "Generated via API." if AGENT_GENERATION_ENABLED else "Dry run complete (no API consumed)."
    return f"Processed {len(saved_paths)} concepts. Mode: {status_msg}"