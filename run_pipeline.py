"""
run_pipeline.py
Corporate Data Science Pipeline: Retail Forecasting and Generative Concept Design.
"""
from src.forecaster import SalesForecaster
from src.concept_generator import HFDiffusionConceptGenerator
import pandas as pd
from pathlib import Path

# ==========================================
# Task Orchestration Configurations
# ==========================================
# TURN THIS TO TRUE when ready to call Hugging Face API
TASK_2_ENABLED = True

# Data & Model Locations
DATA_DIR = "data/processed"
MODELS_DIR = "models"
DISTINCT_ASSORTMENT_COUNT = 4


# ==========================================
# DESIGN PROMPT TEMPLATE RESTORATION
# For human design visibility/control
# ==========================================
_MASTER_STYLE = (
    "Product catalog shot, single isolated garment on invisible ghost mannequin, centered, front view. "
    "Editorial fashion photography. Minimalist concrete studio floor with dramatic top-lighting and a neutral "
    "slate-grey background. High fashion catalog style, 8k resolution, highly detailed texture. Photorealistic."
)

_SPECIFIC_DESIGN_CHANGES = {
    "Hoodie": "Heavyweight structured fleece fabric. Feature subtle monochrome intricate beetle motif embroidered on right chest area.",
    "Shirt": "Lightweight fluid matte black viscose twill. Feature integrated thin metal hardware clips and a removable matching waist belt.",
    "Trousers": "Relaxed track-pant drape with elastic cuffs. Feature very deep slit cargo pockets and a clean hidden waistband (internal tie) for a clean aesthetic."
}

_NEGATIVE_PROMPT = (
    "human, model, woman, man, female, male, face, skin, body, eyes, head, limbs, arms, legs, "
    "full body outfit, mannequin head, standing pose, blur, distorted, sketch, drawing, "
    "illustration, 3d render, flat 2d design, seamless pattern, fabric swatch"
)

def _build_design_prompt(product_type: str, color: str) -> str:
    """Combines metadata traits, master style, and explicit design upgrades."""
    design_delta = _SPECIFIC_DESIGN_CHANGES.get(product_type, f"Solid {color} high-quality fabric construction.")
    full_prompt = f"Genuinely new next-season clothing concept for a {color} {product_type}. {design_delta} {_MASTER_STYLE}"
    return full_prompt


def main():
    global TASK_2_ENABLED
    print("--- MERCHMIX PIPELINE START ---\n")

    # ==========================================
    # SUBTASK 1: Analytics & Evaluation (Distinct)
    # ==========================================
    pipeline_engine = SalesForecaster(models_dir=MODELS_DIR)
    
    # 1A. Evaluate metrics on unit scale
    metrics = pipeline_engine.evaluate_performance(processed_data_dir=DATA_DIR)
    print("\n=== Subtask 1 Metrics (Unit Volume Scale) ===")
    print(f"  MAE:  {metrics['mae']:.2f} garments")
    print(f"  RMSE: {metrics['rmse']:.2f} garments")
    print(f"  R²:   {metrics['r2']:.4f} (variance explained)")

    # 1B. Forecast full catalog and select winners
    winners = pipeline_engine.predict_top_performers(processed_data_dir=DATA_DIR, limit_distinct=DISTINCT_ASSORTMENT_COUNT)
    print(f"\n=== Top {DISTINCT_ASSORTMENT_COUNT} High-Volume Winners (For Task 2 Assortment) ===")
    
    columns_to_show = ["article_id", "product_type_name", "colour_group_name", "sales_last_30d", "predicted_30d_sales_volume"]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(winners[columns_to_show])


    # ==========================================
    # SUBTASK 2: Generative Design (Guarded)
    # ==========================================
    print("\n--- Transition to Subtask 2 Concepts ---")
    if not TASK_2_ENABLED:
        print("INFO: TASK_2_ENABLED is False. Skipping HF API calls. Showing generated prompts only.")
    
    design_generator = None
    if TASK_2_ENABLED:
        try:
            design_generator = HFDiffusionConceptGenerator()
        except Exception as e:
            print(f"CRITICAL Task 2 Init Error: {e}. Generator will be skipped.")
            TASK_2_ENABLED = False

    for idx, row in winners.iterrows():
        p_type = row["product_type_name"]

        if "sock" in p_type.lower():
            print(f"\nINFO: Skipping {p_type} for generative design.")
            continue
            
        color = row["colour_group_name"]
        demographic = row.get("index_name", "general")
        article_id = str(row["article_id"]).zfill(10)
        
        final_prompt = _build_design_prompt(p_type, color)
        
        demo_clean = str(demographic).lower().replace(',', '').replace('/', '_').replace(' ', '_')
        color_clean = str(color).lower().replace(',', '').replace('/', '_').replace(' ', '_')
        type_clean = str(p_type).lower().replace(',', '').replace('/', '_').replace(' ', '_')
        
        stakeholder_filename = f"{demo_clean}_{color_clean}_{type_clean}"
        
        concepts_dir = Path("images/task2/concepts")
        originals_dir = Path("images/task2/originals")
        target_original_path = originals_dir / f"{article_id}.jpg"
        target_concept_path = concepts_dir / f"{stakeholder_filename}.png"
        
        print(f"\nWinner {idx+1} Concept ({p_type}):")
        print(f"  > Reference Original: {target_original_path.as_posix()}")
        print(f"  > Target Concept:     {target_concept_path.as_posix()}")
        print(f"  > Final Design Prompt: {final_prompt[:200]}...") 
        
        if TASK_2_ENABLED and design_generator:
            try:
                design_generator.generate_concept(final_prompt, _NEGATIVE_PROMPT, stakeholder_filename)
            except Exception as call_err:
                print(f"  ❌ Error generating Winner {idx+1}: {call_err}")
        elif not TASK_2_ENABLED:
            print("  > (API skipped by configuration flag)")

    print("\n--- MERCHMIX PIPELINE END ---")

if __name__ == "__main__":
    main()
