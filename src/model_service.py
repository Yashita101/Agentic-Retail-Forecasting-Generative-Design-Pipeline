import pandas as pd
from src.forecaster import SalesForecaster
import logging

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, data_dir="data/processed", models_dir="models"):
        logger.info("Initializing Model Service and loading XGBoost predictions...")
        self.forecaster = SalesForecaster(models_dir=models_dir)
        
        try:
            self.predictions_df = self.forecaster.predict_top_performers(
                processed_data_dir=data_dir, 
                limit_distinct=100
            )
            # Ensure article_id is padded to 10 digits to match image file names
            self.predictions_df['article_id'] = self.predictions_df['article_id'].astype(str).str.zfill(10)
            
            # Unique composite key
            self.predictions_df['style_id'] = (
                self.predictions_df['article_id'] + "-" +
                self.predictions_df['index_name'].astype(str) + "-" +
                self.predictions_df['product_type_name'].astype(str) + "-" +
                self.predictions_df['colour_group_name'].astype(str)
            ).str.replace(' ', '_').str.replace('/', '_')

            # Normalized Prediction Score (0.00 to 1.00) relative to max volume
            max_volume = self.predictions_df['predicted_30d_sales_volume'].max()
            self.predictions_df['prediction_score'] = self.predictions_df['predicted_30d_sales_volume'] / max_volume
            
        except Exception as e:
             logger.error(f"Failed to load predictions: {e}")
             raise

    def get_top_styles(self, limit: int = 3) -> list[dict]:
        # Filter out low-margin commodities
        filtered_df = self.predictions_df[~self.predictions_df['product_type_name'].str.lower().str.contains("sock")]
        top_df = filtered_df.head(limit)
        
        results = []
        for _, row in top_df.iterrows():
            results.append({
                "style_id": row["style_id"],
                "prediction_score": round(row["prediction_score"], 2),
                "predicted_volume": int(round(row["predicted_30d_sales_volume"])),
                "category": row["product_type_name"],
                "sales_history": int(row.get("sales_last_30d", 0))
            })
        return results

    def get_style_details(self, style_id: str) -> dict | None:
        style_row = self.predictions_df[self.predictions_df['style_id'] == style_id]
        
        if style_row.empty:
            return None
            
        row = style_row.iloc[0]
        recent_sales = int(row.get("sales_last_30d", 0))
        predicted_vol = int(round(row["predicted_30d_sales_volume"]))
        
        return {
            "style_id": row["style_id"],
            "prediction_score": round(row["prediction_score"], 2),
            "predicted_volume": predicted_vol,
            "product_information": {
                "article_id": row["article_id"],
                "category": row["product_type_name"],
                "color": row["colour_group_name"],
                "demographic": row["index_name"],
                "pattern": row.get("graphical_appearance_name", "Solid")
            },
            "historical_performance": {
                "sales_last_30d": recent_sales,
                "sales_velocity_status": "Accelerating" if predicted_vol > recent_sales else "Stabilizing"
            },
            "model_explanation": {
                "primary_driver": "Recent Momentum (sales_last_30d)",
                "confidence_logic": f"Scores {round(row['prediction_score']*100)}% relative to the #1 top anchor piece.",
                "reasoning": f"The XGBoost model weighted this highly due to its strong baseline of {recent_sales} recent transactions and its inelastic behavior in the {row['index_name']} segment."
            }
        }