import pytest
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from src.skills import forecaster

def test_model_mae_on_held_out_test_slice(real_held_out_test_slice):
    """Quality Gate: Evaluates actual model predictions on committed held-out test data.
    Fails if MAE exceeds the established production threshold.
    """
    if real_held_out_test_slice is None:
        pytest.skip("Held-out test slice not found in data/processed/. Check committed files.")

    X_slice = real_held_out_test_slice["X"]
    y_true = real_held_out_test_slice["y"]

    # Run the real trained model artifact
    y_pred = forecaster._model.predict(X_slice)

    mae = mean_absolute_error(y_true, y_pred)
    # Calibrated threshold based on offline training baseline
    MAX_PERMISSIBLE_MAE = 0.45
    assert mae <= MAX_PERMISSIBLE_MAE, f"Model MAE regressed: got {mae:.2f}, limit is {MAX_PERMISSIBLE_MAE}"