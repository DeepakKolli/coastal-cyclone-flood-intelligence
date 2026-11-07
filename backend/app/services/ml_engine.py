import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb
from app.core.config import MODELS_DIR
from app.models.schemas import ModelTrainingMetrics

MODEL_ARTIFACT_PATH = MODELS_DIR / "flood_risk_xgboost.joblib"
METRICS_PATH = MODELS_DIR / "latest_metrics.json"

FEATURE_NAMES = [
    "elevation_m",
    "slope_deg",
    "dist_to_coast_km",
    "dist_to_drainage_km",
    "rainfall_24h_mm",
    "wind_speed_kmh",
    "central_pressure_hpa",
    "storm_surge_m",
    "soil_saturation_pct"
]

RISK_LABELS = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Severe"
}

def train_flood_risk_model(df: pd.DataFrame) -> Tuple[Any, ModelTrainingMetrics]:
    """
    Trains an XGBoost multiclass classifier on geographical & meteorological features.
    """
    # Ensure all feature columns exist in df
    missing_feats = [f for f in FEATURE_NAMES if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Training dataset is missing required features: {missing_feats}")
        
    X = df[FEATURE_NAMES].fillna(df[FEATURE_NAMES].median())
    y = df["flood_risk_level"].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Configure XGBoost multiclass model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=4,
        random_state=42,
        eval_metric="mlogloss"
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    raw_pred = model.predict(X_test)
    if hasattr(raw_pred, "ndim") and raw_pred.ndim > 1 and raw_pred.shape[1] > 1:
        y_pred = np.argmax(raw_pred, axis=1)
    else:
        y_pred = raw_pred.astype(int)
        
    acc = float(accuracy_score(y_test, y_pred))
    p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
    
    # Feature importances
    importances = model.feature_importances_
    feat_imp = {feat: round(float(imp), 4) for feat, imp in zip(FEATURE_NAMES, importances)}
    
    # Sort feature importances descending
    feat_imp = dict(sorted(feat_imp.items(), key=lambda item: item[1], reverse=True))
    
    metrics = ModelTrainingMetrics(
        model_name="XGBoost Coastal Flood Risk Classifier v1.0",
        accuracy=round(acc, 4),
        precision_macro=round(float(p), 4),
        recall_macro=round(float(r), 4),
        f1_macro=round(float(f1), 4),
        feature_importances=feat_imp,
        trained_at=datetime.utcnow().isoformat() + "Z",
        sample_size=len(df)
    )
    
    # Persist model artifact
    joblib.dump(model, MODEL_ARTIFACT_PATH)
    
    return model, metrics

def load_or_init_model() -> Any:
    """
    Loads saved model from disk or trains a baseline model on the fly if not yet created.
    """
    if MODEL_ARTIFACT_PATH.exists():
        try:
            return joblib.load(MODEL_ARTIFACT_PATH)
        except Exception as e:
            print(f"Error loading model artifact: {e}, will re-train baseline.")
            
    from app.services.generator import generate_visakhapatnam_grid_dataset
    df_synthetic = generate_visakhapatnam_grid_dataset(
        rainfall_multiplier=1.2,
        base_storm_surge_m=2.0,
        cyclone_wind_kmh=130.0,
        central_pressure_hpa=965.0,
        scenario_name="Baseline Training Synthetic Scenario"
    )
    model, _ = train_flood_risk_model(df_synthetic)
    return model

def predict_grid_risk(grid_df: pd.DataFrame, model: Any = None) -> pd.DataFrame:
    """
    Performs inference over grid cells and adds prediction probabilities, categories,
    and cell-level contributing factors.
    """
    if model is None:
        model = load_or_init_model()
        
    X = grid_df[FEATURE_NAMES].fillna(grid_df[FEATURE_NAMES].median())
    
    # Predict probabilities for each class [Low, Moderate, High, Severe]
    probs = model.predict_proba(X)
    raw_classes = model.predict(X)
    if hasattr(raw_classes, "ndim") and raw_classes.ndim > 1 and raw_classes.shape[1] > 1:
        classes = np.argmax(raw_classes, axis=1)
    else:
        classes = raw_classes.astype(int)
    
    result_df = grid_df.copy()
    result_df["predicted_risk_level"] = classes
    result_df["risk_label"] = [RISK_LABELS[c] for c in classes]
    
    # Continuous risk score percentage (weighted expectation of severity)
    # weights: Low=10%, Mod=40%, High=75%, Severe=100%
    weights = np.array([0.10, 0.40, 0.75, 1.00])
    risk_scores = np.dot(probs, weights) * 100.0
    result_df["risk_score_pct"] = np.round(risk_scores, 1)
    
    # Compute relative contributing factors for each row
    factors_list = []
    for idx, row in result_df.iterrows():
        # Physical drivers: Rainfall, Low Elevation, Surge, Drainage proximity
        rain_contrib = min(40.0, (row["rainfall_24h_mm"] / 250.0) * 40.0)
        elev_contrib = max(0.0, 35.0 - (row["elevation_m"] * 1.5))
        surge_contrib = min(20.0, row["storm_surge_m"] * 6.0)
        drain_contrib = max(0.0, (2.0 - row["dist_to_drainage_km"]) * 5.0)
        
        total_driver = rain_contrib + elev_contrib + surge_contrib + drain_contrib + 1e-5
        factors_list.append({
            "Rainfall": round((rain_contrib / total_driver) * 100.0, 1),
            "Low Elevation / Slope": round((elev_contrib / total_driver) * 100.0, 1),
            "Storm Surge": round((surge_contrib / total_driver) * 100.0, 1),
            "River/Drain Proximity": round((drain_contrib / total_driver) * 100.0, 1),
        })
    result_df["contributing_factors"] = factors_list
    
    return result_df
