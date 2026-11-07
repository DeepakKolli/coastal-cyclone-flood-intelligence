import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "saved_models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class Settings(BaseModel):
    PROJECT_NAME: str = "Coastal Cyclone & Flood Intelligence System"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api"
    
    # Study area bounding box for Visakhapatnam region (EPSG:4326)
    # Covers coastal belt, Port, Kailasagiri, Gajuwaka, Madhurawada, Rushikonda
    VIZAG_BBOX: dict = {
        "min_lat": 17.63,
        "max_lat": 17.88,
        "min_lon": 83.12,
        "max_lon": 83.45,
    }
    
    # Grid resolution in degrees (approx 1 km ~ 0.009 deg)
    GRID_STEP_DEG: float = 0.012
    
    # Canonical field requirements
    CANONICAL_COLUMNS: list = [
        "cell_id",
        "latitude",
        "longitude",
        "elevation_m",
        "slope_deg",
        "dist_to_coast_km",
        "dist_to_drainage_km",
        "rainfall_24h_mm",
        "wind_speed_kmh",
        "central_pressure_hpa",
        "storm_surge_m",
        "soil_saturation_pct",
        "flood_risk_level"  # Ground truth or label: 0=Low, 1=Moderate, 2=High, 3=Severe
    ]

settings = Settings()
