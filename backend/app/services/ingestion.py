import os
import json
import uuid
import pandas as pd
from typing import Dict, List, Any, Tuple
from app.core.config import settings, DATA_DIR
from app.models.schemas import DataQualitySummary

# Canonical mapping synonyms for automated detection
SYNONYM_MAP = {
    "latitude": ["lat", "latitude", "y", "northing"],
    "longitude": ["lon", "long", "longitude", "x", "easting"],
    "elevation_m": ["elev", "elevation", "elevation_m", "height", "dem", "altitude"],
    "slope_deg": ["slope", "slope_deg", "gradient", "incline"],
    "dist_to_coast_km": ["dist_coast", "coast_dist", "distance_to_coast", "dist_to_coast_km"],
    "dist_to_drainage_km": ["dist_drainage", "river_dist", "distance_to_river", "dist_to_drainage_km"],
    "rainfall_24h_mm": ["rain", "rainfall", "rainfall_mm", "precipitation", "precip_24h", "rainfall_24h_mm"],
    "wind_speed_kmh": ["wind", "wind_speed", "wind_speed_kmh", "gust_speed"],
    "central_pressure_hpa": ["pressure", "central_pressure", "pressure_hpa", "central_pressure_hpa", "mslp"],
    "storm_surge_m": ["surge", "storm_surge", "surge_height", "storm_surge_m"],
    "soil_saturation_pct": ["soil_sat", "saturation", "soil_moisture", "soil_saturation_pct"],
    "flood_risk_level": ["label", "flood", "flood_level", "risk_level", "flood_risk_level", "target"],
}

def auto_detect_column_mappings(columns: List[str]) -> Dict[str, str]:
    """
    Attempts to match incoming column headers with canonical schema fields.
    """
    detected = {}
    lower_cols = {col.lower().strip(): col for col in columns}
    
    for canonical_name, synonyms in SYNONYM_MAP.items():
        for syn in synonyms:
            if syn in lower_cols:
                detected[canonical_name] = lower_cols[syn]
                break
    return detected

def validate_and_profile_dataframe(df: pd.DataFrame, mappings: Dict[str, str] = None) -> Tuple[pd.DataFrame, DataQualitySummary]:
    """
    Standardizes column names using mappings, profiles quality, missing values, duplicates, and range boundaries.
    """
    notes = []
    
    # Apply column mappings if provided
    working_df = df.copy()
    if mappings:
        rename_dict = {orig: target for target, orig in mappings.items() if orig in working_df.columns}
        working_df.rename(columns=rename_dict, inplace=True)
    
    total_records = len(working_df)
    duplicates = int(working_df.duplicated().sum())
    
    missing_by_field = {}
    for col in working_df.columns:
        null_count = int(working_df[col].isnull().sum())
        if null_count > 0:
            missing_by_field[col] = null_count
            
    # Check spatial bounding coordinates if present
    out_of_bounds = 0
    if "latitude" in working_df.columns and "longitude" in working_df.columns:
        # Check against Visakhapatnam regional bounding box with slight buffer
        lat_valid = working_df["latitude"].between(settings.VIZAG_BBOX["min_lat"] - 0.5, settings.VIZAG_BBOX["max_lat"] + 0.5)
        lon_valid = working_df["longitude"].between(settings.VIZAG_BBOX["min_lon"] - 0.5, settings.VIZAG_BBOX["max_lon"] + 0.5)
        out_of_bounds = int((~ (lat_valid & lon_valid)).sum())
        if out_of_bounds > 0:
            notes.append(f"{out_of_bounds} records lie outside the regional Visakhapatnam study area.")

    # Check for negative physical values
    if "rainfall_24h_mm" in working_df.columns:
        neg_rain = int((working_df["rainfall_24h_mm"] < 0).sum())
        if neg_rain > 0:
            notes.append(f"{neg_rain} rows contain negative rainfall values (flagged for zero-clipping).")
            working_df["rainfall_24h_mm"] = working_df["rainfall_24h_mm"].clip(lower=0)

    # Determine validation status
    if total_records == 0:
        status = "FAILED"
        notes.append("Dataset is empty.")
    elif duplicates > 0.5 * total_records:
        status = "FAILED"
        notes.append("High duplication detected (>50% duplicate rows).")
    elif len(missing_by_field) > 0 or out_of_bounds > 0:
        status = "WARNINGS"
        notes.append("Non-blocking data quality warnings detected.")
    else:
        status = "PASSED"
        notes.append("Dataset passed all canonical validation checks.")
        
    summary = DataQualitySummary(
        total_records=total_records,
        valid_records=total_records - duplicates,
        missing_values_by_field=missing_by_field,
        duplicate_rows=duplicates,
        coordinate_out_of_bounds=out_of_bounds,
        validation_status=status,
        notes=notes
    )
    
    return working_df, summary

def save_uploaded_dataset(file_content: bytes, filename: str) -> Dict[str, Any]:
    dataset_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(filename)[1].lower()
    
    saved_filename = f"dataset_{dataset_id}{ext}"
    saved_path = DATA_DIR / saved_filename
    
    with open(saved_path, "wb") as f:
        f.write(file_content)
        
    # Read sample to detect columns
    if ext in [".csv", ".txt"]:
        df = pd.read_csv(saved_path, nrows=50)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(saved_path, nrows=50)
    elif ext in [".geojson", ".json"]:
        with open(saved_path, "r", encoding="utf-8") as jf:
            geo_data = json.load(jf)
        # Flatten features if geojson
        if "features" in geo_data:
            props = [f.get("properties", {}) for f in geo_data["features"]]
            df = pd.DataFrame(props)
        else:
            df = pd.DataFrame(geo_data)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        
    detected_cols = list(df.columns)
    auto_mappings = auto_detect_column_mappings(detected_cols)
    
    return {
        "dataset_id": dataset_id,
        "filename": filename,
        "format": ext.replace(".", "").upper(),
        "columns": detected_cols,
        "auto_mappings": auto_mappings,
        "preview_records": df.head(5).to_dict(orient="records"),
        "total_columns": len(detected_cols)
    }
