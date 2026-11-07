from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime

class FieldMappingItem(BaseModel):
    source_column: str
    target_column: str

class SchemaMappingRequest(BaseModel):
    dataset_name: str
    mappings: List[FieldMappingItem]

class DataQualitySummary(BaseModel):
    total_records: int
    valid_records: int
    missing_values_by_field: Dict[str, int]
    duplicate_rows: int
    coordinate_out_of_bounds: int
    validation_status: str  # "PASSED", "WARNINGS", "FAILED"
    notes: List[str]

class GridCellFeature(BaseModel):
    cell_id: str
    latitude: float
    longitude: float
    elevation_m: float
    slope_deg: float
    dist_to_coast_km: float
    dist_to_drainage_km: float
    rainfall_24h_mm: float
    wind_speed_kmh: float
    central_pressure_hpa: float
    storm_surge_m: float
    soil_saturation_pct: float
    predicted_risk_level: Optional[int] = None # 0: Low, 1: Moderate, 2: High, 3: Severe
    risk_score_pct: Optional[float] = None
    risk_label: Optional[str] = None # Low, Moderate, High, Severe
    contributing_factors: Optional[Dict[str, float]] = None
    is_synthetic: bool = True

class PredictionParameters(BaseModel):
    scenario_name: Optional[str] = "Simulated Cyclone Scenario"
    rainfall_multiplier: float = Field(default=1.0, ge=0.0, le=4.0)
    storm_surge_m: float = Field(default=1.5, ge=0.0, le=7.0)
    cyclone_wind_kmh: float = Field(default=120.0, ge=20.0, le=250.0)
    central_pressure_hpa: float = Field(default=970.0, ge=890.0, le=1015.0)

class CycloneTrackPoint(BaseModel):
    timestamp: str
    latitude: float
    longitude: float
    wind_speed_kmh: float
    central_pressure_hpa: float
    category: str
    stage: str # e.g. "Deep Depression", "Cyclonic Storm", "Severe Cyclonic Storm", "Very Severe Cyclonic Storm", "Landfall"

class CycloneEvent(BaseModel):
    event_id: str
    name: str
    year: int
    peak_intensity_category: str
    landfall_location: str
    track: List[CycloneTrackPoint]
    is_historical: bool = True

class RouteNode(BaseModel):
    node_id: str
    name: str
    latitude: float
    longitude: float
    elevation_m: float
    flood_risk_score: float # 0.0 to 1.0

class RouteSegment(BaseModel):
    u: str
    v: str
    road_name: str
    distance_km: float
    risk_cost: float
    risk_category: str
    is_inundated: bool

class RouteRequest(BaseModel):
    origin_node_id: str
    destination_node_id: str
    avoid_high_risk: bool = True

class RouteResult(BaseModel):
    route_nodes: List[RouteNode]
    route_segments: List[RouteSegment]
    total_distance_km: float
    total_risk_penalty: float
    max_segment_risk: float
    risk_level: str
    safety_summary: str

class ModelTrainingMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    feature_importances: Dict[str, float]
    trained_at: str
    sample_size: int
