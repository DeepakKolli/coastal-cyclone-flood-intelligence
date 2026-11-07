import math
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from app.core.config import settings

def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Haversine formula
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Key geographic reference landmarks in Visakhapatnam
VIZAG_LANDMARKS = [
    {"name": "Dolphin's Nose / Yarada Ridge", "lat": 17.675, "lon": 83.295, "elev": 290.0, "type": "hill"},
    {"name": "Simhachalam Hills", "lat": 17.768, "lon": 83.250, "elev": 310.0, "type": "hill"},
    {"name": "Kailasagiri Ridge", "lat": 17.747, "lon": 83.342, "elev": 175.0, "type": "hill"},
    {"name": "Kambalakonda Forest Hills", "lat": 17.810, "lon": 83.330, "elev": 240.0, "type": "hill"},
    {"name": "Visakhapatnam Port / Old City", "lat": 17.695, "lon": 83.290, "elev": 4.5, "type": "coastal_lowland"},
    {"name": "RK Beach / Coastal Road", "lat": 17.712, "lon": 83.318, "elev": 6.0, "type": "coastal"},
    {"name": "Lawson's Bay / MVP Colony", "lat": 17.735, "lon": 83.338, "elev": 8.0, "type": "urban_coastal"},
    {"name": "Rushikonda Beach", "lat": 17.781, "lon": 83.385, "elev": 7.0, "type": "coastal"},
    {"name": "Madhurawada Valley", "lat": 17.820, "lon": 83.360, "elev": 22.0, "type": "valley"},
    {"name": "Meghadrigedda Basin / Airport", "lat": 17.721, "lon": 83.225, "elev": 5.0, "type": "drainage_basin"},
    {"name": "Gajuwaka Industrial Lowlands", "lat": 17.698, "lon": 83.210, "elev": 12.0, "type": "lowland"},
    {"name": "Sheelanagar Low-lying Area", "lat": 17.715, "lon": 83.245, "elev": 6.5, "type": "drainage_basin"},
]

# Approximate coastline vertices for Visakhapatnam coastal arc
COASTLINE_POINTS = [
    (17.63, 83.24),
    (17.67, 83.29),
    (17.71, 83.32),
    (17.74, 83.34),
    (17.78, 83.39),
    (17.83, 83.42),
    (17.88, 83.44),
]

# Major drainage lines (Meghadrigedda, Hanumanthawaka, Pedda Gedda)
DRAINAGE_POINTS = [
    (17.72, 83.22), # Meghadrigedda reservoir outlet
    (17.73, 83.25),
    (17.70, 83.28), # Harbor inlet
    (17.75, 83.33), # Hanumanthawaka Gedda
    (17.80, 83.36), # Madhurawada Nullah
]

def get_approx_distance_to_coast(lat: float, lon: float) -> float:
    return min(calculate_distance_km(lat, lon, clat, clon) for clat, clon in COASTLINE_POINTS)

def get_approx_distance_to_drainage(lat: float, lon: float) -> float:
    return min(calculate_distance_km(lat, lon, dlat, dlon) for dlat, dlon in DRAINAGE_POINTS)

def estimate_terrain_elevation_slope(lat: float, lon: float) -> Tuple[float, float]:
    """
    Computes realistic elevation and slope in degrees based on proximity to
    known hill ridges (Kailasagiri, Simhachalam, Yarada, Kambalakonda) and lowlands.
    """
    base_elev = 14.0
    # Add influence from hill centers using Gaussian attenuation
    hill_elev_sum = 0.0
    for lm in VIZAG_LANDMARKS:
        d = calculate_distance_km(lat, lon, lm["lat"], lm["lon"])
        if lm["type"] == "hill":
            # Hills attenuate over 2.5 - 3.5 km
            hill_elev_sum += lm["elev"] * math.exp(- (d ** 2) / (2 * (1.8 ** 2)))
        elif "lowland" in lm["type"] or "coastal" in lm["type"]:
            # Pull down near coastal / port lowlands
            if d < 1.5:
                base_elev = min(base_elev, lm["elev"] + d * 3.0)

    elevation = max(2.0, base_elev + hill_elev_sum)
    
    # Derive plausible slope from elevation gradient
    if elevation < 10:
        slope = max(0.5, 0.8 + 0.3 * (elevation / 10.0))
    elif elevation < 40:
        slope = 2.0 + (elevation - 10) * 0.15
    else:
        slope = min(32.0, 6.0 + (elevation - 40) * 0.12)
        
    return round(elevation, 1), round(slope, 1)

def generate_visakhapatnam_grid_dataset(
    rainfall_multiplier: float = 1.0,
    base_storm_surge_m: float = 1.5,
    cyclone_wind_kmh: float = 120.0,
    central_pressure_hpa: float = 970.0,
    scenario_name: str = "Simulated Cyclone Scenario",
    add_noise: bool = True
) -> pd.DataFrame:
    """
    Generates a structured, physically consistent spatial dataset across the Visakhapatnam grid.
    Encodes hydro-meteorological laws:
    - High rain + low elevation + low slope + high surge -> High/Severe Flood Risk
    - High elevation (Kailasagiri, Simhachalam) -> Low Flood Risk (High Runoff)
    """
    np.random.seed(42)
    records = []
    
    lats = np.arange(settings.VIZAG_BBOX["min_lat"], settings.VIZAG_BBOX["max_lat"], settings.GRID_STEP_DEG)
    lons = np.arange(settings.VIZAG_BBOX["min_lon"], settings.VIZAG_BBOX["max_lon"], settings.GRID_STEP_DEG)
    
    cell_idx = 1
    for lat in lats:
        for lon in lons:
            cell_id = f"VZ_GRID_{cell_idx:04d}"
            cell_idx += 1
            
            dist_coast = get_approx_distance_to_coast(lat, lon)
            dist_drainage = get_approx_distance_to_drainage(lat, lon)
            elev_m, slope_deg = estimate_terrain_elevation_slope(lat, lon)
            
            # Spatial rainfall variation (higher near coast and orographic hill lift)
            noise_factor = np.random.normal(0, 8.0) if add_noise else 0.0
            base_rain = 180.0 * rainfall_multiplier
            orographic_boost = min(60.0, elev_m * 0.18)
            rainfall_24h = max(10.0, round(base_rain + orographic_boost - (dist_coast * 1.8) + noise_factor, 1))
            
            # Wind speed slightly decreases inland
            wind_speed = max(30.0, round(cyclone_wind_kmh - (dist_coast * 1.5) + np.random.normal(0, 3.0), 1))
            
            # Surge affects strictly coastal lowlands
            if dist_coast < 3.5 and elev_m < 8.0:
                effective_surge = max(0.0, round(base_storm_surge_m * (1.0 - dist_coast / 3.5), 2))
            else:
                effective_surge = 0.0
                
            # Soil saturation %
            soil_sat = min(99.0, max(25.0, round(50.0 + (rainfall_24h / 5.0) - (slope_deg * 0.8), 1)))
            
            # Physics-based hydrological index to assign grounded risk level
            # 1. Inundation depth susceptibility index
            # Low elevation (< 10m) + Flat (< 2 deg) + High rain + Surge + Close to drainage
            susceptibility = 0.0
            
            # Rainfall factor (0 - 40 pts)
            susceptibility += min(40.0, (rainfall_24h / 300.0) * 40.0)
            
            # Elevation penalty/shield (0 - 35 pts)
            if elev_m < 5.0:
                susceptibility += 35.0
            elif elev_m < 12.0:
                susceptibility += 25.0 - (elev_m - 5.0) * 2.0
            elif elev_m < 25.0:
                susceptibility += 10.0 - (elev_m - 12.0) * 0.6
            else:
                susceptibility += 0.0  # Safe from pooling
                
            # Surge impact (0 - 15 pts)
            susceptibility += min(15.0, effective_surge * 4.5)
            
            # Drainage proximity (0 - 10 pts)
            if dist_drainage < 1.0:
                susceptibility += 10.0 * (1.0 - dist_drainage)
                
            # Slope reduction: steep terrain drains quickly
            if slope_deg > 8.0:
                susceptibility -= min(20.0, (slope_deg - 8.0) * 1.2)
                
            susceptibility = max(0.0, min(100.0, susceptibility))
            
            # Map susceptibility into 4 discrete classes
            # 0: Low (< 30)
            # 1: Moderate (30 - 55)
            # 2: High (55 - 75)
            # 3: Severe (> 75)
            if susceptibility < 32.0:
                flood_risk_level = 0
            elif susceptibility < 56.0:
                flood_risk_level = 1
            elif susceptibility < 76.0:
                flood_risk_level = 2
            else:
                flood_risk_level = 3
                
            records.append({
                "cell_id": cell_id,
                "latitude": round(lat, 4),
                "longitude": round(lon, 4),
                "elevation_m": elev_m,
                "slope_deg": slope_deg,
                "dist_to_coast_km": round(dist_coast, 2),
                "dist_to_drainage_km": round(dist_drainage, 2),
                "rainfall_24h_mm": rainfall_24h,
                "wind_speed_kmh": wind_speed,
                "central_pressure_hpa": central_pressure_hpa,
                "storm_surge_m": effective_surge,
                "soil_saturation_pct": soil_sat,
                "susceptibility_score": round(susceptibility, 1),
                "flood_risk_level": flood_risk_level,
                "scenario_name": scenario_name,
                "is_synthetic": True,
            })
            
    df = pd.DataFrame(records)
    return df

def get_predefined_cyclone_events() -> List[Dict[str, Any]]:
    """
    Curated realistic cyclone events impacting Visakhapatnam coastal region:
    - Cyclone Hudhud (2014) - Category 4 Very Severe Cyclonic Storm (Direct Hit on Vizag)
    - Cyclone Michaung (2023) - Severe Cyclonic Storm causing severe coastal inundation
    - Cyclone Gulab (2021) - Cyclonic storm with high riverine backwater surge
    - Cyclone Simulated 2026 - High-intensity projected test scenario
    """
    return [
        {
            "event_id": "CYC-2014-HUDHUD",
            "name": "Cyclone Hudhud (Historic)",
            "year": 2014,
            "peak_intensity_category": "Extremely Severe Cyclonic Storm",
            "landfall_location": "Visakhapatnam (Kailasagiri - Port Corridor)",
            "is_historical": True,
            "track": [
                {"timestamp": "2014-10-10 06:00", "latitude": 15.2, "longitude": 86.8, "wind_speed_kmh": 120, "central_pressure_hpa": 980, "category": "Severe Cyclonic Storm", "stage": "Bay of Bengal Deep Water"},
                {"timestamp": "2014-10-11 00:00", "latitude": 16.1, "longitude": 85.5, "wind_speed_kmh": 165, "central_pressure_hpa": 962, "category": "Very Severe Cyclonic Storm", "stage": "Approaching AP Coast"},
                {"timestamp": "2014-10-11 18:00", "latitude": 17.0, "longitude": 84.2, "wind_speed_kmh": 195, "central_pressure_hpa": 950, "category": "Extremely Severe Cyclonic Storm", "stage": "Offshore Visakhapatnam"},
                {"timestamp": "2014-10-12 06:30", "latitude": 17.72, "longitude": 83.31, "wind_speed_kmh": 215, "central_pressure_hpa": 940, "category": "Extremely Severe Cyclonic Storm", "stage": "Landfall at Visakhapatnam"},
                {"timestamp": "2014-10-12 18:00", "latitude": 18.2, "longitude": 82.8, "wind_speed_kmh": 110, "central_pressure_hpa": 978, "category": "Cyclonic Storm", "stage": "Inland Dissipation (Araku)"},
            ]
        },
        {
            "event_id": "CYC-2023-MICHAUNG",
            "name": "Cyclone Michaung (Historic)",
            "year": 2023,
            "peak_intensity_category": "Severe Cyclonic Storm",
            "landfall_location": "Bapatla-Machilipatnam with massive coastal surge in Vizag",
            "is_historical": True,
            "track": [
                {"timestamp": "2023-12-03 12:00", "latitude": 13.5, "longitude": 82.0, "wind_speed_kmh": 90, "central_pressure_hpa": 992, "category": "Cyclonic Storm", "stage": "South-West Bay"},
                {"timestamp": "2023-12-04 12:00", "latitude": 14.8, "longitude": 80.8, "wind_speed_kmh": 110, "central_pressure_hpa": 984, "category": "Severe Cyclonic Storm", "stage": "Coastal Transit"},
                {"timestamp": "2023-12-05 08:00", "latitude": 16.3, "longitude": 81.2, "wind_speed_kmh": 105, "central_pressure_hpa": 986, "category": "Severe Cyclonic Storm", "stage": "Landfall Near Bapatla"},
                {"timestamp": "2023-12-05 20:00", "latitude": 17.4, "longitude": 82.6, "wind_speed_kmh": 75, "central_pressure_hpa": 994, "category": "Deep Depression", "stage": "Heavy Inundation in Visakhapatnam"},
            ]
        },
        {
            "event_id": "CYC-2026-SIMULATED",
            "name": "Projected Cyclone Scenario 2026",
            "year": 2026,
            "peak_intensity_category": "Very Severe Cyclonic Storm",
            "landfall_location": "Bheemunipatnam - Visakhapatnam Coastal Arc",
            "is_historical": False,
            "track": [
                {"timestamp": "2026-11-06 00:00", "latitude": 15.8, "longitude": 85.9, "wind_speed_kmh": 115, "central_pressure_hpa": 982, "category": "Severe Cyclonic Storm", "stage": "Tracking North-West"},
                {"timestamp": "2026-11-06 18:00", "latitude": 16.9, "longitude": 84.6, "wind_speed_kmh": 150, "central_pressure_hpa": 966, "category": "Very Severe Cyclonic Storm", "stage": "Approaching Vizag Coast"},
                {"timestamp": "2026-11-07 06:00", "latitude": 17.75, "longitude": 83.35, "wind_speed_kmh": 165, "central_pressure_hpa": 958, "category": "Very Severe Cyclonic Storm", "stage": "Projected Landfall"},
                {"timestamp": "2026-11-07 18:00", "latitude": 18.15, "longitude": 82.95, "wind_speed_kmh": 95, "central_pressure_hpa": 988, "category": "Cyclonic Storm", "stage": "Weakening Inland"},
            ]
        }
    ]
