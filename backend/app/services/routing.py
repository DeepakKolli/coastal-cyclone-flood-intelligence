import math
import networkx as nx
from typing import List, Dict, Any, Tuple
from app.models.schemas import RouteNode, RouteSegment, RouteResult
from app.services.generator import calculate_distance_km

# Realistic road graph nodes for Visakhapatnam
VIZAG_ROAD_NODES = {
    "VZP_PORT": {"name": "Visakhapatnam Port / Old City", "lat": 17.695, "lon": 83.292, "elev": 4.2},
    "VZP_RK_BEACH": {"name": "RK Beach / Collector Office", "lat": 17.712, "lon": 83.318, "elev": 6.5},
    "VZP_RTC_COMPLEX": {"name": "Dwaraka Nagar / RTC Complex", "lat": 17.727, "lon": 83.303, "elev": 18.0},
    "VZP_SIRIPURAM": {"name": "Siripuram / AU Campus", "lat": 17.722, "lon": 83.317, "elev": 24.5},
    "VZP_MADDILAPALEM": {"name": "Maddilapalem Junction", "lat": 17.734, "lon": 83.319, "elev": 20.0},
    "VZP_MVP_COLONY": {"name": "MVP Colony Coastal Hub", "lat": 17.738, "lon": 83.336, "elev": 11.5},
    "VZP_HANUMANTHAWAKA": {"name": "Hanumanthawaka Nullah Junction", "lat": 17.755, "lon": 83.332, "elev": 14.0},
    "VZP_KAILASAGIRI_FT": {"name": "Kailasagiri Foothills", "lat": 17.750, "lon": 83.348, "elev": 42.0},
    "VZP_RUSHIKONDA": {"name": "Rushikonda Beach & IT Park", "lat": 17.785, "lon": 83.382, "elev": 14.5},
    "VZP_MADHURAWADA": {"name": "Madhurawada Center (NH16)", "lat": 17.818, "lon": 83.355, "elev": 28.0},
    "VZP_ANANDAPURAM": {"name": "Anandapuram Elevated Junction", "lat": 17.865, "lon": 83.385, "elev": 52.0},
    "VZP_NAD_JUNCTION": {"name": "NAD Kotha Road Flyover", "lat": 17.735, "lon": 83.238, "elev": 23.0},
    "VZP_AIRPORT": {"name": "Vizag Airport (Meghadrigedda)", "lat": 17.721, "lon": 83.224, "elev": 5.2},
    "VZP_SHEELANAGAR": {"name": "Sheelanagar Low-lying Link", "lat": 17.712, "lon": 83.248, "elev": 7.8},
    "VZP_GAJUWAKA": {"name": "Gajuwaka Industrial Center", "lat": 17.695, "lon": 83.208, "elev": 13.5},
    "VZP_SIMHACHALAM": {"name": "Simhachalam High-Ground Bypass", "lat": 17.765, "lon": 83.245, "elev": 68.0},
}

# Road segments (edges) connecting the nodes
VIZAG_ROAD_EDGES = [
    # Coastal Road / Beach Corridor (High vulnerability to surge and flooding)
    ("VZP_PORT", "VZP_RK_BEACH", "Port-Beach Boulevard", True),
    ("VZP_RK_BEACH", "VZP_SIRIPURAM", "Waltair Uplands Road", False),
    ("VZP_RK_BEACH", "VZP_MVP_COLONY", "Coastal Beach Road", True),
    ("VZP_MVP_COLONY", "VZP_KAILASAGIRI_FT", "Beach Road - Sagar Nagar Ext", True),
    ("VZP_KAILASAGIRI_FT", "VZP_RUSHIKONDA", "Marine Drive to Rushikonda", True),
    ("VZP_RUSHIKONDA", "VZP_MADHURAWADA", "Rushikonda-Madhurawada Hill Road", False),
    
    # Central Urban Arteries
    ("VZP_PORT", "VZP_RTC_COMPLEX", "Convent Junction Road", False),
    ("VZP_RTC_COMPLEX", "VZP_SIRIPURAM", "VIP Road", False),
    ("VZP_RTC_COMPLEX", "VZP_MADDILAPALEM", "Asilmetta - Maddilapalem Road", False),
    ("VZP_SIRIPURAM", "VZP_MADDILAPALEM", "AU Engineering College Road", False),
    ("VZP_MADDILAPALEM", "VZP_MVP_COLONY", "Isukathota Main Road", False),
    ("VZP_MADDILAPALEM", "VZP_HANUMANTHAWAKA", "NH16 Maddilapalem-Hanumanthawaka", False),
    ("VZP_MVP_COLONY", "VZP_HANUMANTHAWAKA", "Venkojipalem Nullah Link", True),
    
    # Northern Corridors
    ("VZP_HANUMANTHAWAKA", "VZP_MADHURAWADA", "NH16 Zoo Park - Madhurawada Stretch", False),
    ("VZP_MADHURAWADA", "VZP_ANANDAPURAM", "NH16 Madhurawada to Anandapuram", False),
    
    # High Ground / Inland Western Elevated Bypass Corridor (Safest during coastal surge)
    ("VZP_RTC_COMPLEX", "VZP_NAD_JUNCTION", "BRTS / National Highway Corridor", False),
    ("VZP_NAD_JUNCTION", "VZP_SIMHACHALAM", "Simhachalam Hill Approach Road", False),
    ("VZP_SIMHACHALAM", "VZP_HANUMANTHAWAKA", "Adavivaram - Hanumanthawaka Elevated Pass", False),
    ("VZP_SIMHACHALAM", "VZP_ANANDAPURAM", "Pendurthi - Anandapuram Northern Ring Road", False),
    
    # Southern Industrial & Airport links (Low-lying near Meghadrigedda)
    ("VZP_NAD_JUNCTION", "VZP_AIRPORT", "Airport Highway Link", True),
    ("VZP_AIRPORT", "VZP_SHEELANAGAR", "Meghadrigedda Basin Causeway", True),
    ("VZP_SHEELANAGAR", "VZP_GAJUWAKA", "Gajuwaka Industrial Highway", False),
    ("VZP_SHEELANAGAR", "VZP_PORT", "Port Heavy Transit Road", True),
    ("VZP_NAD_JUNCTION", "VZP_GAJUWAKA", "NAD - Gajuwaka Main Flyover Corridor", False),
]

def build_visakhapatnam_road_graph(inundation_penalty_factor: float = 8.0) -> nx.Graph:
    """
    Constructs a weighted network graph for Visakhapatnam.
    Calculates geographic distance and dynamically assigns flood vulnerability risk costs.
    """
    G = nx.Graph()
    
    # Add nodes
    for node_id, data in VIZAG_ROAD_NODES.items():
        # Node flood risk: very high if low elevation near coast or river
        elev = data["elev"]
        lat = data["lat"]
        lon = data["lon"]
        
        # Risk heuristic based on elevation
        if elev < 6.0:
            node_risk = 0.85
        elif elev < 10.0:
            node_risk = 0.60
        elif elev < 16.0:
            node_risk = 0.35
        elif elev < 30.0:
            node_risk = 0.15
        else:
            node_risk = 0.05
            
        G.add_node(
            node_id,
            name=data["name"],
            lat=lat,
            lon=lon,
            elevation_m=elev,
            flood_risk_score=node_risk
        )
        
    # Add edges
    for u, v, road_name, is_coastal_lowland in VIZAG_ROAD_EDGES:
        u_data = VIZAG_ROAD_NODES[u]
        v_data = VIZAG_ROAD_NODES[v]
        dist_km = calculate_distance_km(u_data["lat"], u_data["lon"], v_data["lat"], v_data["lon"])
        
        # Segment average elevation
        avg_elev = (u_data["elev"] + v_data["elev"]) / 2.0
        
        # Flood susceptibility of segment
        if is_coastal_lowland and avg_elev < 10.0:
            segment_risk_score = 0.82
            risk_category = "Severe Risk"
            is_inundated = True
        elif is_coastal_lowland or avg_elev < 12.0:
            segment_risk_score = 0.58
            risk_category = "High Risk"
            is_inundated = False
        elif avg_elev < 20.0:
            segment_risk_score = 0.28
            risk_category = "Moderate Risk"
            is_inundated = False
        else:
            segment_risk_score = 0.08
            risk_category = "Low Risk"
            is_inundated = False
            
        # Standard travel weight (pure distance)
        standard_weight = dist_km
        
        # Risk-penalized weight: heavily penalize segments vulnerable to flood/surge
        risk_cost = dist_km * (1.0 + (segment_risk_score ** 2) * inundation_penalty_factor)
        
        G.add_edge(
            u, v,
            road_name=road_name,
            distance_km=round(dist_km, 2),
            segment_risk_score=round(segment_risk_score, 2),
            risk_category=risk_category,
            is_inundated=is_inundated,
            standard_weight=round(standard_weight, 2),
            risk_cost=round(risk_cost, 2),
        )
        
    return G

def find_evacuation_routes(origin_id: str, dest_id: str, avoid_high_risk: bool = True) -> RouteResult:
    """
    Computes an evacuation route using A* graph search.
    If avoid_high_risk is True, paths through flooded or surge-exposed coastal corridors
    receive high cost penalties, naturally steering the route through elevated bypasses
    (e.g. Simhachalam bypass, NH16 elevated sections).
    """
    G = build_visakhapatnam_road_graph()
    
    if origin_id not in G.nodes or dest_id not in G.nodes:
        raise ValueError(f"Origin or destination node not found in network: {origin_id} -> {dest_id}")
        
    weight_key = "risk_cost" if avoid_high_risk else "standard_weight"
    
    # A* heuristic: straight-line distance in km
    def heuristic(u, v):
        u_node = G.nodes[u]
        v_node = G.nodes[v]
        return calculate_distance_km(u_node["lat"], u_node["lon"], v_node["lat"], v_node["lon"])
        
    try:
        path_nodes = nx.astar_path(G, origin_id, dest_id, heuristic=heuristic, weight=weight_key)
    except nx.NetworkXNoPath:
        raise ValueError("No viable road connection between chosen nodes.")
        
    route_nodes = []
    for nid in path_nodes:
        nd = G.nodes[nid]
        route_nodes.append(RouteNode(
            node_id=nid,
            name=nd["name"],
            latitude=nd["lat"],
            longitude=nd["lon"],
            elevation_m=nd["elevation_m"],
            flood_risk_score=nd["flood_risk_score"]
        ))
        
    route_segments = []
    total_dist = 0.0
    total_penalty = 0.0
    max_risk = 0.0
    has_inundated = False
    
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i+1]
        edge = G[u][v]
        dist = edge["distance_km"]
        rcost = edge["risk_cost"]
        rscore = edge["segment_risk_score"]
        rcat = edge["risk_category"]
        inundated = edge["is_inundated"]
        
        total_dist += dist
        total_penalty += (rcost - dist)
        if rscore > max_risk:
            max_risk = rscore
        if inundated:
            has_inundated = True
            
        route_segments.append(RouteSegment(
            u=u,
            v=v,
            road_name=edge["road_name"],
            distance_km=dist,
            risk_cost=rcost,
            risk_category=rcat,
            is_inundated=inundated
        ))
        
    if max_risk < 0.3:
        overall_risk = "Low Risk Corridor"
        summary = "Recommended evacuation path via elevated inland corridors. Low hazard of flash flooding."
    elif max_risk < 0.6:
        overall_risk = "Moderate Risk"
        summary = "Path encounters moderate drainage zones. Proceed with cautious vigilance."
    else:
        overall_risk = "High / Inundation Hazard"
        summary = "CAUTION: This route traverses coastal or low-lying drainage zones susceptible to storm surge."
        
    if avoid_high_risk and not has_inundated:
        summary += " Successfully bypassed low-lying coastal flood roads."
        
    return RouteResult(
        route_nodes=route_nodes,
        route_segments=route_segments,
        total_distance_km=round(total_dist, 2),
        total_risk_penalty=round(total_penalty, 2),
        max_segment_risk=round(max_risk, 2),
        risk_level=overall_risk,
        safety_summary=summary
    )

def get_all_road_network_data() -> Dict[str, Any]:
    """
    Returns full network graph formatted for map visualization (nodes + links with risk status).
    """
    G = build_visakhapatnam_road_graph()
    nodes = []
    for nid, d in G.nodes(data=True):
        nodes.append({
            "id": nid,
            "name": d["name"],
            "lat": d["lat"],
            "lon": d["lon"],
            "elevation_m": d["elevation_m"],
            "flood_risk_score": d["flood_risk_score"]
        })
    edges = []
    for u, v, d in G.edges(data=True):
        u_d = G.nodes[u]
        v_d = G.nodes[v]
        edges.append({
            "source": u,
            "target": v,
            "road_name": d["road_name"],
            "distance_km": d["distance_km"],
            "segment_risk_score": d["segment_risk_score"],
            "risk_category": d["risk_category"],
            "is_inundated": d["is_inundated"],
            "coordinates": [
                [u_d["lat"], u_d["lon"]],
                [v_d["lat"], v_d["lon"]]
            ]
        })
    return {"nodes": nodes, "edges": edges}
