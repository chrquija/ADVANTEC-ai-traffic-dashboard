# Plotly + OpenStreetMap helpers for the dashboard maps.

from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import requests
import plotly.graph_objects as go
import streamlit as st

# =========================
# Ordered corridor nodes (south/bottom → north/top)
# =========================
NODES_ORDER: List[str] = [
    "Avenue 52",
    "Calle Tampico",
    "Village Shopping Ctr",
    "Avenue 50",
    "Sagebrush Ave",
    "Eisenhower Dr",
    "Avenue 48",
    "Avenue 47",
    "Point Happy Simon",
    "Hwy 111",
    "Channel Drive",
    "Miles Avenue",
    "Via Sevilla",
    "Fred Waring Drive",
    "Palm Royale Drive",
    "Avenue of the States",
    "Avenue 42",
    "Avenue 41",
    "Harris Lane",
    "Country Club Drive",
    "I-10 Interchange",
    "Varner Road",
    "Market Pl",
    "Del Webb",
    # Highway 111 pair (Palm Canyon Dr area)
    "Canyon Plaza West",
    "Jermaine Gibson",
    # Highway 111 (Palm Desert area)
    "Parkview Drive",
    "Cook Street",
    "Washington Street",
    "Adams St",
    "Monroe Street",
    "Indio Blvd",
    # Highway 111 - E Palm Canyon Drive
    "Canyon Plaza Drive",
    "Perez Road",
    "Auto Park Drive",
    "Bankside Drive",
    "Cathedral Canyon Drive",
    "Buddy Rogers Avenue",
    "Van Fleet Street",
    "Date Palm Drive",
    "Sun Gate Way",
    "Officer Jermain Gibson",
]

# GeoJSON for each adjacent segment (A → B) along the corridor
# Keys can be (Origin, Destination) or (Origin, Destination, Corridor)
SEGMENT_URLS: Dict[Union[Tuple[str, str], Tuple[str, str, str]], str] = {
    # Existing segments (Avenue 52 to Hwy 111)
    ("Avenue 52",
     "Calle Tampico"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Avenue52_CalleTampico.geojson",
    ("Calle Tampico",
     "Village Shopping Ctr"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/CalleTampico_VillageShoppingctr.geojson",
    ("Village Shopping Ctr",
     "Avenue 50"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/villageshoppingctr_ave50.geojson",
    ("Avenue 50",
     "Sagebrush Ave"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Avenue50_sagebrushave.geojson",
    ("Sagebrush Ave",
     "Eisenhower Dr"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/sagebrushave_eisenhowerdr.geojson",
    ("Eisenhower Dr",
     "Avenue 48"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/eisenhowerdr_avenue48.geojson",
    ("Avenue 48",
     "Avenue 47"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/avenue48_avenue47.geojson",
    ("Avenue 47",
     "Point Happy Simon"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/avenue47_pointhappysimon.geojson",
    ("Point Happy Simon",
     "Hwy 111"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/pointhappysimon_hwy111.geojson",

    # New Northbound segments (Hwy 111 to Del Webb)
    ("Hwy 111",
     "Channel Drive"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/NB_Hwy111_ChannelDrive.geojson",
    ("Channel Drive",
     "Miles Avenue"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/NB_ChannelDrive%20_MilesAvenue.geojson",
    ("Miles Avenue",
     "Via Sevilla"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/NB_MilesAvenue%20_ViaSevilla.geojson",
    ("Via Sevilla",
     "Fred Waring Drive"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/NB_ViaSevilla%20_FredWaringDrive.geojson",
    ("Fred Waring Drive",
     "Palm Royale Drive"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/NB_FredWaringDrive%20_PalmRoyaleDrive.geojson",
    ("Palm Royale Drive",
     "Avenue of the States"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/NB_PalmRoyaleDrive_AvenueoftheStates.geojson",
    ("Avenue of the States",
     "Avenue 42"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/NB_AvenueoftheStates_Avenue42.geojson",
    ("Avenue 42",
     "Avenue 41"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/NB_Avenue%2042_Avenue41.geojson",

    # Northbound Segments
    ("Avenue 41", "Harris Lane"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/119to120_NBAvenue41_HarrisLn.geojson",
    ("Harris Lane", "Country Club Drive"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/120to121_NBHarrisLn_CountryClubDr.geojson",
    ("Country Club Drive", "I-10 Interchange"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/121-122_NBCountryClubDr_I-10%20Interchange.geojson",
    ("I-10 Interchange", "Varner Road"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/122-123_NBI-10Interchange_VarnerRd.geojson",
    ("Varner Road", "Market Pl"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/123-124_NBVarnerRd_MarketPl.geojson",
    ("Market Pl", "Del Webb"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/124-125_NBMarketPl_DelWebb.geojson",

    # Southbound Segments
    ("Harris Lane", "Avenue 41"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/SB_HarrisLane_Avenue41.geojson",
    ("Country Club Drive", "Harris Lane"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/SB_CountryClubDrive_HarrisLane.geojson",
    ("I-10 Interchange", "Country Club Drive"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/122-121_SBI-10Interchange_CountryClubDrive.geojson",
    ("Varner Road", "I-10 Interchange"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/123-122_SBVarnerRd_I-10Interchange.geojson",
    ("Market Pl", "Varner Road"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/124-123_SBMarketPl_VarnerRd.geojson",
    ("Del Webb", "Market Pl"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/125-124_SBDelWebb_MarketPl.geojson",

    # Highway 111: Canyon Plaza West ↔ Jermaine Gibson
    ("Canyon Plaza West",
     "Jermaine Gibson"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Segment_EB_JermaineGibson_CanyonPlazaWest.geojson",
    ("Jermaine Gibson",
    "Canyon Plaza West"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Segment_WB_CanyonPlazaWest_JermaineGibson.geojson",

    # Highway 111: Parkview Drive ↔ Cook Street
    ("Cook Street",
     "Parkview Drive"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Hwy111_CookStreet_ParkviewDrive.geojson",
    ("Parkview Drive",
     "Cook Street"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Hwy111_ParkviewDrive_CookStreet.geojson",

    # Highway 111: Cook Street ↔ Washington Street
    ("Cook Street",
     "Washington Street"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Hwy111_CookSt_to_WashingtonSt.geojson",
    ("Washington Street",
     "Cook Street"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Hwy111_WashingtonSt_to_CookSt.geojson",

    # Highway 111: Washington Street ↔ Monroe Street
    ("Washington Street",
     "Monroe Street"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Hwy111EB_WashingtonSt_MonroeSt.geojson",
    ("Monroe Street",
     "Washington Street"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Hwy111WB_MonroeSt_WashingtonStreet.geojson",

    # Highway 111: Monroe Street ↔ Indio Blvd
    ("Monroe Street",
     "Indio Blvd"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Highway111EB_MonroeSt_IndioBlvd.geojson",
    ("Indio Blvd",
     "Monroe Street"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Highway111WB_IndioBlvd_MonroeSt.geojson",

    # Avenue 47 segments
    ("Washington Street", "Adams St", "Avenue 47"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/EB_Ave47_WashingtonSt-_AdamsSt.geojson",
    ("Adams St", "Washington Street", "Avenue 47"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/WB_Ave47_AdamsSt-_WashingtonSt.geojson",

    # Highway 111 segments
    ("Washington Street", "Adams St", "Highway 111"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/EB_hwy111_WashingtonSt_AdamsSt.geojson",
    ("Adams St", "Washington Street", "Highway 111"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/WB_hwy111_AdamsSt_WashingtonSt.geojson",

    # Adams St segments
    ("Hwy 111", "Avenue 48", "Adams St"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/NB_AdamsSt_Highway111_Avenue48.geojson",
    ("Avenue 48", "Hwy 111", "Adams St"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/NB_AdamsSt_Avenue48_Highway111.geojson",
}

# =========================
# Label → Node mapping (display labels used in the app → canonical corridor node)
# Add/adjust freely; the highlight logic will compare by NODE (not label).
# =========================
INTERSECTION_TO_NODE: Dict[str, str] = {
    # Existing canonical labels (Avenue 52 to Hwy 111)
    "Washington St & Avenue 52": "Avenue 52",
    "Washington St & Calle Tampico": "Calle Tampico",
    "Washington St & Village Shopping Center": "Village Shopping Ctr",
    "Washington St & Avenue 50": "Avenue 50",
    "Washington St & Sagebrush Avenue": "Sagebrush Ave",
    "Washington St & Eisenhower Drive": "Eisenhower Dr",
    "Washington St & Avenue 48": "Avenue 48",
    "Washington St & Avenue 47": "Avenue 47",
    "Washington St & Point Happy Simon": "Point Happy Simon",
    "Washington St & Point Happy Way": "Point Happy Simon",
    "Point Happy Way": "Point Happy Simon",
    "Avenue 52": "Avenue 52",
    "Calle Tampico": "Calle Tampico",
    "Village Shopping Center": "Village Shopping Ctr",
    "Avenue 50": "Avenue 50",
    "Sagebrush Avenue": "Sagebrush Ave",
    "Eisenhower": "Eisenhower Dr",
    "Avenue 48": "Avenue 48",
    "Avenue 47": "Avenue 47",
    "Channel Drive": "Channel Drive",
    "Miles Avenue": "Miles Avenue",
    "Via Sevilla": "Via Sevilla",
    "Avenue 42": "Avenue 42",
    "Harris Lane": "Harris Lane",
    "Country Club Drive": "Country Club Drive",
    "Varner Road": "Varner Road",
    "Washington St & Hwy 111": "Hwy 111",

    # New northern intersections (Hwy 111 to Country Club Drive)
    "Washington St & Channel Drive": "Channel Drive",
    "Washington St & Miles Avenue": "Miles Avenue",
    "Washington St & Via Sevilla": "Via Sevilla",
    "Washington St & Fred Waring Drive": "Fred Waring Drive",
    "Washington St & Palm Royale Drive": "Palm Royale Drive",
    "Washington St & Avenue of the States": "Avenue of the States",
    "Washington St & Avenue 42": "Avenue 42",
    "Washington St & Avenue 41": "Avenue 41",
    "Washington St & Harris Lane": "Harris Lane",
    "Washington St & Country Club Drive": "Country Club Drive",
    "Washington St & I-10 Interchange": "I-10 Interchange",
    "Washington St & I-10": "I-10 Interchange",
    "Washington St & Varner Road": "Varner Road",
    "Washington St & Varner": "Varner Road",
    "Washington St & Market Pl": "Market Pl",
    "Washington St & Market Place": "Market Pl",
    "Washington St & Del Webb": "Del Webb",
    "Washington St & Del Webb Blvd": "Del Webb",

    # Highway 111 intersections (Palm Desert area)
    "Hwy 111 & Parkview Drive": "Parkview Drive",
    "Highway 111 & Parkview Drive": "Parkview Drive",
    "Hwy 111 & Cook Street": "Cook Street",
    "Highway 111 & Cook Street": "Cook Street",
    "Hwy 111 & Washington Street": "Washington Street",
    "Highway 111 & Washington Street": "Washington Street",
    "Hwy 111 & Monroe Street": "Monroe Street",
    "Highway 111 & Monroe Street": "Monroe Street",
    "Hwy 111 & Indio Blvd": "Indio Blvd",
    "Highway 111 & Indio Blvd": "Indio Blvd",

    # Avenue 47 & Adams St
    "Avenue 47 & Adams St": "Adams St",
    "Highway 111 & Adams St": "Adams St",
    "Hwy 111 & Adams St": "Adams St",
    "Washington Street & Adams St": "Adams St",

    # Adams St corridor intersections
    "Adams St & Avenue 48": "Avenue 48",
    "Adams St & Ave 48": "Avenue 48",
    "Adams St & Highway 111": "Hwy 111",
    "Adams St & Hwy 111": "Hwy 111",

    # Common variants / aliases (existing)
    "Washington St & Avenue 52": "Avenue 52",
    "Washington St & Avenue 50": "Avenue 50",
    "Washington St & Avenue 48": "Avenue 48",
    "Washington St & Avenue 47": "Avenue 47",
    "Washington St & Ave 48": "Avenue 48",
    "Washington St & Ave 47": "Avenue 47",
    "Washington St & Ave 52": "Avenue 52",
    "Washington St & Ave 50": "Avenue 50",
    "Washington St & Avenue52": "Avenue 52",
    "Washington St & Avenue50": "Avenue 50",
    "Washington St & Avenue48": "Avenue 48",
    "Washington St & Avenue47": "Avenue 47",
    "Washington St & Village Shop Ctr": "Village Shopping Ctr",
    "Washington St & Sagebrush Ave": "Sagebrush Ave",
    "Washington St & Eisenhower": "Eisenhower Dr",
    "Washington St & Eisenhower Dr": "Eisenhower Dr",
    "Washington St & Village Shopping Ctr": "Village Shopping Ctr",
    "Washington St & Village Shopping Center": "Village Shopping Ctr",
    "Washington St & Point Happy Way": "Point Happy Simon",

    # Common variants for new intersections
    "Washington St & Fred Waring Dr": "Fred Waring Drive",
    "Washington St & Fred Waring": "Fred Waring Drive",
    "Washington St & Palm Royale Dr": "Palm Royale Drive",
    "Washington St & Avenue of States": "Avenue of the States",
    "Washington St & Ave of the States": "Avenue of the States",
    "Washington St & Ave 42": "Avenue 42",
    "Washington St & Ave 41": "Avenue 41",
    "Washington St & Country Club Dr": "Country Club Drive",

    # Highway 111 - E Palm Canyon Drive
    "Canyon Plaza Drive": "Canyon Plaza Drive",
    "Perez Road": "Perez Road",
    "Auto Park Drive": "Auto Park Drive",
    "Bankside Drive": "Bankside Drive",
    "Cathedral Canyon Drive": "Cathedral Canyon Drive",
    "Buddy Rogers Avenue": "Buddy Rogers Avenue",
    "Van Fleet Street": "Van Fleet Street",
    "Date Palm Drive": "Date Palm Drive",
    "Sun Gate Way": "Sun Gate Way",
    "Officer Jermain Gibson": "Officer Jermain Gibson",

    # Raw IDs for Highway 111 - E Palm Canyon Drive
    "EPalmCanyonDr_and_CanyonPlazaDr": "Canyon Plaza Drive",
    "EPalmCanyonDr_and_PerezRd": "Perez Road",
    "EPalmCanyonDr_and_AutoParkDrive": "Auto Park Drive",
    "EPalmCanyonDr_and_BanksideDrive": "Bankside Drive",
    "EPalmCanyonDr_and_CathedralCanyonDrive": "Cathedral Canyon Drive",
    "EPalmCanyonDrive_and_CathedralCanyonDrive": "Cathedral Canyon Drive",
    "EPalmCanyonDr_and_BuddyRogersAvenue": "Buddy Rogers Avenue",
    "EPalmCanyonDr_and_VanFleetStreet": "Van Fleet Street",
    "EPalmCanyonDr_and_DatePalmDrive": "Date Palm Drive",
    "EPalmCanyonDr_and_SunGateWay": "Sun Gate Way",
    "EPalmCanyonDr_and_OfficerJermainGibson": "Officer Jermain Gibson",
    "EPalmCanyonDr_and_OfficeJermainGibson": "Officer Jermain Gibson",
}

# Extra loose aliases (label normalization pass before INTERSECTION_TO_NODE)
LABEL_ALIASES: Dict[str, str] = {

    # Existing aliases
    "Washington Street & Avenue 52": "Washington St & Avenue 52",
    "Washington Street & Avenue52": "Washington St & Avenue52",
    "Washington Street & Avenue 50": "Washington St & Avenue 50",
    "Washington Street & Calle Tampico": "Washington St & Calle Tampico",
    "Washington Street & Eisenhower Dr": "Washington St & Eisenhower Dr",
    "Washington Street & Village Shop Ctr": "Washington St & Village Shop Ctr",
    "Washington Street & Village Shopping Ctr": "Washington St & Village Shopping Ctr",
    "Washington Street & Village Shopping Center": "Washington St & Village Shopping Center",
    "Washington Street & Ave 48": "Washington St & Ave 48",
    "Washington Street & Ave 47": "Washington St & Ave 47",
    "Washington Street & Hwy 111": "Washington St & Hwy 111",
    "Washington St & Village Shp Ctr": "Washington St & Village Shop Ctr",

    # New aliases for northern intersections
    "Washington Street & Channel Drive": "Washington St & Channel Drive",
    "Washington Street & Miles Avenue": "Washington St & Miles Avenue",
    "Washington Street & Via Sevilla": "Washington St & Via Sevilla",
    "Washington Street & Fred Waring Drive": "Washington St & Fred Waring Drive",
    "Washington Street & Fred Waring Dr": "Washington St & Fred Waring Dr",
    "Washington Street & Fred Waring": "Washington St & Fred Waring",
    "Washington Street & Palm Royale Drive": "Washington St & Palm Royale Drive",
    "Washington Street & Palm Royale Dr": "Washington St & Palm Royale Dr",
    "Washington Street & Avenue of the States": "Washington St & Avenue of the States",
    "Washington Street & Avenue of States": "Washington St & Avenue of States",
    "Washington Street & Ave of the States": "Washington St & Ave of the States",
    "Washington Street & Avenue 42": "Washington St & Avenue 42",
    "Washington Street & Ave 42": "Washington St & Ave 42",
    "Washington Street & Avenue 41": "Washington St & Avenue 41",
    "Washington Street & Ave 41": "Washington St & Ave 41",
    "Washington Street & Harris Lane": "Washington St & Harris Lane",
    "Washington Street & Country Club Drive": "Washington St & Country Club Drive",
    "Washington Street & Monroe Street": "Hwy 111 & Monroe Street",
    "Washington Street & Indio Blvd": "Hwy 111 & Indio Blvd",
    "Hwy 111 & Monroe St": "Monroe Street",
    "Highway 111 & Monroe St": "Monroe Street",
    "Hwy 111 & Indio Boulevard": "Indio Blvd",
    "Highway 111 & Indio Boulevard": "Indio Blvd",

    # Highway 111 - E Palm Canyon Drive
    "Highway 111 & Canyon Plaza Drive": "Canyon Plaza Drive",
    "Highway 111 & Perez Road": "Perez Road",
    "Highway 111 & Auto Park Drive": "Auto Park Drive",
    "Highway 111 & Bankside Drive": "Bankside Drive",
    "Highway 111 & Cathedral Canyon Drive": "Cathedral Canyon Drive",
    "Highway 111 & Buddy Rogers Avenue": "Buddy Rogers Avenue",
    "Highway 111 & Van Fleet Street": "Van Fleet Street",
    "Highway 111 & Date Palm Drive": "Date Palm Drive",
    "Highway 111 & Sun Gate Way": "Sun Gate Way",
    "Highway 111 & Officer Jermain Gibson": "Officer Jermain Gibson",
}


# =========================
# Utilities
# =========================
def _normalize_label(label: Optional[str]) -> Optional[str]:
    """Light normalization: collapse known aliases to a canonical display label."""
    if not label:
        return label
    s = " ".join(str(label).strip().split())
    # First pass: dictionary aliases
    if s in LABEL_ALIASES:
        s = LABEL_ALIASES[s]
    # Minor cleanup (common punctuation/spacing variants)
    s = (
        s.replace("Street", "St")
        .replace("  ", " ")
        .replace("Village Shop Ctr", "Village Shop Ctr")
    ).strip()
    return s


def _label_to_node(label: Optional[str]) -> Optional[str]:
    """
    Resolve an input label to a canonical corridor node.
    Accepts both full display labels (e.g., "Washington St & Avenue 52") and
    raw node names coming from data (e.g., "Avenue 52").
    Returns None if unknown.
    """
    if not label:
        return None
    s = _normalize_label(label)
    # If the input already matches a known corridor node, accept it directly.
    if s in NODES_ORDER:
        return s
    # Otherwise, map via display-label dictionary.
    return INTERSECTION_TO_NODE.get(s)


@st.cache_data(show_spinner=False)
def _fetch_geojson(url: str) -> Optional[dict]:
    """Fetch GeoJSON from a URL (cached)."""
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Unable to load GeoJSON: {url} ({e})")
        return None


def _segment_pairs_between(origin: str, destination: str, nodes_order: List[str]) -> List[Tuple[str, str]]:
    """Return the corridor segments (A→B pairs) between origin and destination based on the ordered node list."""
    if origin not in nodes_order or destination not in nodes_order or origin == destination:
        return []
    i0, i1 = nodes_order.index(origin), nodes_order.index(destination)
    imin, imax = (i0, i1) if i0 < i1 else (i1, i0)
    return [(nodes_order[i], nodes_order[i + 1]) for i in range(imin, imax)]


def _lines_from_geojson(gj: dict) -> List[List[Tuple[float, float]]]:
    """
    Extract line coordinate sequences from a GeoJSON (LineString or MultiLineString).
    Returns list of polyline coordinate arrays as (lat, lon) tuples.
    """
    if not gj:
        return []

    def _as_lines(geom: dict) -> List[List[Tuple[float, float]]]:
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        lines: List[List[Tuple[float, float]]] = []
        if gtype == "LineString":
            # coordinates are [lon, lat]
            lines.append([(float(y[1]), float(y[0])) for y in coords if isinstance(y, (list, tuple)) and len(y) >= 2])
        elif gtype == "MultiLineString":
            for part in coords:
                lines.append([(float(y[1]), float(y[0])) for y in part if isinstance(y, (list, tuple)) and len(y) >= 2])
        return lines

    out: List[List[Tuple[float, float]]] = []
    if gj.get("type") == "FeatureCollection":
        for feat in gj.get("features", []):
            geom = feat.get("geometry", {})
            out.extend(_as_lines(geom))
    elif gj.get("type") in ("LineString", "MultiLineString"):
        out.extend(_as_lines(gj))
    return [line for line in out if len(line) >= 2]


# =========================
# Explicit Coordinates for intersections not covered by GeoJSON segments
# =========================
EXPLICIT_NODE_COORDS: Dict[str, Tuple[float, float]] = {
    "Canyon Plaza Drive": (33.78739, -116.48206),
    "Perez Road": (33.78535, -116.47582),
    "Auto Park Drive": (33.78370, -116.47361),
    "Bankside Drive": (33.78092, -116.46996),
    "Cathedral Canyon Drive": (33.77985, -116.46608),
    "Buddy Rogers Avenue": (33.77974, -116.46443),
    "Van Fleet Street": (33.77915, -116.46248),
    "Date Palm Drive": (33.77725, -116.45721),
    "Sun Gate Way": (33.77651, -116.45453),
    "Officer Jermain Gibson": (33.77683, -116.45246),
    "Point Happy Simon": (33.71273, -116.29186),
}


def _derive_node_coords_from_segments() -> Dict[str, Tuple[float, float]]:
    """
    Build approximate node coordinates using segment endpoints and explicit overrides.
    """
    node_coords: Dict[str, Tuple[float, float]] = EXPLICIT_NODE_COORDS.copy()
    for key, url in SEGMENT_URLS.items():
        corridor = None
        if len(key) == 3:
            a, b, corridor = key
        else:
            a, b = key
        gj = _fetch_geojson(url)
        if not gj:
            continue
        lines = _lines_from_geojson(gj)
        if not lines:
            continue
        line = max(lines, key=lambda l: len(l))
        start_lat, start_lon = line[0]
        end_lat, end_lon = line[-1]
        if corridor:
            node_coords.setdefault(f"{a} ({corridor})", (start_lat, start_lon))
            node_coords.setdefault(f"{b} ({corridor})", (end_lat, end_lon))
        else:
            node_coords.setdefault(a, (start_lat, start_lon))
            node_coords.setdefault(b, (end_lat, end_lon))
    return node_coords


def get_node_coordinates() -> Dict[str, Tuple[float, float]]:
    """
    Public accessor for node coordinates (cached/derived).
    Returns dict: { 'Node Name': (lat, lon) }
    """
    return _derive_node_coords_from_segments()


# =========================
# Builders
# =========================
# Nodes for which we currently have volume/availability in Pg.2 (13 locations)
AVAILABLE_INTERSECTION_NODES = {
    "Avenue 52",
    "Calle Tampico",
    "Village Shopping Ctr",
    "Avenue 50",
    "Sagebrush Ave",
    "Eisenhower Dr",
    "Avenue 48",
    "Avenue 47",
    "Point Happy Simon",
    "Channel Drive",
    "Miles Avenue",
    "Via Sevilla",
    "Avenue 42",
    "Harris Lane",
    "Country Club Drive",
    "I-10 Interchange",
    "Varner Road",
    "Market Pl",
    "Del Webb",
    # Highway 111 (Palm Desert area)
    "Parkview Drive",
    "Cook Street",
    "Washington Street",
    "Adams St",
    "Monroe Street",
    "Indio Blvd",
    # Highway 111 - E Palm Canyon Drive
    "Canyon Plaza Drive",
    "Perez Road",
    "Auto Park Drive",
    "Bankside Drive",
    "Cathedral Canyon Drive",
    "Buddy Rogers Avenue",
    "Van Fleet Street",
    "Date Palm Drive",
    "Sun Gate Way",
    "Officer Jermain Gibson",
}


def build_corridor_map(origin: str, destination: str, satellite: bool = False) -> Optional[go.Figure]:
    """
    Tab 1: Show the selected O→D corridor segment(s).
    Draws the path using your GeoJSON segments and highlights start/end.
    """
    if not origin or not destination or origin == destination:
        return None

    corridor = st.session_state.get("t1_corridor_selector")

    # 1. Check for a direct corridor-specific segment first (useful for non-adjacent nodes)
    if corridor and ((origin, destination, corridor) in SEGMENT_URLS or (destination, origin, corridor) in SEGMENT_URLS):
        pairs = [(origin, destination)]
    else:
        # 2. Fallback to breaking the path into segments based on canonical NODES_ORDER
        pairs = _segment_pairs_between(origin, destination, NODES_ORDER)

    if not pairs:
        return None

    fig = go.Figure()
    all_lats: List[float] = []
    all_lons: List[float] = []

    # Draw each segment polyline
    for pair in pairs:
        url = None
        if corridor:
            # Try corridor-specific segment first
            url = SEGMENT_URLS.get((pair[0], pair[1], corridor)) or SEGMENT_URLS.get((pair[1], pair[0], corridor))
        
        if not url:
            # Fallback to general segment
            url = SEGMENT_URLS.get(pair) or SEGMENT_URLS.get((pair[1], pair[0]))  # fallback to reversed if needed
            
        if not url:
            st.info(f"No GeoJSON registered for segment {pair[0]} → {pair[1]}")
            continue

        gj = _fetch_geojson(url)
        if not gj:
            continue

        lines = _lines_from_geojson(gj)
        for line in lines:
            lats = [p[0] for p in line]
            lons = [p[1] for p in line]
            all_lats.extend(lats)
            all_lons.extend(lons)
            fig.add_trace(
                go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode="lines",
                    line=dict(width=5, color="#1f77b4"),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{pair[0]} → {pair[1]}",
                )
            )

    if not all_lats or not all_lons:
        return None

    # Start/end markers from derived node coordinates
    node_coords = _derive_node_coords_from_segments()
    
    # Try corridor-specific markers first
    start_latlon = node_coords.get(f"{origin} ({corridor})") or node_coords.get(origin)
    end_latlon = node_coords.get(f"{destination} ({corridor})") or node_coords.get(destination)

    if start_latlon:
        fig.add_trace(
            go.Scattermapbox(
                lat=[start_latlon[0]],
                lon=[start_latlon[1]],
                mode="markers+text",
                marker=dict(size=20, color="#2ECC71"),
                text=[f"Start: {origin}"],
                textposition="top right",
                showlegend=False,
                hoverinfo="text",
                name="Start",
            )
        )
    if end_latlon:
        fig.add_trace(
            go.Scattermapbox(
                lat=[end_latlon[0]],
                lon=[end_latlon[1]],
                mode="markers+text",
                marker=dict(size=20, color="#E74C3C"),
                text=[f"End: {destination}"],
                textposition="top right",
                showlegend=False,
                hoverinfo="text",
                name="End",
            )
        )

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=float(np.mean(all_lats)), lon=float(np.mean(all_lons))),
            zoom=12,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=600,
        showlegend=True,
        title=f"Corridor Segment: {origin} → {destination}",
    )

    # Satellite toggle logic
    if satellite:
        try:
            token = st.secrets.get("mapbox_token", "")
        except Exception:
            token = ""
        if token:
            fig.update_layout(mapbox_style="satellite-streets", mapbox_accesstoken=token)
        else:
            # Fallback to free Esri Satellite tiles if no Mapbox token
            fig.update_layout(
                mapbox_style="white-bg",
                mapbox_layers=[
                    {
                        "below": 'traces',
                        "sourcetype": "raster",
                        "sourceattribution": "Esri, Maxar, Earthstar Geographics",
                        "source": [
                            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        ]
                    }
                ]
            )

    # Legend
    fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode="lines", line=dict(width=4, color="#1f77b4"), showlegend=True, name="Corridor Path"))
    fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode="markers", marker=dict(size=10, color="#2ECC71"), showlegend=True, name="Start"))
    fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode="markers", marker=dict(size=10, color="#E74C3C"), showlegend=True, name="End"))
    fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode="markers", marker=dict(size=10, color="#5DADE2"), showlegend=True, name="Intersection"))

    fig.update_layout(
        legend=dict(
            title=dict(text="Legend"),
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=12),
        )
    )
    return fig


def build_intersection_map(intersection_label: str, satellite: bool = False) -> Optional[go.Figure]:
    """
    Tab 2: Show a dot for the selected intersection.
    Resolves display label -> corridor node, derives lat/lon from segments, and marks it.
    """
    if not intersection_label:
        return None

    node_key = _label_to_node(intersection_label)
    node_coords = _derive_node_coords_from_segments()
    latlon = node_coords.get(node_key) if node_key else None

    if not latlon:
        st.info(f"Location for '{intersection_label}' is not known yet. Update INTERSECTION_TO_NODE or segment data.")
        return None

    lat, lon = latlon
    fig = go.Figure()
    fig.add_trace(
        go.Scattermapbox(
            lat=[lat],
            lon=[lon],
            mode="markers+text",
            marker=dict(size=25, color="#E74C3C"),
            text=[intersection_label],
            textposition="top right",
            hoverinfo="text",
            name=intersection_label,
        )
    )
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat, lon=lon),
            zoom=14,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=600,
        showlegend=True,
        title=f"Intersection: {intersection_label}",
    )

    # Satellite toggle logic
    if satellite:
        try:
            token = st.secrets.get("mapbox_token", "")
        except Exception:
            token = ""
        if token:
            fig.update_layout(mapbox_style="satellite-streets", mapbox_accesstoken=token)
        else:
            # Fallback to free Esri Satellite tiles if no Mapbox token
            fig.update_layout(
                mapbox_style="white-bg",
                mapbox_layers=[
                    {
                        "below": 'traces',
                        "sourcetype": "raster",
                        "sourceattribution": "Esri, Maxar, Earthstar Geographics",
                        "source": [
                            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        ]
                    }
                ]
            )

    # Legend
    fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode="markers", marker=dict(size=10, color="#5DADE2"), showlegend=True, name="Intersection"))

    fig.update_layout(
        legend=dict(
            title=dict(text="Legend"),
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=12),
        )
    )
    return fig


def build_intersections_overview(selected_label: Optional[Union[str, List[str]]] = None, corridor: Optional[str] = None,
                                 tooltip_map: Optional[Dict[str, str]] = None, satellite: bool = False) -> Optional[go.Figure]:
    """
    Tab 2/4: Show ALL intersections as dots. If 'selected_label' is provided,
    highlight the corresponding NODE (so label variants still match).
    Title reflects the selected intersection when provided.

    Special handling: when selected_label is None ("All Intersections" in UI),
    highlight intersections based on the selected corridor.

    tooltip_map: Optional dict mapping 'Intersection Name' -> 'Custom HTML Tooltip'
    """
    node_coords = _derive_node_coords_from_segments()
    if not node_coords:
        return None

    # Collapse to a single display label per node to avoid duplicates on the map
    node_to_label: Dict[str, str] = {}
    for lbl, node in INTERSECTION_TO_NODE.items():
        if node in node_coords and node not in node_to_label:
            node_to_label[node] = lbl

    if not node_to_label:
        return None

    # Determine which nodes should be highlighted (red)
    selected_nodes: set = set()
    if selected_label and selected_label != "All Intersections":
        if isinstance(selected_label, list):
            for lbl in selected_label:
                node = _label_to_node(lbl)
                if node:
                    selected_nodes.add(node)
        else:
            node = _label_to_node(selected_label)
            if node:
                selected_nodes.add(node)
    else:
        # "All Intersections" case → highlight based on corridor
        if corridor == "Washington Street":
            highlight_list = [
                "Avenue 52", "Calle Tampico", "Village Shopping Ctr", "Avenue 50",
                "Sagebrush Ave", "Eisenhower Dr", "Avenue 48", "Avenue 47", "Point Happy Simon",
                "Channel Drive", "Miles Avenue", "Via Sevilla", "Avenue 42", "Harris Lane"
            ]
        elif corridor == "Highway 111":
            highlight_list = [
                "Parkview Drive", "Cook Street", "Washington Street", "Monroe Street", "Indio Blvd"
            ]
        elif corridor == "Highway 111 - E Palm Canyon Drive":
            highlight_list = [
                "Canyon Plaza Drive", "Perez Road", "Auto Park Drive", "Bankside Drive",
                "Cathedral Canyon Drive", "Buddy Rogers Avenue", "Van Fleet Street",
                "Date Palm Drive", "Sun Gate Way", "Officer Jermain Gibson"
            ]
        else:
            # Fallback to all available intersections
            highlight_list = list(AVAILABLE_INTERSECTION_NODES)

        selected_nodes = set(n for n in highlight_list if n in node_to_label)

    # Build point sets
    sel_lat, sel_lon, sel_text, sel_hover = [], [], [], []
    oth_lat, oth_lon, oth_text, oth_hover = [], [], [], []
    for node, label in node_to_label.items():
        lat, lon = node_coords[node]

        # 1. Map Label: Always use the clean name
        map_label = label

        # 2. Tooltip: Check if a custom string exists in the map (by Label or Node ID)
        tooltip = label
        if tooltip_map:
            tooltip = tooltip_map.get(label, tooltip_map.get(node, label))

        if node in selected_nodes:
            sel_lat.append(lat);
            sel_lon.append(lon);
            sel_text.append(map_label);
            sel_hover.append(tooltip)
        else:
            oth_lat.append(lat);
            oth_lon.append(lon);
            oth_text.append(map_label);
            oth_hover.append(tooltip)

    if not (sel_lat or oth_lat):
        return None

    fig = go.Figure()

    if oth_lat:
        fig.add_trace(
            go.Scattermapbox(
                lat=oth_lat,
                lon=oth_lon,
                mode="markers+text",
                marker=dict(size=15, color="#5DADE2"),
                text=oth_text,
                hovertext=oth_hover,  # Use separate hover text
                textposition="top right",
                hoverinfo="text",  # Tells Plotly to use hovertext property
                name="Intersections",
                showlegend=False,
                hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
            )
        )

    if sel_lat:
        fig.add_trace(
            go.Scattermapbox(
                lat=sel_lat,
                lon=sel_lon,
                mode="markers+text",
                marker=dict(size=25, color="#E74C3C"),
                text=sel_text,
                hovertext=sel_hover,  # Use separate hover text
                textposition="top right",
                hoverinfo="text",  # Tells Plotly to use hovertext property
                name="Selected",
                showlegend=False,
                hoverlabel=dict(bgcolor="white", font_size=14, font_family="Arial"),
            )
        )

    # Determine map center: focus on selected/highlighted nodes if available
    if sel_lat and sel_lon:
        center_lat = float(np.mean(sel_lat))
        center_lon = float(np.mean(sel_lon))
        # Dynamic zoom: tighter if single node, wider if corridor
        zoom_level = 14 if len(sel_lat) == 1 else 12.5
    else:
        # Fallback to center of all known nodes
        all_lats = [coords[0] for coords in node_coords.values()]
        all_lons = [coords[1] for coords in node_coords.values()]
        center_lat = float(np.mean(all_lats))
        center_lon = float(np.mean(all_lons))
        zoom_level = 11

    title_text = "All Intersections" if not selected_label else f"Intersection: {selected_label}"

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom_level,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=600,
        showlegend=True,
        title=title_text,
    )

    # Satellite toggle logic
    if satellite:
        try:
            token = st.secrets.get("mapbox_token", "")
        except Exception:
            token = ""
        if token:
            fig.update_layout(mapbox_style="satellite-streets", mapbox_accesstoken=token)
        else:
            # Fallback to free Esri Satellite tiles if no Mapbox token
            fig.update_layout(
                mapbox_style="white-bg",
                mapbox_layers=[
                    {
                        "below": 'traces',
                        "sourcetype": "raster",
                        "sourceattribution": "Esri, Maxar, Earthstar Geographics",
                        "source": [
                            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        ]
                    }
                ]
            )

    # Legend
    fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode="markers", marker=dict(size=10, color="#5DADE2"), showlegend=True, name="Intersection"))
    fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode="markers", marker=dict(size=10, color="#E74C3C"), showlegend=True, name="Selected"))

    fig.update_layout(
        legend=dict(
            title=dict(text="Legend"),
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=12),
        )
    )
    return fig


def build_all_segments_overview(satellite: bool = False) -> Optional[go.Figure]:
    """
    Optional: Overview map drawing all corridor segments (thin polylines).
    Useful as a context map.
    """
    fig = go.Figure()
    all_lats: List[float] = []
    all_lons: List[float] = []

    for pair, url in SEGMENT_URLS.items():
        gj = _fetch_geojson(url)
        if not gj:
            continue
        lines = _lines_from_geojson(gj)
        for line in lines:
            lats = [p[0] for p in line]
            lons = [p[1] for p in line]
            all_lats.extend(lats)
            all_lons.extend(lons)
            fig.add_trace(
                go.Scattermapbox(
                    lat=lats,
                    lon=lons,
                    mode="lines",
                    line=dict(width=3, color="#5DADE2"),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{pair[0]} → {pair[1]}",
                )
            )

    if not all_lats or not all_lons:
        return None

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=float(np.mean(all_lats)), lon=float(np.mean(all_lons))),
            zoom=12,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=600,
        showlegend=True,
        title="Corridor Overview",
    )

    # Satellite toggle logic
    if satellite:
        try:
            token = st.secrets.get("mapbox_token", "")
        except Exception:
            token = ""
        if token:
            fig.update_layout(mapbox_style="satellite-streets", mapbox_accesstoken=token)
        else:
            # Fallback to free Esri Satellite tiles if no Mapbox token
            fig.update_layout(
                mapbox_style="white-bg",
                mapbox_layers=[
                    {
                        "below": 'traces',
                        "sourcetype": "raster",
                        "sourceattribution": "Esri, Maxar, Earthstar Geographics",
                        "source": [
                            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
                        ]
                    }
                ]
            )

    # Legend
    fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode="lines", line=dict(width=4, color="#5DADE2"), showlegend=True, name="Corridor Segments"))
    fig.add_trace(go.Scattermapbox(lat=[None], lon=[None], mode="markers", marker=dict(size=10, color="#5DADE2"), showlegend=True, name="Intersection"))

    fig.update_layout(
        legend=dict(
            title=dict(text="Legend"),
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=1,
            font=dict(size=12),
        )
    )
    return fig
