# Plotly + OpenStreetMap helpers for the dashboard maps.

from typing import Dict, List, Tuple, Optional

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
    # Highway 111 pair (Palm Canyon Dr area)
    "Canyon Plaza West",
    "Jermaine Gibson",
]

# GeoJSON for each adjacent segment (A → B) along the corridor
SEGMENT_URLS: Dict[Tuple[str, str], str] = {
    # Existing segments (Avenue 52 to Hwy 111)
    ("Avenue 52",
     "Calle Tampico"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Avenue52_CalleTampico.geojson",
    ("Calle Tampico",
     "Village Shopping Ctr"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/CalleTampico_VillageShoppingctr.geojson",
    ("Village Shopping Ctr",
     "Avenue 50"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/villageshoppingctr_ave50.geojson",
    ("Avenue 50",
     "Sagebrush Ave"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Avenue50_sagebrushave.geojson",
    ("Sagebrush Ave",
     "Eisenhower Dr"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/sagebrushave_eisenhowerdr.geojson",
    ("Eisenhower Dr",
     "Avenue 48"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/eisenhowerdr_avenue48.geojson",
    ("Avenue 48",
     "Avenue 47"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/avenue48_avenue47.geojson",
    ("Avenue 47",
     "Point Happy Simon"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/avenue47_pointhappysimon.geojson",
    ("Point Happy Simon",
     "Hwy 111"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/pointhappysimon_hwy111.geojson",

    # New northbound segments (Hwy 111 to Country Club Drive)
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

    # Special case: Harris Lane segments (southbound only data available)
    ("Avenue 41",
     "Harris Lane"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/SB_HarrisLane_Avenue41.geojson",
    ("Harris Lane",
     "Country Club Drive"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/SB_CountryClubDrive_HarrisLane.geojson",

    # Highway 111: Canyon Plaza West ↔ Jermaine Gibson
    ("Canyon Plaza West",
     "Jermaine Gibson"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Segment_EB_JermaineGibson_CanyonPlazaWest.geojson",
    ("Jermaine Gibson",
     "Canyon Plaza West"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Segment_WB_CanyonPlazaWest_JermaineGibson.geojson",
}

# =========================
# Label → Node mapping (display labels used in the app → canonical corridor node)
# Add/adjust freely; the highlight logic will compare by NODE (not label).
# =========================
INTERSECTION_TO_NODE: Dict[str, str] = {
    # Existing canonical labels (Avenue 52 to Hwy 111)
    "Washington St & Avenue52": "Avenue 52",
    "Washington St & Calle Tampico": "Calle Tampico",
    "Washington St & Village Shop Ctr": "Village Shopping Ctr",
    "Washington St & Avenue50": "Avenue 50",
    "Washington St & Sagebrush Ave": "Sagebrush Ave",
    "Washington St & Eisenhower": "Eisenhower Dr",
    "Washington St & Ave48": "Avenue 48",
    "Washington St & Ave47": "Avenue 47",
    "Washington St & Point Happy Simon": "Point Happy Simon",
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

    # Common variants / aliases (existing)
    "Washington St & Avenue 52": "Avenue 52",
    "Washington St & Avenue 50": "Avenue 50",
    "Washington St & Ave 48": "Avenue 48",
    "Washington St & Ave 47": "Avenue 47",
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
    "Washington Street & Country Club Dr": "Washington St & Country Club Dr",
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


def _derive_node_coords_from_segments() -> Dict[str, Tuple[float, float]]:
    """
    Build approximate node coordinates using segment endpoints.
    For each (A,B) segment, take the first and last point of its longest polyline and assign to A and B if missing.
    """
    node_coords: Dict[str, Tuple[float, float]] = {}
    for (a, b), url in SEGMENT_URLS.items():
        gj = _fetch_geojson(url)
        if not gj:
            continue
        lines = _lines_from_geojson(gj)
        if not lines:
            continue
        line = max(lines, key=lambda l: len(l))
        start_lat, start_lon = line[0]
        end_lat, end_lon = line[-1]
        node_coords.setdefault(a, (start_lat, start_lon))
        node_coords.setdefault(b, (end_lat, end_lon))
    return node_coords


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
    "Channel Drive",
    "Miles Avenue",
    "Via Sevilla",
    "Avenue 42",
    "Harris Lane",
}
def build_corridor_map(origin: str, destination: str) -> Optional[go.Figure]:
    """
    Tab 1: Show the selected O→D corridor segment(s).
    Draws the path using your GeoJSON segments and highlights start/end.
    """
    if not origin or not destination or origin == destination:
        return None

    pairs = _segment_pairs_between(origin, destination, NODES_ORDER)
    if not pairs:
        return None

    fig = go.Figure()
    all_lats: List[float] = []
    all_lons: List[float] = []

    # Draw each segment polyline
    for pair in pairs:
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
                    name=f"{pair[0]} → {pair[1]}",
                )
            )

    if not all_lats or not all_lons:
        return None

    # Start/end markers from derived node coordinates
    node_coords = _derive_node_coords_from_segments()
    start_latlon = node_coords.get(origin)
    end_latlon = node_coords.get(destination)

    if start_latlon:
        fig.add_trace(
            go.Scattermapbox(
                lat=[start_latlon[0]],
                lon=[start_latlon[1]],
                mode="markers+text",
                marker=dict(size=13, color="#2ECC71"),
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
                marker=dict(size=13, color="#E74C3C"),
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
        height=360,
        showlegend=False,
        title=f"Corridor Segment: {origin} → {destination}",
    )
    return fig


def build_intersection_map(intersection_label: str) -> Optional[go.Figure]:
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
            marker=dict(size=14, color="#1f77b4"),
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
        height=320,
        showlegend=False,
        title=f"Intersection: {intersection_label}",
    )
    return fig


def build_intersections_overview(selected_label: Optional[str] = None) -> Optional[go.Figure]:
    """
    Tab 2/4: Show ALL intersections as dots. If 'selected_label' is provided,
    highlight the corresponding NODE (so label variants still match).
    Title reflects the selected intersection when provided.

    Special handling: when selected_label is None ("All Intersections" in UI),
    highlight only the currently available intersections (13 nodes) in red.
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
    if selected_label:
        node = _label_to_node(selected_label)
        if node:
            selected_nodes.add(node)
    else:
        # "All Intersections" case → highlight the 13 available ones
        selected_nodes = set(n for n in AVAILABLE_INTERSECTION_NODES if n in node_to_label)

    # Build point sets
    sel_lat, sel_lon, sel_text = [], [], []
    oth_lat, oth_lon, oth_text = [], [], []
    for node, label in node_to_label.items():
        lat, lon = node_coords[node]
        if node in selected_nodes:
            sel_lat.append(lat); sel_lon.append(lon); sel_text.append(label)
        else:
            oth_lat.append(lat); oth_lon.append(lon); oth_text.append(label)

    if not (sel_lat or oth_lat):
        return None

    fig = go.Figure()

    if oth_lat:
        fig.add_trace(
            go.Scattermapbox(
                lat=oth_lat,
                lon=oth_lon,
                mode="markers+text",
                marker=dict(size=11, color="#5DADE2"),
                text=oth_text,
                textposition="top right",
                hoverinfo="text",
                name="Intersections",
            )
        )

    if sel_lat:
        fig.add_trace(
            go.Scattermapbox(
                lat=sel_lat,
                lon=sel_lon,
                mode="markers+text",
                marker=dict(size=15, color="#E74C3C"),
                text=sel_text,
                textposition="top right",
                hoverinfo="text",
                name="Selected",
            )
        )

    all_lats = [coords[0] for coords in node_coords.values()]
    all_lons = [coords[1] for coords in node_coords.values()]

    title_text = "All Intersections" if not selected_label else f"Intersection: {selected_label}"

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=float(np.mean(all_lats)), lon=float(np.mean(all_lons))),
            zoom=12,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=360,
        showlegend=False,
        title=title_text,
    )
    return fig


def build_all_segments_overview() -> Optional[go.Figure]:
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
                    name=f"{pair[0]} → {pair[1]}",
                )
            )

    # Add intersection dots (as in the other maps) so users can see nodes clearly
    try:
        node_coords = _derive_node_coords_from_segments()
        if node_coords:
            node_lats = [v[0] for v in node_coords.values()]
            node_lons = [v[1] for v in node_coords.values()]
            node_text = list(node_coords.keys())
            # Also contribute to auto-centering if no polylines added
            all_lats.extend(node_lats)
            all_lons.extend(node_lons)
            fig.add_trace(
                go.Scattermapbox(
                    lat=node_lats,
                    lon=node_lons,
                    mode="markers",
                    marker=dict(size=11, color="#1F618D"),
                    text=node_text,
                    hoverinfo="text",
                    name="Intersections",
                )
            )
    except Exception:
        # Keep the overview resilient even if node derivation fails
        pass

    if not all_lats or not all_lons:
        return None

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=float(np.mean(all_lats)), lon=float(np.mean(all_lons))),
            zoom=12,
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        height=340,
        showlegend=False,
        title="Corridor Overview",
    )
    return fig