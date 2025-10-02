
# Plotly + OpenStreetMap helpers for the dashboard maps.

from typing import Dict, List, Tuple, Optional

import numpy as np
import requests
import plotly.graph_objects as go
import streamlit as st


# Ordered corridor nodes (south/bottom → north/top). Make sure these match your labels.
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
]

# GeoJSON for each adjacent segment (A → B) along the corridor
SEGMENT_URLS: Dict[Tuple[str, str], str] = {
    ("Avenue 52", "Calle Tampico"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Avenue52_CalleTampico.geojson",
    ("Calle Tampico", "Village Shopping Ctr"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/CalleTampico_VillageShoppingctr.geojson",
    ("Village Shopping Ctr", "Avenue 50"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/villageshoppingctr_ave50.geojson",
    ("Avenue 50", "Sagebrush Ave"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/Avenue50_sagebrushave.geojson",
    ("Sagebrush Ave", "Eisenhower Dr"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/sagebrushave_eisenhowerdr.geojson",
    ("Eisenhower Dr", "Avenue 48"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/eisenhowerdr_avenue48.geojson",
    ("Avenue 48", "Avenue 47"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/avenue48_avenue47.geojson",
    ("Avenue 47", "Point Happy Simon"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/avenue47_pointhappysimon.geojson",
    ("Point Happy Simon", "Hwy 111"): "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/Geojason/pointhappysimon_hwy111.geojson",
}

# Map your Tab 2 intersection display labels to node keys (adjust as needed).
INTERSECTION_TO_NODE: Dict[str, str] = {
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
}


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

    node_key = INTERSECTION_TO_NODE.get(intersection_label, intersection_label)
    node_coords = _derive_node_coords_from_segments()
    latlon = node_coords.get(node_key)

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
    Tab 2: Show ALL intersections as dots. If 'selected_label' is provided, that dot is highlighted.
    Title reflects the selected intersection if any.
    """
    node_coords = _derive_node_coords_from_segments()
    if not node_coords:
        return None

    all_labels = list(INTERSECTION_TO_NODE.keys())
    if not all_labels:
        return None

    points = []
    for label in all_labels:
        node_key = INTERSECTION_TO_NODE.get(label, label)
        latlon = node_coords.get(node_key)
        if latlon:
            points.append((label, latlon[0], latlon[1]))
    if not points:
        return None

    sel_lat, sel_lon, sel_text = [], [], []
    oth_lat, oth_lon, oth_text = [], [], []
    for label, lat, lon in points:
        if selected_label and label == selected_label:
            sel_lat.append(lat); sel_lon.append(lon); sel_text.append(label)
        else:
            oth_lat.append(lat); oth_lon.append(lon); oth_text.append(label)

    fig = go.Figure()

    if oth_lat:
        fig.add_trace(go.Scattermapbox(
            lat=oth_lat, lon=oth_lon,
            mode="markers+text",
            marker=dict(size=11, color="#5DADE2"),
            text=oth_text, textposition="top right",
            hoverinfo="text",
            name="Intersections",
        ))

    if sel_lat:
        fig.add_trace(go.Scattermapbox(
            lat=sel_lat, lon=sel_lon,
            mode="markers+text",
            marker=dict(size=15, color="#E74C3C"),
            text=sel_text, textposition="top right",
            hoverinfo="text",
            name="Selected",
        ))

    all_lats = [p[1] for p in points]
    all_lons = [p[2] for p in points]

    # Dynamic title: show selected label when provided
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