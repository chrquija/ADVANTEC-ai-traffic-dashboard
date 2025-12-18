# Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Plotly for chart helpers
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _fix_raw_url(url: str) -> str:
    """
    GitHub RAW URLs must be:
      https://raw.githubusercontent.com/<user>/<repo>/<branch>/<path>
    Some of your links used '/refs/heads/main/'. This converts them.
    """
    return url.replace("/refs/heads/", "/")


def _safe_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _normalize_acyclica_headers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tolerant column normalizer for Acyclica CSVs.
    """
    if df is None or df.empty:
        return df
    # Lower + strip spaces to catch variants, then map back to canon names
    lowmap = {c: "".join(str(c).strip().split()).lower() for c in df.columns}
    df = df.rename(columns=lowmap)
    canon = {
        "localdatetime": "local_datetime",
        "corridorid": "corridor_id",
        "direction": "direction",
        "metric": "metric",
        "strength": "Strength",
        "firsts": "Firsts",
        "lasts": "Lasts",
        "minimum": "Minimum",
        "maximum": "Maximum",
        # include segment identifier if present
        "segmentid": "segment_id",
        "segment_id": "segment_id",
    }
    for src, tgt in canon.items():
        if src in df.columns:
            df = df.rename(columns={src: tgt})
    return df


# =========================
# Data loading
# =========================

# Iteris ClearGuide Data

@st.cache_data(show_spinner=False)
def load_traffic_data():
    """
    Load and combine all corridor traffic data from GitHub (Iteris-style).
    Optimized for memory and speed:
    - Fixes bad RAW URL pattern
    - Reads only necessary columns
    - Parses datetime on read
    - Downcasts numeric columns and categorizes strings
    """
    data_sources = {
        # Existing segments (Avenue 52 to Highway 111)
        "Avenue 52 → Calle Tampico": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/1_2_LONG_NSB_Ave52_CalleTampico_WashSt_1hr_septojuly.csv",
        "Calle Tampico → Village Shopping Ctr": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/2_3_LONG_NSB_CalleTampico_VillageShoppingCtr_WashSt_1hr_septojuly.csv",
        "Village Shopping Ctr → Avenue 50": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/3_4_LONG_NSB_VillageShoppingCtr_Avenue50_WashSt_1hr_septojuly.csv",
        "Avenue 50 → Sagebrush Ave": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/4_5_LONG_NSB_Ave50_SagebrushAve_WashSt_1hr_septojuly.csv",
        "Sagebrush Ave → Eisenhower Dr": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/5_6_LONG_NSB_SagebrushAve_EisenhowerDr_WashSt_1hr_septojuly.csv",
        "Eisenhower Dr → Avenue 48": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/6_7_LONG_NSB_EisenhowerDr_Avenue48_WashSt_1hr_septojuly.csv",
        "Avenue 48 → Avenue 47": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/7_8_LONG_NSB_Ave48_Ave47_WashSt_1hr_septojuly.csv",
        "Avenue 47 → Point Happy Simon": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/8_9_LONG_NSB_Ave47_PointHappySimon_WashSt_1hr_septojuly.csv",
        "Point Happy Simon → Hwy 111": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/9_10_LONG_NSB_PointHappySimon_WashSt_1hr_septojuly.csv",

        # New segments extending north from Highway 111
        "Hwy 111 → Channel Drive": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/10_11_LONG_NSB_Hwy111_to_ChannelDrive.csv",
        "Channel Drive → Miles Avenue": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/11_12_LONG_NSB_ChannelDrive_to_MilesAvenue.csv",
        "Miles Avenue → Via Sevilla": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/12_13_LONG_NSB_MilesAvenue_to_ViaSevilla.csv",
        "Via Sevilla → Fred Waring Drive": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/13_14_LONG_NSB_ViaSevilla_FredWaringDrive.csv",
        "Fred Waring Drive → Palm Royale Drive": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/14_15_LONG_NSB_FredWaringDrive_to_PalmRoyaleDrive.csv",
        "Palm Royale Drive → Avenue of the States": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/15_16_LONG_NSB_PalmRoyaleDrive_to_AvenueoftheStates.csv",
        "Avenue of the States → Avenue 42": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/16_17_LONG_NSB_AvenueoftheStates_to_Avenue42.csv",
        "Avenue 42 → Avenue 41": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/17_18_LONG_NSB_Avenue42_to_Avenue41.csv",
        "Avenue 41 → Country Club Drive": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/18_20_LONG_NB_Avenue41_to_Countryclubdrive.csv",

        # Southbound only segments
        "Harris Lane → Avenue 41 (SB)": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/19_18_LONG_SB_Harrislane_avenue41.csv",
        "Country Club Drive → Harris Lane (SB)": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/20_19_LONG_SB_CountryClubDrive_to_HarrisLane.csv",
    }

    usecols = [
        "local_datetime",
        "corridor_id",
        "direction",
        "average_delay",
        "average_traveltime",
        "average_speed",
    ]

    all_data = []
    for segment_name, url in data_sources.items():
        url = _fix_raw_url(url)
        try:
            df = pd.read_csv(
                url,
                usecols=lambda c: c in usecols,  # tolerate column mismatches
                parse_dates=["local_datetime"],
                dtype={
                    "corridor_id": "string",
                    "direction": "string",
                },
            )
            if df.empty:
                continue
            # Ensure needed numeric columns exist even if missing in source
            for c in ["average_delay", "average_traveltime", "average_speed"]:
                if c not in df.columns:
                    df[c] = np.nan
            # Assign segment name, reduce memory
            df["segment_name"] = segment_name
            for c in ("direction", "segment_name"):
                if c in df.columns:
                    df[c] = df[c].astype("category")
            for c in ("average_delay", "average_traveltime", "average_speed"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
            all_data.append(df)
        except Exception as e:
            st.error(f"Error loading {segment_name}: {e}")

    if not all_data:
        return pd.DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.dropna(subset=["local_datetime"]).sort_values("local_datetime").reset_index(drop=True)
    return combined_df

#Kinetic mobility data
@st.cache_data(show_spinner=False)
def load_volume_data():
    """
    Load consolidated volume data for Pg.2 Kinetic Mobility from unified CSV.
    Uses new file with added corridor_id and EB/WB directions.
    Ensures presence of: local_datetime, corridor_id, intersection_name, direction, total_volume.
    """
    volume_url = _fix_raw_url(
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/VOLUME/KMOB_LONG/FULL_AvailableVolumeCounts_WashingtonCorridor.csv"
    )

    def _norm_dir(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip().str.upper()
        map_dir = {
            "N": "NB", "NB": "NB", "NORTH": "NB", "NORTHBOUND": "NB",
            "S": "SB", "SB": "SB", "SOUTH": "SB", "SOUTHBOUND": "SB",
            "E": "EB", "EB": "EB", "EAST": "EB", "EASTBOUND": "EB",
            "W": "WB", "WB": "WB", "WEST": "WB", "WESTBOUND": "WB",
        }
        return s.map(map_dir).fillna(s)

    def _friendly_label(raw: str) -> str:
        if not isinstance(raw, str):
            return ""
        r = raw
        # Explicit mapping for the 13 intersections (south → north), plus tolerated aliases

        explicit = {
            # Southbound to northbound official list
            "Washington_St_and_Avenue52": "Avenue 52",
            "Washington_St_and_CalleTampico": "Calle Tampico",
            "Washington_St_and_Calle_Tampico": "Calle Tampico",  # Add this for underscore version in case it doesn't work.
            "Washington_St_and_Village_Shopping_Center": "Village Shopping Center",
            # Add this line for the abbreviated version
            "Washington_St_and_Avenue50": "Avenue 50",
            "Washington_St_and_Sagebrush_Avenue": "Sagebrush Avenue",
            "Washington_St_and_Eisenhower": "Eisenhower",
            "Washington_St_and_Avenue48": "Avenue 48",
            "Washington_St_and_Avenue47": "Avenue 47",
            "Washington_St_and_ChannelDrive": "Channel Drive",
            "Washington_St_and_MilesAvenue": "Miles Avenue",
            "Washington_St_and_ViaSevilla": "Via Sevilla",
            "Washington_St_to_Avenue42": "Avenue 42",
            "Washington_St_and_HarrisLane": "Harris Lane",
            # Aliases sometimes seen in data
            "Washington_St_and_Channel_Drive": "Channel Drive",
            "Washington_St_and_MilesAve": "Miles Avenue",
        }
        if r in explicit:
            return explicit[r]
        # Fallback normalization for any unexpected IDs: trim corridor prefix and tidy underscores
        r2 = (r.replace("_", " ")
                .replace("Washington St and ", "")
                .replace("Washington_St_and_", "")
                .replace("Washington and ", "")
                .replace("Washington Street and ", "")
                .replace("Washington St & ", "")
                .replace("Washington Street & ", ""))
        # Clean double spaces
        r2 = " ".join(r2.split())
        # Avoid aggressive expansions that caused typos; only minor spacing fixes
        # Insert space between 'Avenue' and trailing digits (e.g., 'Avenue47' → 'Avenue 47')
        try:
            import re
            r2 = re.sub(r"(Avenue)(\d)", r"\\1 \\2", r2)
        except Exception:
            pass
        return r2.strip()

    try:
        df = pd.read_csv(volume_url)
        # Ensure datetime
        if "local_datetime" in df.columns:
            df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")
        elif "datetime" in df.columns:
            df = df.rename(columns={"datetime": "local_datetime"})
            df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")
        else:
            raise RuntimeError("Missing local_datetime column in volume CSV")
        df = df.dropna(subset=["local_datetime"]).sort_values("local_datetime").reset_index(drop=True)

        # Corridor id (string)
        if "corridor_id" not in df.columns:
            df["corridor_id"] = "Washington_Street"
        df["corridor_id"] = df["corridor_id"].astype("string")

        # Direction normalization
        if "direction" in df.columns:
            df["direction"] = _norm_dir(df["direction"]).astype("category")
        else:
            df["direction"] = "NB"

        # Intersection key/name extraction
        # Prefer explicit id columns
        inter_key_col = None
        for c in ["intersection_id", "intersection", "intersection_key", "segment_id"]:
            if c in df.columns:
                inter_key_col = c
                break
        if inter_key_col is None and "intersection_name" in df.columns:
            inter_key_col = "intersection_name"
        if inter_key_col is None:
            raise RuntimeError("Missing intersection identifier column in volume CSV")

        # Build friendly intersection_name
        df["intersection_name"] = df[inter_key_col].astype(str).apply(_friendly_label)

        # Ensure numeric volume column
        vol_col = "total_volume" if "total_volume" in df.columns else ("volume" if "volume" in df.columns else None)
        if vol_col is None:
            raise RuntimeError("Missing total_volume/volume column in volume CSV")
        df["total_volume"] = pd.to_numeric(df[vol_col], errors="coerce")

        # Memory optimizations
        for c in ("intersection_name",):
            df[c] = df[c].astype("string")
        for c in ("corridor_id", "direction"):
            if c in df.columns:
                df[c] = df[c].astype("category")

        return df
    except Exception as e:
        st.error(f"Error loading volume data: {e}")
        return pd.DataFrame()


# -------------------------
# Acyclica (Long + Wide)
# -------------------------
# Support multiple Acyclica sources and combine them
ACYCLICA_URLS = [
    _fix_raw_url(
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/MASTER_Acyclica_Traveltime_speed.csv"
    ),
    # New master with extended Washington Street segments (Ave 52 ↔ Country Club Dr)
    _fix_raw_url(
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/MASTER_1hr_Acyclica_Traveltime_speed_Ave52toCountryClubDrive.csv"
    ),
    # Highway 111: Canyon Plaza West ↔ Jermaine Gibson (EB/WB combined long-format)
    _fix_raw_url(
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/MASTER_EWB_1hr_PalmCanyon_CanyonPlazaWest_to_JermainGibson.csv"
    ),
    # Highway 111: Cook Street ↔ Parkview Drive (EB/WB)
    _fix_raw_url(
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/refs/heads/main/Highway_111_DATA/ACYCLICA/MASTER_EWB_1hr_Hwy111_CookStreet_to_Parkview.csv"
    ),
]

@st.cache_data(show_spinner=False)
def load_acyclica_data() -> pd.DataFrame:
    """
    Load Acyclica travel time & speed in LONG format from one or more CSVs.
    Returns a cleaned DataFrame or raises RuntimeError on failure.
    Columns (normalized):
      local_datetime, corridor_id, direction, metric, Strength, Firsts, Lasts, Minimum, Maximum
    """
    frames: list[pd.DataFrame] = []
    errors = []
    for url in ACYCLICA_URLS:
        try:
            frames.append(pd.read_csv(url))
        except Exception as e:
            errors.append(str(e))
    if not frames:
        # Raise so the caller decides where/how to display the error (main area, not sidebar)
        raise RuntimeError(f"Error loading Acyclica data: {' | '.join(errors) if errors else 'no sources available'}")

    df = pd.concat(frames, ignore_index=True)

    if df is None or df.empty:
        raise RuntimeError("Acyclica CSV is empty.")

    # Normalize headers
    df = _normalize_acyclica_headers(df)

    required = ["local_datetime","corridor_id","direction","metric","Strength","Firsts","Lasts","Minimum","Maximum"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Acyclica CSV missing required columns: {', '.join(missing)}")

    # Types & cleanup
    df["local_datetime"] = pd.to_datetime(df["local_datetime"], errors="coerce")
    df = df.dropna(subset=["local_datetime"])  # keep only valid times
    for c in ["Strength","Firsts","Lasts","Minimum","Maximum"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Metric normalization: ensure exactly {TravelTime, Speed}
    m = df["metric"].astype(str).str.lower().str.strip()
    m = m.str.replace("_", " ", regex=False)
    m = m.str.replace("-", " ", regex=False)
    m = m.str.replace("traveltime", "travel time", regex=False)
    m = m.str.replace("travel_time", "travel time", regex=False)
    m = m.str.replace("traveltime(min)", "travel time", regex=False)
    df["metric"] = np.where(m.str.contains("travel") & ~m.str.contains("speed"), "TravelTime", "Speed")

    # Direction normalization
    dir_raw = df["direction"].astype(str).str.strip().str.upper()
    # Map common variants to canonical set
    def _norm_dir(x: str) -> str:
        if x in {"E", "EB", "EAST", "EASTBOUND", "E/B", "EAST-BOUND"}:
            return "EASTBOUND"
        if x in {"W", "WB", "WEST", "WESTBOUND", "W/B", "WEST-BOUND"}:
            return "WESTBOUND"
        if x in {"N", "NB", "NORTH", "NORTHBOUND", "N/B", "NORTH-BOUND"}:
            return "NB"
        if x in {"S", "SB", "SOUTH", "SOUTHBOUND", "S/B", "SOUTH-BOUND"}:
            return "SB"
        return x
    df["direction"] = dir_raw.map(_norm_dir)

    # Corridor normalization: fix known variants/typos
    # Ensure new Highway 111 data appears as "Highway 111" in dropdowns
    corr = (
        df["corridor_id"].astype(str).str.strip().replace({
            # Common variants
            "HIghway111": "Highway 111",
            "Highway111": "Highway 111",
            "Hwy 111": "Highway 111",
            # Additional likely variants
            "HIghway 111": "Highway 111",
            "highway 111": "Highway 111",
            "Highway-111": "Highway 111",
            "Highway_111": "Highway 111",
            "Hwy111": "Highway 111",
        })
    )
    # Pattern-based safety net: collapse spaces/dashes/underscores for matching
    key = corr.str.lower().str.replace("[ _-]", "", regex=True)
    corr = np.where(key == "highway111", "Highway 111", corr)
    df["corridor_id"] = corr

    df = df.sort_values(["local_datetime","direction","metric"]).reset_index(drop=True)
    return df



@st.cache_data(show_spinner=False)
def acyclica_long_to_hourly(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert long → wide for KPI/plots.
    Output columns:
      local_datetime, corridor_id, direction, average_traveltime, average_speed, average_delay (NaN), segment_name, segment_id (if available)
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame()

    # Include segment_id in index if present to preserve O→D segment selection
    index_cols = ["local_datetime", "corridor_id", "direction"]
    if "segment_id" in df_long.columns:
        index_cols.append("segment_id")

    piv = (
        df_long.pivot_table(
            index=index_cols,
            columns="metric",
            values="Strength",
            aggfunc="mean",
        )
        .reset_index()
        .rename(columns={"TravelTime":"average_traveltime","Speed":"average_speed"})
    )

    # Ensure presence
    for col in ["average_traveltime","average_speed"]:
        if col not in piv.columns:
            piv[col] = np.nan

    piv["average_delay"] = np.nan  # Acyclica doesn't provide delay
    # Prefer segment_id as segment_name if available, else corridor_id
    if "segment_id" in piv.columns:
        piv["segment_name"] = piv["segment_id"].astype(str)
    else:
        piv["segment_name"] = piv["corridor_id"].astype(str)

    piv["local_datetime"] = pd.to_datetime(piv["local_datetime"], errors="coerce")
    sort_cols = ["local_datetime", "direction"] + (["segment_id"] if "segment_id" in piv.columns else [])
    piv = piv.dropna(subset=["local_datetime"]).sort_values(sort_cols).reset_index(drop=True)
    return piv


# =========================
# Small data getters
# =========================
@st.cache_data(show_spinner=False)
def get_corridor_df() -> pd.DataFrame:
    df = load_traffic_data()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = _safe_to_datetime(df.copy(), "local_datetime")
    needed = {"segment_name", "average_delay", "average_traveltime", "average_speed", "direction"}
    missing = needed - set(df.columns)
    if missing:
        st.warning(f"Traffic dataset is missing columns: {', '.join(missing)}")
    return df


@st.cache_data(show_spinner=False)
def get_volume_df() -> pd.DataFrame:
    df = load_volume_data()
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = _safe_to_datetime(df.copy(), "local_datetime")
    needed = {"intersection_name", "total_volume", "direction"}
    missing = needed - set(df.columns)
    if missing:
        st.warning(f"Volume dataset is missing columns: {', '.join(missing)}")
    return df


@st.cache_data(show_spinner=False)
def get_acyclica_long_df() -> pd.DataFrame:
    """
    Long-format Acyclica (for Incident/Peak/Event detection).
    """
    return load_acyclica_data()


@st.cache_data(show_spinner=False)
def get_acyclica_df() -> pd.DataFrame:
    """
    Wide-format Acyclica for KPI/plots (Iteris-like):
      local_datetime, corridor_id, direction, average_traveltime, average_speed, average_delay, segment_name
    """
    long_df = load_acyclica_data()
    if long_df is None or len(long_df) == 0:
        return pd.DataFrame()
    wide = acyclica_long_to_hourly(long_df)
    return _safe_to_datetime(wide.copy(), "local_datetime")


def get_performance_rating(score: float):
    """
    Map a 0..100 score to a label + CSS class used by the UI badges.
    """
    if score > 80:
        return " Excellent", "badge-excellent"
    if score > 60:
        return " Good", "badge-good"
    if score > 40:
        return " Fair", "badge-fair"
    if score > 20:
        return " Poor", "badge-poor"
    return " Critical", "badge-critical"


# =========================
# Interpretable KPI helpers (for Performance tab)
# =========================
def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def compute_perf_kpis_interpretable(df: pd.DataFrame, high_delay_threshold: float) -> dict:
    """
    Compute five interpretable KPIs for Iteris-style (wide) data.
    """
    if df is None or df.empty:
        return {
            "avg_tt": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Average Travel Time"},
            "planning_time": {"value": 0.0, "unit": "min", "score": 50.0, "help": "Planning Time (95th)"},
            "buffer_index": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Buffer Index"},
            "reliability": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Reliability Index"},
            "congestion_freq": {"value": 0.0, "unit": "%", "score": 50.0, "help": "Congestion Frequency"},
        }

    # Coerce numeric
    for c in ("average_delay", "average_traveltime", "average_speed"):
        if c in df:
            df[c] = _coerce_num(df[c])

    # Average TT
    avg_tt = float(np.nanmean(df["average_traveltime"])) if "average_traveltime" in df else 0.0

    # Planning time (P95)
    if "average_traveltime" in df and df["average_traveltime"].notna().any():
        p95_tt = float(np.nanpercentile(df["average_traveltime"].dropna(), 95))
    else:
        p95_tt = 0.0

    # Buffer Index
    buffer_index = ((p95_tt - avg_tt) / avg_tt * 100.0) if avg_tt > 0 else 0.0

    # Reliability Index = 100 - CV%
    if avg_tt > 0 and "average_traveltime" in df:
        cv_tt = float(np.nanstd(df["average_traveltime"])) / avg_tt * 100.0
    else:
        cv_tt = 0.0
    reliability = max(0.0, 100.0 - cv_tt)

    # Congestion Frequency (% of hours with delay > threshold)
    if "average_delay" in df and df["average_delay"].notna().any():
        total_hours = int(df["average_delay"].count())
        cong_hours = int((df["average_delay"] > high_delay_threshold).sum())
        cong_freq = (cong_hours / total_hours * 100.0) if total_hours > 0 else 0.0
    else:
        cong_freq, cong_hours, total_hours = 0.0, 0, 0

    # Normalized scores (0..100, higher = better)
    def _minmax_score(series: pd.Series, val: float) -> float:
        series = pd.to_numeric(series, errors="coerce").dropna()
        if len(series) < 2:
            return 50.0
        mn, mx = float(series.min()), float(series.max())
        if mx <= mn:
            return 50.0
        frac = (val - mn) / (mx - mn)  # lower is better
        return float(max(0.0, min(100.0, 100.0 * (1.0 - frac))))

    if "average_traveltime" in df and df["average_traveltime"].notna().any():
        score_avg_tt = _minmax_score(df["average_traveltime"], avg_tt)
        score_plan = _minmax_score(df["average_traveltime"], p95_tt)
    else:
        score_avg_tt = score_plan = 50.0

    score_buffer = float(max(0.0, 100.0 - min(max(buffer_index, 0.0), 100.0)))
    score_reliability = float(max(0.0, min(100.0, reliability)))
    score_congestion = float(max(0.0, min(100.0, 100.0 - cong_freq)))

    return {
        "avg_tt": {
            "value": avg_tt,
            "unit": "min",
            "score": score_avg_tt,
            "help": "Average Travel Time\n\nWhat it means: The typical door-to-door trip time for this route with your current filters.\nWhy it exists: Gives a quick sense of what most trips take.\nHow it’s calculated: Average of the hourly O-D trip times.\nFormula: mean(travel_time).",
        },
        "planning_time": {
            "value": p95_tt,
            "unit": "min",
            "score": score_plan,
            "help": "Planning Time (95th)\n\nWhat it means: 95th-percentile travel time in your filtered period.\nPurpose: captures a realistic worst-case for planning.",
        },
        "buffer_index": {
            "value": buffer_index,
            "unit": "%",
            "score": score_buffer,
            "help": "Buffer Index = (P95 − mean) / mean × 100.",
        },
        "reliability": {
            "value": reliability,
            "unit": "%",
            "score": score_reliability,
            "help": "Reliability Index = 100 − CV%, where CV% = stdev/mean × 100.",
        },
        "congestion_freq": {
            "value": cong_freq,
            "unit": "%",
            "score": score_congestion,
            "extra": f"Hours > {high_delay_threshold:.0f}s: {cong_hours}/{total_hours}",
            "help": "Share of hours with delay above your chosen threshold.",
        },
    }


def render_badge(score: float) -> str:
    """
    Turn a 0..100 'goodness' score into your visual badge HTML, using get_performance_rating.
    """
    label, css = get_performance_rating(score)
    return f'<span class="performance-badge {css}">{label}</span>'


# =========================
# Chart helpers
# =========================
def performance_chart(data: pd.DataFrame, metric_type: str = "delay"):
    if data.empty:
        return None
    metric_type = metric_type.lower().strip()
    if metric_type == "delay":
        y_col, title, color = "average_delay", "Traffic Delay Analysis", "#e74c3c"
        y_label = "Average Delay (seconds)"
        dist_x_label = "Average Delay (seconds)"
    else:
        y_col, title, color = "average_traveltime", "Travel Time Analysis", "#3498db"
        y_label = "Average Travel Time (minutes)"
        dist_x_label = "Average Travel Time (minutes)"

    dd = data.dropna(subset=["local_datetime", y_col]).sort_values("local_datetime")

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Time Series Analysis", "Distribution Analysis"),
        # Increase spacing between the top chart and the Distribution subtitle to prevent overlap
        vertical_spacing=0.25,
    )

    # Time series plot
    fig.add_trace(
        go.Scatter(
            x=dd["local_datetime"],
            y=dd[y_col],
            mode="lines+markers",
            name=f"{metric_type.title()} Trend",
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ),
        row=1, col=1,
    )

    # Shade missing-data gaps on the time-series panel
    try:
        times = pd.to_datetime(dd["local_datetime"]).sort_values().reset_index(drop=True)
        if len(times) >= 3:
            deltas = times.diff().dropna()
            # Use a robust expected interval: median delta
            med = deltas.median()
            if pd.notna(med) and med > pd.Timedelta(0):
                gap_threshold = med * 1.5
                # Iterate pairs to find large gaps
                gap_spans = []
                for i in range(1, len(times)):
                    dt = times[i] - times[i - 1]
                    if dt > gap_threshold:
                        gap_spans.append((times[i - 1], times[i]))
                for start, end in gap_spans:
                    # Add a light grey band for the gap region
                    fig.add_vrect(
                        x0=start,
                        x1=end,
                        fillcolor="#95a5a6",
                        opacity=0.18,
                        line_width=0,
                        layer="below",
                        row=1,
                        col=1,
                    )
    except Exception:
        # Be resilient: if any error occurs in gap shading, continue without it
        pass

    # Distribution histogram
    fig.add_trace(
        go.Histogram(
            x=dd[y_col],
            nbinsx=30,
            name=f"{metric_type.title()} Distribution",
            marker_color=color,
            opacity=0.75,
        ),
        row=2, col=1,
    )

    fig.update_layout(
        height=600,
        title=title,
        showlegend=True,
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(title_text="Date/Time", row=1, col=1)
    fig.update_yaxes(title_text=y_label, row=1, col=1)
    fig.update_xaxes(title_text=dist_x_label, row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Number of Hours)", row=2, col=1)

    return fig


def volume_charts(
    data: pd.DataFrame,
    theoretical_link_capacity_vph: int,
    high_volume_threshold_vph: int,
):
    if data.empty:
        return None, None, None
    dd = data.dropna(subset=["local_datetime", "total_volume", "intersection_name"]).copy()
    dd.sort_values("local_datetime", inplace=True)

    # 1) Trend by intersection
    fig1 = px.line(
        dd,
        x="local_datetime",
        y="total_volume",
        color="intersection_name",
        title=" Traffic Volume Trends by Intersection",
        labels={"total_volume": "Volume (vehicles/hour)", "local_datetime": "Date/Time"},
        template="plotly_white",
    )
    fig1.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # 2) Distribution + Hourly heatmap
    fig2 = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Volume Distribution by Intersection", "Hourly Avg Volume Heatmap"),
        vertical_spacing=0.12,
    )

    # Box plots
    for name, g in dd.groupby("intersection_name", sort=False):
        fig2.add_trace(go.Box(y=g["total_volume"], name=name, boxpoints="outliers"), row=1, col=1)

    dd["hour"] = dd["local_datetime"].dt.hour
    hourly_avg = dd.groupby(["hour", "intersection_name"], as_index=False)["total_volume"].mean()
    hourly_pivot = hourly_avg.pivot(index="intersection_name", columns="hour", values="total_volume").sort_index()

    fig2.add_trace(
        go.Heatmap(
            z=hourly_pivot.values,
            x=hourly_pivot.columns,
            y=hourly_pivot.index,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Avg Volume (vph)"),
        ),
        row=2, col=1,
    )
    fig2.update_layout(
        height=800,
        title=" Volume Distribution & Capacity Analysis",
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # 3) Peak hour by intersection
    hourly_volume = dd.groupby(["hour", "intersection_name"], as_index=False)["total_volume"].mean()
    fig3 = px.line(
        hourly_volume,
        x="hour",
        y="total_volume",
        color="intersection_name",
        title=" Average Hourly Volume Patterns",
        labels={"total_volume": "Average Volume (vph)", "hour": "Hour of Day"},
        template="plotly_white",
    )
    fig3.add_hline(
        y=theoretical_link_capacity_vph,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Theoretical Capacity ({theoretical_link_capacity_vph:,} vph)",
    )
    fig3.add_hline(
        y=high_volume_threshold_vph,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"High Volume Threshold ({high_volume_threshold_vph:,} vph)",
    )
    fig3.update_layout(
        height=500,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig1, fig2, fig3


# =========================
# Date range UI helper
# =========================
def date_range_preset_controls(min_date: datetime.date, max_date: datetime.date, key_prefix: str):
    """
    Presets that default to Last 30 Days on first load, persist in session_state,
    and won't clobber custom picks.
    """
    k_range = f"{key_prefix}_range"

    # Default to LAST 30 DAYS (bounded by min_date)
    if k_range not in st.session_state:
        default_start = max(min_date, max_date - timedelta(days=30))
        st.session_state[k_range] = (default_start, max_date)

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button(" Last 7 Days", key=f"{key_prefix}_7d"):
            st.session_state[k_range] = (max(min_date, max_date - timedelta(days=7)), max_date)
    with c2:
        if st.button(" Last 30 Days", key=f"{key_prefix}_30d"):
            st.session_state[k_range] = (max(min_date, max_date - timedelta(days=30)), max_date)
    with c3:
        if st.button(" Full Range", key=f"{key_prefix}_full"):
            st.session_state[k_range] = (min_date, max_date)

    custom = st.date_input(
        "Custom Date Range",
        value=st.session_state[k_range],
        min_value=min_date,
        max_value=max_date,
        key=f"{key_prefix}_custom",
    )
    if custom != st.session_state[k_range]:
        st.session_state[k_range] = custom

    return st.session_state[k_range]


# =========================
# Data availability helper (for sidebar preview)
# =========================
def compute_data_availability(
    df: pd.DataFrame,
    *,
    datetime_col: str = "local_datetime",
    intersection_col: str | None = None,
    intersection: str | None = None,
    max_gaps: int = 3,
    current_date: datetime | None = None,
) -> dict:
    """
    Compute a compact availability summary for an (optionally filtered) dataframe:
      - date/time range (min → max) of available data
      - approximate size in MB (memory footprint of filtered frame)
      - list of missing ranges (contiguous sequences), limited to `max_gaps` entries
        • If `current_date` is provided, expected timeline extends to `current_date` (now),
          so gaps after the last observed point up to `current_date` are counted as missing.
          Future times after `current_date` are never considered missing.

    Returns a dict with keys: start (Timestamp), end (Timestamp), size_mb (float), gaps (list[str]), tail_gap (str | None).
    """
    out = {"start": None, "end": None, "size_mb": 0.0, "gaps": [], "tail_gap": None}
    if df is None or df.empty or datetime_col not in df.columns:
        return out

    dfx = df.copy()
    # Optional intersection filter
    if intersection and intersection_col and intersection_col in dfx.columns and intersection != "All Intersections":
        dfx = dfx[dfx[intersection_col] == intersection]
        if dfx.empty:
            return out

    # Ensure datetime and sort
    dfx[datetime_col] = pd.to_datetime(dfx[datetime_col], errors="coerce")
    dfx = dfx.dropna(subset=[datetime_col]).sort_values(datetime_col).reset_index(drop=True)
    if dfx.empty:
        return out

    start_ts = dfx[datetime_col].iloc[0]
    end_ts = dfx[datetime_col].iloc[-1]
    out["start"], out["end"] = start_ts, end_ts

    # Approx memory size in MB of the filtered data
    try:
        size_mb = float(dfx.memory_usage(deep=True).sum()) / (1024.0 ** 2)
    except Exception:
        size_mb = 0.0
    out["size_mb"] = size_mb

    # Determine expected frequency robustly; prefer hourly when timestamps align on the hour
    s = dfx[datetime_col]
    # Normalize to naive timestamps for ops
    try:
        s = s.dt.tz_convert(None)
    except Exception:
        pass

    inferred = None
    try:
        inferred = pd.infer_freq(s)
    except Exception:
        inferred = None

    # Heuristic based on alignment of minutes within hour
    minutes = s.dt.minute
    seconds = s.dt.second
    if seconds.notna().any() and (seconds != 0).any():
        # If seconds present, fall back to minute-based checks
        seconds_unique = seconds.dropna().unique()
    # Prefer hourly if all timestamps land on :00
    if minutes.notna().all() and (minutes == 0).all():
        freq = "H"
    elif (minutes % 15 == 0).all():
        freq = "15T"
    elif (minutes % 5 == 0).all():
        freq = "5T"
    else:
        # Use mode of deltas in minutes and snap to common granularities
        deltas_min = s.diff().dropna().dt.total_seconds() / 60.0
        if not deltas_min.empty:
            common_min = float(deltas_min.mode().iloc[0])
            if common_min <= 6:
                freq = "5T"
            elif common_min <= 15:
                freq = "15T"
            elif common_min <= 30:
                freq = "30T"
            elif common_min <= 90:
                freq = "H"
            else:
                # If very sparse, still treat as hourly for gap grouping
                freq = "H"
        else:
            # Fall back to whatever Pandas inferred or hourly
            freq = inferred if inferred in ("5T", "15T", "30T", "H") else "H"

    # Build expected range up to current_date (if provided), otherwise up to end_ts
    # Ensure we don't consider future past current_date
    try:
        now_cap = pd.Timestamp(current_date) if current_date is not None else None
    except Exception:
        now_cap = None
    expected_end = now_cap if now_cap is not None and now_cap > end_ts else end_ts

    # Build the expected index and find missing timestamps
    try:
        full_index = pd.date_range(start=start_ts.floor(freq), end=expected_end.ceil(freq), freq=freq)
    except Exception:
        # As a very safe fallback, use hourly
        full_index = pd.date_range(start=start_ts.floor("H"), end=expected_end.ceil("H"), freq="H")
        freq = "H"

    # Align actual to same freq precision
    try:
        if freq.endswith("T"):
            actual = s.dt.floor(freq)
        else:
            actual = s.dt.floor("H")
    except Exception:
        actual = s.dt.floor("H")

    missing = pd.Index(full_index).difference(pd.Index(actual.unique())).sort_values()
    # Precompute step for later use (tail gap too)
    step = (full_index[1] - full_index[0]) if len(full_index) > 1 else (pd.Timedelta(minutes=5) if isinstance(freq, str) and freq.endswith("T") else pd.Timedelta(hours=1))

    # Tail gap (from first expected tick after the last observed tick → now/current_date), only if current_date > end_ts
    tail_gap_tuple = None
    try:
        if now_cap is not None and now_cap > end_ts:
            last_tick = actual.max()
            next_tick = last_tick + step if pd.notna(last_tick) else start_ts
            # We want to extend to the end of the present day (last expected tick today), not just 'now'.
            # Compute the last tick of today based on the inferred step.
            today_start = now_cap.normalize()
            day_end_candidate = today_start + pd.Timedelta(days=1) - step
            end_at = day_end_candidate
            if end_at > next_tick:
                tail_gap_tuple = (next_tick, end_at)
    except Exception:
        tail_gap_tuple = None

    if len(missing) == 0 and not tail_gap_tuple:
        out["gaps"] = []
        return out

    # Group consecutive missing timestamps into ranges
    gaps = []
    if len(missing) > 0:
        run_start = missing[0]
        prev = missing[0]
        for ts in missing[1:]:
            if ts - prev > step + pd.Timedelta(seconds=1):
                gaps.append((run_start, prev))
                run_start = ts
            prev = ts
        gaps.append((run_start, prev))

    # Format with times
    def _fmt_range(a: pd.Timestamp, b: pd.Timestamp) -> str:
        same_day = (a.date() == b.date())
        if same_day:
            a_str = a.strftime("%b %d, %Y %I:%M %p")
            b_str = b.strftime("%I:%M %p")
            if a == b:
                return a_str
            return f"{a_str}–{b_str}"
        if a.year == b.year:
            return f"{a.strftime('%b %d %I:%M %p')}–{b.strftime('%b %d, %Y %I:%M %p')}"
        return f"{a.strftime('%b %d, %Y %I:%M %p')}–{b.strftime('%b %d, %Y %I:%M %p')}"

    gap_strs = [_fmt_range(a, b) for a, b in gaps]
    if len(gap_strs) > max_gaps:
        extra = len(gap_strs) - max_gaps
        gap_strs = gap_strs[:max_gaps] + [f"… and {extra} more"]
    out["gaps"] = gap_strs

    # Attach tail gap formatted string if present
    if tail_gap_tuple is not None:
        out["tail_gap"] = _fmt_range(tail_gap_tuple[0], tail_gap_tuple[1])
    else:
        out["tail_gap"] = None

    return out


# =========================
# Processing
# =========================
def process_traffic_data(df, date_range, granularity, time_filter=None, start_hour=None, end_hour=None):
    """
    Process traffic data based on date range and granularity selections.
    Optimizations:
    - Early date/time filtering to minimize rows
    - Robust time_filter label handling (ASCII vs en dash)
    - If base data are sub-hourly (e.g., 5-min), aggregate to Hourly before applying hourly filters
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = df.copy()

    # Ensure datetime
    df["local_datetime"] = pd.to_datetime(df.get("local_datetime", pd.NaT), errors="coerce")
    df = df.dropna(subset=["local_datetime"])  # safety

    # Early date range filter
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date is not None and end_date is not None:
            mask = (df["local_datetime"].dt.date >= start_date) & (df["local_datetime"].dt.date <= end_date)
            df = df.loc[mask]
    if df.empty:
        return df

    # Normalize time filter label (handle en dash, extra spaces)
    tf = str(time_filter or "").replace("–", "-").strip()

    # If granular is Hourly but timestamps are sub-hourly, consolidate first
    def _hourly_group(gdf: pd.DataFrame, by_cols: list[str], metrics: list[str], how: str = "mean") -> pd.DataFrame:
        gdf = gdf.copy()
        gdf["hour"] = gdf["local_datetime"].dt.floor("H")
        agg = gdf.groupby(by_cols + ["hour"], observed=True)[metrics].agg(how).reset_index()
        agg = agg.rename(columns={"hour": "local_datetime"})
        return agg

    is_corridor = "segment_name" in df.columns
    is_volume = "intersection_id" in df.columns or "total_volume" in df.columns

    if is_corridor:
        # Ensure numeric dtypes
        for c in ("average_delay", "average_traveltime", "average_speed"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # Aggregate to requested granularity
        if granularity == "Hourly":
            # Consolidate sub-hourly to hourly if needed
            if (df["local_datetime"].dt.minute != 0).any():
                df = _hourly_group(df, ["corridor_id", "direction", "segment_name"], ["average_delay", "average_traveltime", "average_speed"], how="mean")
        elif granularity == "Daily":
            df["date_group"] = df["local_datetime"].dt.date
            df = df.groupby(["date_group", "corridor_id", "direction", "segment_name"], observed=True)[
                ["average_delay", "average_traveltime", "average_speed"]
            ].mean().reset_index().rename(columns={"date_group": "local_datetime"})
            df["local_datetime"] = pd.to_datetime(df["local_datetime"])
        elif granularity == "Weekly":
            df["week_group"] = df["local_datetime"].dt.to_period("W").dt.start_time
            df = df.groupby(["week_group", "corridor_id", "direction", "segment_name"], observed=True)[
                ["average_delay", "average_traveltime", "average_speed"]
            ].mean().reset_index().rename(columns={"week_group": "local_datetime"})
        elif granularity == "Monthly":
            df["month_group"] = df["local_datetime"].dt.to_period("M").dt.start_time
            df = df.groupby(["month_group", "corridor_id", "direction", "segment_name"], observed=True)[
                ["average_delay", "average_traveltime", "average_speed"]
            ].mean().reset_index().rename(columns={"month_group": "local_datetime"})

        # Apply time-of-day filters only when Hourly
        if granularity == "Hourly" and tf:
            hrs = df["local_datetime"].dt.hour
            if tf == "Peak Hours (7-9 AM, 4-6 PM)":
                df = df[hrs.between(7, 9) | hrs.between(16, 18)]
            elif tf == "AM Peak (7-9 AM)":
                df = df[hrs.between(7, 9)]
            elif tf == "PM Peak (4-6 PM)":
                df = df[hrs.between(16, 18)]
            elif tf == "Off-Peak":
                df = df[~hrs.between(7, 9) & ~hrs.between(16, 18)]
            elif tf == "Custom Range" and start_hour is not None and end_hour is not None:
                df = df[hrs.between(int(start_hour), int(end_hour) - 1)]

        return df.sort_values("local_datetime").reset_index(drop=True)

    if is_volume:
        # Volume aggregation
        if "total_volume" in df.columns:
            df["total_volume"] = pd.to_numeric(df["total_volume"], errors="coerce").fillna(0)
        if granularity == "Hourly":
            if (df["local_datetime"].dt.minute != 0).any():
                # sum volumes to the hour
                df = _hourly_group(df, ["intersection_id", "direction", "intersection_name" if "intersection_name" in df.columns else "intersection_id"], ["total_volume"], how="sum")
        elif granularity == "Daily":
            df["date_group"] = df["local_datetime"].dt.date
            df = df.groupby(["date_group", "intersection_id", "direction", "intersection_name"], observed=True)["total_volume"].sum().reset_index().rename(columns={"date_group": "local_datetime"})
            df["local_datetime"] = pd.to_datetime(df["local_datetime"])
        elif granularity == "Weekly":
            df["week_group"] = df["local_datetime"].dt.to_period("W").dt.start_time
            df = df.groupby(["week_group", "intersection_id", "direction", "intersection_name"], observed=True)["total_volume"].sum().reset_index().rename(columns={"week_group": "local_datetime"})
        elif granularity == "Monthly":
            df["month_group"] = df["local_datetime"].dt.to_period("M").dt.start_time
            df = df.groupby(["month_group", "intersection_id", "direction", "intersection_name"], observed=True)["total_volume"].sum().reset_index().rename(columns={"month_group": "local_datetime"})

        # Hourly time filter for volume
        if granularity == "Hourly" and tf:
            hrs = df["local_datetime"].dt.hour
            if tf == "Peak Hours (7-9 AM, 4-6 PM)":
                df = df[hrs.between(7, 9) | hrs.between(16, 18)]
            elif tf == "AM Peak (7-9 AM)":
                df = df[hrs.between(7, 9)]
            elif tf == "PM Peak (4-6 PM)":
                df = df[hrs.between(16, 18)]
            elif tf == "Off-Peak":
                df = df[~hrs.between(7, 9) & ~hrs.between(16, 18)]
            elif tf == "Custom Range" and start_hour is not None and end_hour is not None:
                df = df[hrs.between(int(start_hour), int(end_hour) - 1)]

        return df.sort_values("local_datetime").reset_index(drop=True)

    # Fallback - return filtered df
    return df.sort_values("local_datetime").reset_index(drop=True)

#Function to help download 5 minute CSV

def create_5min_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert filtered data to 5-minute intervals by interpolating between existing data points.
    This creates more granular data for detailed analysis without requiring actual 5-minute source data.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Make a copy and ensure datetime column exists
    result_df = df.copy()
    if "local_datetime" not in result_df.columns:
        return result_df

    # Ensure datetime is properly formatted
    result_df["local_datetime"] = pd.to_datetime(result_df["local_datetime"], errors="coerce")
    result_df = result_df.dropna(subset=["local_datetime"]).sort_values("local_datetime")

    if len(result_df) < 2:
        return result_df

    # Create 5-minute intervals between min and max dates
    start_time = result_df["local_datetime"].min().floor("5T")
    end_time = result_df["local_datetime"].max().ceil("5T")

    # Generate 5-minute time range
    time_range = pd.date_range(start=start_time, end=end_time, freq="5T")

    # For each unique combination of non-datetime columns, interpolate
    id_cols = [col for col in result_df.columns if col not in ["local_datetime"]
               and not pd.api.types.is_numeric_dtype(result_df[col])]
    numeric_cols = [col for col in result_df.columns
                    if col != "local_datetime" and pd.api.types.is_numeric_dtype(result_df[col])]

    if not id_cols:  # If no grouping columns, treat as single series
        # Simple case - just one series to interpolate
        temp_df = pd.DataFrame({"local_datetime": time_range})
        merged = pd.merge(temp_df, result_df, on="local_datetime", how="left")

        # Set datetime as index for time-based interpolation
        merged_indexed = merged.set_index("local_datetime")

        # Interpolate numeric columns using linear method (safer than time method)
        for col in numeric_cols:
            if col in merged_indexed.columns:
                merged_indexed[col] = merged_indexed[col].interpolate(method="linear")

        # Reset index to get datetime back as column
        merged = merged_indexed.reset_index()
        return merged.dropna()

    # Complex case - multiple groups to interpolate
    all_interpolated = []

    for group_vals, group_df in result_df.groupby(id_cols):
        if len(group_df) < 2:
            continue

        # Create time range for this group
        group_start = group_df["local_datetime"].min().floor("5T")
        group_end = group_df["local_datetime"].max().ceil("5T")
        group_time_range = pd.date_range(start=group_start, end=group_end, freq="5T")

        # Create base dataframe with 5-minute intervals
        temp_df = pd.DataFrame({"local_datetime": group_time_range})

        # Add the group identifier columns
        if isinstance(group_vals, tuple):
            for i, col in enumerate(id_cols):
                temp_df[col] = group_vals[i]
        else:
            temp_df[id_cols[0]] = group_vals

        # Merge with existing data
        merged = pd.merge(temp_df, group_df, on=["local_datetime"] + id_cols, how="left")

        # Set datetime as index for interpolation
        merged_indexed = merged.set_index("local_datetime")

        # Interpolate numeric columns using linear method
        for col in numeric_cols:
            if col in merged_indexed.columns:
                merged_indexed[col] = merged_indexed[col].interpolate(method="linear")

        # Reset index to get datetime back as column
        merged = merged_indexed.reset_index()

        # Only keep rows that have interpolated data
        merged = merged.dropna(subset=numeric_cols, how="all")
        all_interpolated.append(merged)

    if all_interpolated:
        return pd.concat(all_interpolated, ignore_index=True).sort_values("local_datetime")
    else:
        return pd.DataFrame()
