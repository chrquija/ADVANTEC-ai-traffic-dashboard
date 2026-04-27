# Python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Plotly for chart helpers
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Shared UI utils (scoped loader and tab highlight)
try:
    from ui_utils import get_dynamic_xaxis_params
except ImportError:
    try:
        from core.ui_utils import get_dynamic_xaxis_params
    except ImportError:
        get_dynamic_xaxis_params = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _fix_raw_url(url: str) -> str:
    """
    GitHub RAW URLs must be:
      https://raw.githubusercontent.com/<user>/<repo>/<branch>/<path>
    Some of your links used '/main/'. This converts them.
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

def get_corridor_display_name(raw_name: str) -> str:
    """
    Map raw corridor identifiers, slugs, or internal names to human-readable display names.
    Ensures that 'Washington Street', 'Avenue 47', 'Highway 111', and 'Adams St' are returned.
    """
    if not raw_name or pd.isna(raw_name):
        return "Washington Street"

    raw_name = str(raw_name).strip()
    raw_lower = raw_name.lower()

    # 1. Catch segment/O-D identifiers for Avenue 47 and Highway 111 corridors
    # These have patterns like 'adamsst_to_washingtonst' or 'washingtonst_to_adamsst'
    if "adamsst" in raw_lower and "washingtonst" in raw_lower:
        # This is an Avenue 47 or Highway 111 segment - but we need the corridor_name
        # to determine which. Return None here to let the caller use corridor_name instead.
        # Actually, we don't have context here, so we'll handle this at the load level.
        pass  # Fall through to other checks

    # 2. Catch segment/O-D identifiers (e.g., 'ave52_to_calletampico')
    # and map them to the parent corridor (Washington Street).
    # But EXCLUDE the adamsst/washingtonst patterns since those belong to Ave47/Hwy111
    if ("_to_" in raw_lower or "→" in raw_lower) and not ("adamsst" in raw_lower or "washingtonst" in raw_lower and "adamsst" in raw_lower):
        return "Washington Street"

    # 2.5 E Palm Canyon Drive (Subset of Hwy 111, keep separate)
    if "palm canyon" in raw_lower or "epalmcanyon" in raw_lower:
        return "Highway 111 - E Palm Canyon Drive"

    # 3. Highway 111 variants
    if any(x in raw_lower for x in ["highway 111", "hwy111", "highway111", "hwy 111"]):
        return "Highway 111"
    # Be careful with just "111" - only match if it's clearly Highway 111
    if raw_lower == "111" or raw_lower.startswith("111 "):
        return "Highway 111"

    # 4. Avenue 47 variants
    if any(x in raw_lower for x in ["avenue 47", "avenue47", "ave 47", "ave47"]):
        return "Avenue 47"

    # 5. Adams St variants
    if any(x in raw_lower for x in ["adams st", "adamsst", "adams street"]):
        return "Adams St"

    # 6. Default everything else to Washington Street
    # This includes 'Washington', 'Wash St', 'Fred Waring Drive', etc.
    return "Washington Street"


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
        "Avenue 52 → Calle Tampico": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/1_2_LONG_NSB_Ave52_CalleTampico_WashSt_1hr_septojuly.csv",
        "Calle Tampico → Village Shopping Ctr": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/2_3_LONG_NSB_CalleTampico_VillageShoppingCtr_WashSt_1hr_septojuly.csv",
        "Village Shopping Ctr → Avenue 50": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/3_4_LONG_NSB_VillageShoppingCtr_Avenue50_WashSt_1hr_septojuly.csv",
        "Avenue 50 → Sagebrush Ave": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/4_5_LONG_NSB_Ave50_SagebrushAve_WashSt_1hr_septojuly.csv",
        "Sagebrush Ave → Eisenhower Dr": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/5_6_LONG_NSB_SagebrushAve_EisenhowerDr_WashSt_1hr_septojuly.csv",
        "Eisenhower Dr → Avenue 48": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/6_7_LONG_NSB_EisenhowerDr_Avenue48_WashSt_1hr_septojuly.csv",
        "Avenue 48 → Avenue 47": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/7_8_LONG_NSB_Ave48_Ave47_WashSt_1hr_septojuly.csv",
        "Avenue 47 → Point Happy Simon": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/8_9_LONG_NSB_Ave47_PointHappySimon_WashSt_1hr_septojuly.csv",
        "Point Happy Simon → Hwy 111": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/9_10_LONG_NSB_PointHappySimon_WashSt_1hr_septojuly.csv",

        # New segments extending north from Highway 111
        "Hwy 111 → Channel Drive": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/10_11_LONG_NSB_Hwy111_to_ChannelDrive.csv",
        "Channel Drive → Miles Avenue": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/11_12_LONG_NSB_ChannelDrive_to_MilesAvenue.csv",
        "Miles Avenue → Via Sevilla": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/12_13_LONG_NSB_MilesAvenue_to_ViaSevilla.csv",
        "Via Sevilla → Fred Waring Drive": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/13_14_LONG_NSB_ViaSevilla_FredWaringDrive.csv",
        "Fred Waring Drive → Palm Royale Drive": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/14_15_LONG_NSB_FredWaringDrive_to_PalmRoyaleDrive.csv",
        "Palm Royale Drive → Avenue of the States": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/15_16_LONG_NSB_PalmRoyaleDrive_to_AvenueoftheStates.csv",
        "Avenue of the States → Avenue 42": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/16_17_LONG_NSB_AvenueoftheStates_to_Avenue42.csv",
        "Avenue 42 → Avenue 41": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/17_18_LONG_NSB_Avenue42_to_Avenue41.csv",
        # New Northbound Segments
        "Avenue 41 → Harris Lane": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/119_120_1hr_LONG_NB_Avenue41_to_HarrisLn.csv",
        "Harris Lane → Country Club Drive": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/120_121_1hr_LONG_NB_HarrisLn_to_CountryClubDrive.csv",
        "Country Club Drive → I-10 Interchange": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/121_122_1hr_LONG_NB_CountryClubDrive_to_I10interchange.csv",
        "I-10 Interchange → Varner Road": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/122_123_1hr_LONG_NB_I10interchange_to_VarnerRd.csv",
        "Varner Road → Market Pl": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/123_124_1hr_LONG_NB_VarnerRd_to_MarketPl.csv",
        "Market Pl → Del Webb": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/124_125_1hr_LONG_NB_MarketPl_to_DelWebb.csv",

        # New Southbound Segments
        "I-10 Interchange → Country Club Drive": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/122_121_1hr_LONG_SB_I10interchange_to_CountryClubDrive.csv",
        "Varner Road → I-10 Interchange": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/123_122_1hr_LONG_SB_VarnerRd_to_I10interchange.csv",
        "Market Pl → Varner Road": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/124_123_1hr_LONG_SB_MarketPl_to_VarnerRd.csv",
        "Del Webb → Market Pl": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/125_124_1hr_LONG_SB_DelWebb_to_MarketPl.csv",

        # Existing Southbound segments (kept as requested)
        "Harris Lane → Avenue 41": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/19_18_LONG_SB_Harrislane_avenue41.csv",
        "Country Club Drive → Harris Lane": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/20_19_LONG_SB_CountryClubDrive_to_HarrisLane.csv",

        # --- New Fixed-Endpoint Corridors (Single-file both directions) ---
        "Avenue 47 Corridor": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/EWB_AVE47CORR_WashingtonSt_to_AdamsSt_1hr_Sep24TOApr26.csv",
        "Highway 111 Corridor": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/76_EWB_HWY111CORR_WashingtonSt_to_AdamsSt_1hr_Sep24TOApr26.csv",
        "Adams St Corridor": "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/NSB_ADAMSSTCORR_Avenue48_to_Highway111_1hr_Sep24TOApr26.csv",
    }

    usecols = [
        "local_datetime",
        "corridor_id",
        "corridor_name",  # Added for new corridors
        "segment_name",   # Added for new corridors
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

            # --- Column & Data Normalization ---

            # 1. Segment Name Identification (must happen BEFORE corridor_id overwrite)
            # In some files (Ave 47/Hwy 111), corridor_id contains segment info (adamsst_to_washingtonst)
            # while corridor_name contains the corridor name.
            if "corridor_name" in df.columns and "segment_name" not in df.columns:
                df["segment_name"] = df["corridor_id"]

            # 2. Standardize Corridor Name
            if "corridor_name" in df.columns:
                df["corridor_id"] = df["corridor_name"]

            if "corridor_id" in df.columns:
                df["corridor_id"] = df["corridor_id"].astype(str).apply(get_corridor_display_name)
            else:
                # Default for files without a corridor column
                df["corridor_id"] = "Washington Street"

            # 3. Segment Name & Direction Normalization
            if "segment_name" in df.columns and not df["segment_name"].isna().all():
                seg_map = {
                    "adamsst_to_washingtonst": "Adams St → Washington Street",
                    "washingtonst_to_adamsst": "Washington Street → Adams St",
                    "avenue48_to_highway111": "Avenue 48 → Hwy 111",
                    "highway111_to_avenue48": "Hwy 111 → Avenue 48",
                }
                df["segment_name"] = df["segment_name"].astype(str).replace(seg_map)

                # Ensure direction is set correctly for these fixed segments
                df.loc[df["segment_name"] == "Adams St → Washington Street", "direction"] = "Westbound"
                df.loc[df["segment_name"] == "Washington Street → Adams St", "direction"] = "Eastbound"
                df.loc[df["segment_name"] == "Hwy 111 → Avenue 48", "direction"] = "Southbound"
                df.loc[df["segment_name"] == "Avenue 48 → Hwy 111", "direction"] = "Northbound"
            else:
                # Fallback to dictionary key for standard multi-file corridors (NB/SB)
                df["segment_name"] = segment_name

            # Ensure needed numeric columns exist even if missing in source
            for c in ["average_delay", "average_traveltime", "average_speed"]:
                if c not in df.columns:
                    df[c] = np.nan

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
    # Support multiple volume sources: Washington Street, Highway 111, E Palm Canyon Dr
    volume_urls = [
        _fix_raw_url(
            "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/VOLUME/KMOB_LONG/FULL_AvailableVolumeCounts_WashingtonCorridor_UPDATED.csv"
        ),
        _fix_raw_url(
            "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/VOLUME/KMOB_LONG/ALL_KMOB_Hwy111.csv"
        ),
        _fix_raw_url(
            "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/VOLUME/KMOB_LONG/ALL_1hr_EPalmCanyonDr_Oct24ToNovember25.csv"
        ),
    ]

    def _norm_dir(s: pd.Series) -> pd.Series:
        s = s.astype(str).str.strip().str.upper()
        map_dir = {
            "N": "NB", "NB": "NB", "NORTH": "NB", "NORTHBOUND": "NB",
            "S": "SB", "SB": "SB", "SOUTH": "SB", "SOUTHBOUND": "SB",
            "E": "EB", "EB": "EB", "EAST": "EB", "EASTBOUND": "EB",
            "W": "WB", "WB": "WB", "WEST": "WB", "WESTBOUND": "WB",
            "WEST-BOUND": "WB", "EAST-BOUND": "EB", "NORTH-BOUND": "NB", "SOUTH-BOUND": "SB",
        }
        return s.map(map_dir)

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
            "Washington_St_and_PointHappyWay": "Point Happy Simon",
            "Washington_St_and_Point_Happy_Way": "Point Happy Simon",
            "Washington_St_and_Avenue48": "Avenue 48",
            "Washington_St_and_Avenue47": "Avenue 47",
            "Washington_St_and_ChannelDrive": "Channel Drive",
            "Washington_St_and_MilesAvenue": "Miles Avenue",
            "Washington_St_and_ViaSevilla": "Via Sevilla",
            "Washington_St_to_Avenue42": "Avenue 42",
            "Washington_St_and_HarrisLane": "Harris Lane",
            "Washington_St_and_VarnerRoad": "Varner Road",
            "Washington_St_and_CountryClubDrive": "Country Club Drive",
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
        frames = []
        for url in volume_urls:
            try:
                temp_df = pd.read_csv(url)
                # Assign specific corridor name for E Palm Canyon Dr dataset to avoid conflict with Hwy111 (Indio)
                if "EPalmCanyonDr" in url:
                    temp_df["corridor_id"] = "Highway 111 - E Palm Canyon Drive"
                frames.append(temp_df)
            except Exception as e:
                st.warning(f"Could not load volume data from {url}: {e}")
        
        if not frames:
            raise RuntimeError("No volume data could be loaded from any source.")
        
        df = pd.concat(frames, ignore_index=True)
        
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
            df["corridor_id"] = "Washington Street"

        # Normalize corridor_id naming to "Highway 111" and "Washington Street"
        df["corridor_id"] = df["corridor_id"].astype(str).apply(get_corridor_display_name)
        df["corridor_id"] = df["corridor_id"].astype("category")

        # Direction normalization
        if "direction" in df.columns:
            df["direction"] = _norm_dir(df["direction"])
            # Drop any rows where direction is not recognized (e.g., "-")
            df = df.dropna(subset=["direction"])
            df["direction"] = df["direction"].astype("category")
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
        def _hwy111_friendly(raw: str) -> str:
            hwy111_map = {
                "Highway111_and_ParkviewDrive": "Park View Drive",
                "Highway111_and_Highway74": "Highway 74",
                "Highway111_and_JackalopeTrail": "Jackalope Trail",
                "Highway111_and_ShieldsRoad": "Shields Road",
                "Highway111_and_OasisStreet": "Oasis Street",
                "Highway111_and_SmurrStreet": "Smurr Street",
                "Highway111_and_JacksonStreet": "Jackson Street",
                "Highway111_and_GolfCenterParkway": "Golf Center Parkway",
                "Highway111_and_IndioBlvd": "Indio Blvd",
                # E Palm Canyon Drive segment
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
                "EPalmCanyonDr_and_OfficeJermainGibson": "Officer Jermain Gibson"
            }
            return hwy111_map.get(raw, _friendly_label(raw))

        df["intersection_name"] = df.apply(
            lambda row: _hwy111_friendly(str(row[inter_key_col])) 
            if "highway 111" in str(row.get("corridor_id")).lower() 
            else _friendly_label(str(row[inter_key_col])), 
            axis=1
        )

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
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/MASTER_Acyclica_Traveltime_speed.csv"
    ),
    # New master with extended Washington Street segments (Ave 52 â†” Country Club Dr)
    _fix_raw_url(
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/MASTER_1hr_Acyclica_Traveltime_speed_Ave52toCountryClubDrive.csv"
    ),
    # Highway 111: Canyon Plaza West â†” Jermaine Gibson (EB/WB combined long-format)
    _fix_raw_url(
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/DELAY_TRAVELTIME_SPEED_byintersection/LONGFORMAT/MASTER_EWB_1hr_PalmCanyon_CanyonPlazaWest_to_JermainGibson.csv"
    ),
    # Highway 111: Cook Street â†” Parkview Drive (EB/WB)
    _fix_raw_url(
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/Highway_111_DATA/ACYCLICA/MASTER_EWB_1hr_Hwy111_CookStreet_to_Parkview.csv"
    ),
    # Highway 111: Cook Street â†” Washington Street (EB/WB)
    _fix_raw_url(
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/Highway_111_DATA/ACYCLICA/MASTER_EWB_1hr_hwy111_CookStreet_to_WashingtonStreet.csv"
    ),
    # Highway 111: Washington Street â†” Monroe Street (EB/WB)
    _fix_raw_url(
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/Highway_111_DATA/ACYCLICA/MASTER_EWB_1hr_Hwy111_WashingtonStreet_to_MonroeStreet.csv"
    ),
    # Highway 111: Monroe Street â†” Indio Blvd (EB/WB)
    _fix_raw_url(
        "https://raw.githubusercontent.com/chrquija/ADVANTEC-ai-traffic-dashboard/main/Highway_111_DATA/ACYCLICA/MASTER_EWB_1hr_Hwy111_Monroe_to_IndioBlvd.csv"
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
        return np.nan
    df["direction"] = dir_raw.apply(_norm_dir)
    # Drop any rows with invalid directions (like "-")
    df = df.dropna(subset=["direction"])

    # Segment ID normalization (strip spaces if column exists)
    if "segment_id" in df.columns:
        df["segment_id"] = df["segment_id"].astype(str).str.strip()

    # Corridor normalization: fix known variants/typos
    df["corridor_id"] = df["corridor_id"].astype(str).apply(get_corridor_display_name)

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

    # Include segment_id in index if present to preserve O→D segment selection.
    # Note: pivot_table/groupby drops rows with NaN in index.
    # Fill NaN segment_id with a placeholder to avoid dropping older master data.
    index_cols = ["local_datetime", "corridor_id", "direction"]
    if "segment_id" in df_long.columns:
        df_long = df_long.copy()
        df_long["segment_id"] = df_long["segment_id"].fillna("no_segment_id")
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
    # Prefer segment_id as segment_name if available, else corridor_id.
    # If segment_id is our placeholder, fallback to corridor_id.
    if "segment_id" in piv.columns:
        piv["segment_name"] = np.where(
            piv["segment_id"] == "no_segment_id",
            piv["corridor_id"].astype(str),
            piv["segment_id"].astype(str)
        )
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
            "help": "Average Travel Time\n\nWhat it means: The typical door-to-door trip time for this route with your current filters.\nWhy it exists: Gives a quick sense of what most trips take.\nHow itâ€™s calculated: Average of the hourly O-D trip times.\nFormula: mean(travel_time).",
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
            "help": "Buffer Index = (P95 - mean) / mean x 100.",
        },
        "reliability": {
            "value": reliability,
            "unit": "%",
            "score": score_reliability,
            "help": "Reliability Index = 100 - CV%, where CV% = stdev/mean x 100.",
        },
        "congestion_freq": {
            "value": cong_freq,
            "unit": "%",
            "score": score_congestion,
            "extra": f"Hours > {high_delay_threshold:.0f} min delay: {cong_hours}/{total_hours}",
            "help": "The percentage of hours where average corridor delay exceeded the threshold in minutes. This refrences the HCM 6th Edition Level of Service E definition of approximately 55 seconds or 1 minute of control delay indicating unstable flow. Source: Highway Capacity Manual 6th Edition, TRB 2016. https://www.trb.org/Main/Blurbs/175169.aspx",
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

def format_chart_title(
    corridor_name: str,
    direction: str,
    origin: str,
    destination: str,
    date_range: tuple,
    metric_description: str
) -> str:
    """
    Generate a clean 2-line title for charts following the corridor-based system.
    """
    # Line 1: [Corridor Name]: [Direction] [Origin] to [Destination]
    # Ensure human-readable name
    corridor_display = get_corridor_display_name(corridor_name)
    direction = direction.title() if direction else ""
    origin = origin.title() if origin else ""
    destination = destination.title() if destination else ""

    line1 = f"<b>{corridor_display}: {direction} {origin} to {destination}</b>"

    # Date formatting: March 5, 2025
    def format_date_simple(d):
        return f"{d.strftime('%B')} {d.day}, {d.year}"

    if date_range and len(date_range) == 2:
        start_date, end_date = date_range
        # Check if they are the same date
        if hasattr(start_date, 'date'):
            d1, d2 = start_date.date(), end_date.date()
        else:
            d1, d2 = start_date, end_date

        if d1 == d2:
            date_str = format_date_simple(d1)
        else:
            date_str = f"{format_date_simple(d1)} to {format_date_simple(d2)}"
    else:
        date_str = "Selected Period"

    # Line 2: [Date] | [Metric description]
    line2 = f"<span style='font-size: 16px; font-weight: normal; color: #222;'>{date_str} | {metric_description}</span>"

    return f"{line1}<br>{line2}"


def performance_chart(
    data: pd.DataFrame,
    metric_type: str = "delay",
    direction_label: str = "",
    corridor_name: str = "Washington Street",
    origin: str = "",
    destination: str = "",
    date_range: tuple = None
):
    if data.empty:
        return None, None
    metric_type = metric_type.lower().strip()
    if metric_type == "delay":
        y_col, color = "average_delay", "#e74c3c"
        y_label = "Average Delay (minutes)"
        trend_metric = "Delay by Hour"
        dist_metric = "Delay Distribution"
    else:
        y_col, color = "average_traveltime", "#3498db"
        y_label = "Average Travel Time (minutes)"
        trend_metric = "Travel Time by Hour"
        dist_metric = "Travel Time Distribution"

    dd = data.dropna(subset=["local_datetime", y_col]).sort_values("local_datetime")
    if dd.empty:
        return None, None

    # Distribution stats for annotation
    series = dd[y_col]
    mean_val = float(series.mean()) if not series.empty else 0.0
    p25 = float(series.quantile(0.25)) if not series.empty else 0.0
    p75 = float(series.quantile(0.75)) if not series.empty else 0.0
    p95 = float(series.quantile(0.95)) if not series.empty else 0.0
    unit = "min"
    dist_annotation = f"Most hours: {p25:.1f}-{p75:.1f} {unit}. Worst 5%: above {p95:.1f} {unit}."

    # 1. TIME SERIES CHART
    fig_trend = go.Figure()

    ts_hover_tmpl = (
        "%{x|%b %d, %Y %I:%M %p}<br>Avg Delay: %{y:.1f} min"
        if metric_type == "delay"
        else "%{x|%b %d, %Y %I:%M %p}<br>Avg Travel Time: %{y:.1f} min"
    )

    fig_trend.add_trace(
        go.Scatter(
            x=dd["local_datetime"],
            y=dd[y_col],
            mode="lines+markers",
            name=f"{metric_type.title()} Trend",
            line=dict(color=color, width=2.5),
            marker=dict(size=5),
            hovertemplate=ts_hover_tmpl,
        )
    )

    # Shade missing-data gaps
    try:
        times = pd.to_datetime(dd["local_datetime"]).sort_values().reset_index(drop=True)
        if len(times) >= 3:
            deltas = times.diff().dropna()
            med = deltas.median()
            if pd.notna(med) and med > pd.Timedelta(0):
                gap_threshold = med * 1.5
                for i in range(1, len(times)):
                    dt = times[i] - times[i - 1]
                    if dt > gap_threshold:
                        fig_trend.add_vrect(
                            x0=times[i - 1], x1=times[i],
                            fillcolor="#95a5a6", opacity=0.15, line_width=0, layer="below"
                        )
    except Exception:
        pass

    # AM, MD, PM Shading for 1-day range
    if date_range and len(date_range) == 2:
        try:
            d0 = pd.to_datetime(date_range[0]).date()
            d1 = pd.to_datetime(date_range[1]).date()
            if d0 == d1:
                d_str = d0.strftime("%Y-%m-%d")
                # AM (05:00-10:00)
                fig_trend.add_vrect(
                    x0=f"{d_str} 05:00", x1=f"{d_str} 10:00",
                    fillcolor="orange", opacity=0.1, layer="below", line_width=0,
                    annotation_text="AM", annotation_position="top left",
                    annotation_font=dict(size=12, color="orange")
                )
                # MD (11:00-15:00)
                fig_trend.add_vrect(
                    x0=f"{d_str} 11:00", x1=f"{d_str} 15:00",
                    fillcolor="green", opacity=0.1, layer="below", line_width=0,
                    annotation_text="MD", annotation_position="top left",
                    annotation_font=dict(size=12, color="green")
                )
                # PM (16:00-20:00)
                fig_trend.add_vrect(
                    x0=f"{d_str} 16:00", x1=f"{d_str} 20:00",
                    fillcolor="red", opacity=0.1, layer="below", line_width=0,
                    annotation_text="PM", annotation_position="top left",
                    annotation_font=dict(size=12, color="red")
                )
        except Exception:
            pass

    trend_title = format_chart_title(corridor_name, direction_label, origin, destination, date_range, trend_metric)
    fig_trend.update_layout(
        title=dict(text=trend_title, font=dict(size=22, color="#000"), x=0, y=0.9, yanchor='top'),
        height=450,
        showlegend=False,
        template="plotly_white",
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        margin=dict(t=160, b=60, l=70, r=30),
        xaxis=dict(
            title=dict(text="Date/Time", font=dict(size=18, color="#333")),
            tickfont=dict(size=14),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=18, color="#333")),
            tickfont=dict(size=14),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        ),
    )

    # Custom X-axis logic
    if get_dynamic_xaxis_params and not dd.empty:
        start_date = dd["local_datetime"].min()
        end_date = dd["local_datetime"].max()
        params = get_dynamic_xaxis_params(start_date, end_date)
        fig_trend.update_xaxes(dtick=params["dtick"], tickformat=params["tickformat"])

    # 2. DISTRIBUTION CHART
    fig_dist = go.Figure()

    hover_tmpl = (
        "%{y:d} hours had an avg delay between %{x} min"
        if metric_type == "delay"
        else "%{y:d} hours had a travel time between %{x} min"
    )

    fig_dist.add_trace(
        go.Histogram(
            x=dd[y_col],
            nbinsx=30,
            name=f"{metric_type.title()} Distribution",
            marker_color=color,
            opacity=0.8,
            hovertemplate=hover_tmpl,
        )
    )

    fig_dist.add_vline(x=mean_val, line_dash="dash", line_color="#333", annotation_text="Avg")
    fig_dist.add_vline(x=p95, line_dash="dot", line_color="#333", annotation_text="95th %ile")

    dist_full_metric = f"{dist_metric}<br><span style='font-size:13px; color:#666;'>{dist_annotation}</span>"
    dist_title = format_chart_title(corridor_name, direction_label, origin, destination, date_range, dist_full_metric)

    fig_dist.update_layout(
        title=dict(text=dist_title, font=dict(size=22, color="#000"), x=0, y=0.9, yanchor='top'),
        height=450,
        showlegend=False,
        template="plotly_white",
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        margin=dict(t=160, b=60, l=70, r=30),
        xaxis=dict(
            title=dict(text=y_label, font=dict(size=18, color="#333")),
            tickfont=dict(size=14),
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text="Frequency (Hours)", font=dict(size=18, color="#333")),
            tickfont=dict(size=14),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)"
        ),
    )

    return fig_trend, fig_dist


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
        markers=True,
    )
    fig1.update_layout(
        height=550,
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        margin=dict(t=160, b=60, l=70, r=30),
        title=dict(font=dict(size=22, color="#000"), x=0, y=0.9, yanchor='top'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            title=dict(text="Date/Time", font=dict(size=18, weight="bold", color="#333")),
            tickfont=dict(size=14),
            dtick=21600000,
            tickformat="%b %d\n%I:%M %p"
        ),
        yaxis=dict(title=dict(text="Vehicle Volume Counts", font=dict(size=18, weight="bold", color="#333")), tickfont=dict(size=14)),
    )

    # 2) Distribution + Hourly heatmap
    fig2 = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Volume Distribution by Intersection", "Hourly Volume Heatmap"),
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
            colorbar=dict(title="Volume (vehicles)"),
        ),
        row=2, col=1,
    )
    fig2.update_layout(
        height=800,
        title=dict(text=" Volume Distribution & Capacity Analysis", font=dict(size=22, color="#000"), x=0, y=0.9, yanchor='top'),
        template="plotly_white",
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        margin=dict(t=160, b=60, l=70, r=30),
    )

    # 3) Peak hour by intersection
    hourly_volume = dd.groupby(["hour", "intersection_name"], as_index=False)["total_volume"].mean()
    fig3 = px.line(
        hourly_volume,
        x="hour",
        y="total_volume",
        color="intersection_name",
        title=" Hourly Volume Patterns",
        labels={"total_volume": "Volume (vehicles)", "hour": "Hour of Day"},
        template="plotly_white",
        markers=True,
    )
    fig3.add_hline(
        y=theoretical_link_capacity_vph,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Theoretical Capacity ({theoretical_link_capacity_vph:,} vehicles)",
    )
    fig3.add_hline(
        y=high_volume_threshold_vph,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"High Volume Threshold ({high_volume_threshold_vph:,} vehicles)",
    )
    fig3.update_layout(
        height=550,
        plot_bgcolor = "white",
        paper_bgcolor = "white",
        margin=dict(t=160, b=60, l=70, r=30),
        title=dict(font=dict(size=22, color="#000"), x=0, y=0.9, yanchor='top'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title=dict(text="Hour of Day", font=dict(size=18, weight="bold", color="#333")), tickfont=dict(size=14)),
        yaxis=dict(title=dict(text="Vehicle Volume Counts", font=dict(size=18, weight="bold", color="#333")), tickfont=dict(size=14)),
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
        â€¢ If `current_date` is provided, expected timeline extends to `current_date` (now),
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
        if isinstance(intersection, list):
            dfx = dfx[dfx[intersection_col].isin(intersection)]
        else:
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
            return f"{a_str}-{b_str}"
        if a.year == b.year:
            return f"{a.strftime('%b %d %I:%M %p')}-{b.strftime('%b %d, %Y %I:%M %p')}"
        return f"{a.strftime('%b %d, %Y %I:%M %p')}-{b.strftime('%b %d, %Y %I:%M %p')}"

    gap_strs = [_fmt_range(a, b) for a, b in gaps]
    if len(gap_strs) > max_gaps:
        extra = len(gap_strs) - max_gaps
        gap_strs = gap_strs[:max_gaps] + [f"... and {extra} more"]
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
    tf = str(time_filter or "").replace("-", "-").strip()

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


def compute_missing_strength_gaps(df, start_date, end_date, granularity):
    """
    Identifies contiguous missing-data gaps for the Strength column.
    Handles wide-format data (average_speed, average_traveltime columns).
    Returns a DataFrame with columns:
    segment_id, direction, metric, missing_start, missing_end, missing_hours, missing_days, full_day_aligned, gap_type
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["segment_id", "direction", "metric", "missing_start", "missing_end", "missing_hours", "missing_days", "full_day_aligned", "gap_type"])

    # Map granularity to pandas frequency
    freq_map = {"Hourly": "1H", "Daily": "1D", "Weekly": "W", "Monthly": "MS"}
    freq = freq_map.get(granularity, "1H")

    # Hour multiplier for duration calculation
    interval_hours = {"1H": 1.0, "1D": 24.0, "W": 168.0, "MS": 720.0}.get(freq, 1.0)

    # Create expected date range
    expected_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    results = []

    # Map the wide columns back to Metric names for the report
    metric_cols = {
        "average_speed": "Speed",
        "average_traveltime": "Travel Time"
    }

    group_cols = ["segment_id", "direction"]
    temp_df = df.copy()
    for col in group_cols:
        if col not in temp_df.columns:
            temp_df[col] = "Unknown"

    if "local_datetime" not in temp_df.columns:
        return pd.DataFrame(columns=["segment_id", "direction", "metric", "missing_start", "missing_end", "missing_hours", "missing_days", "full_day_aligned", "gap_type"])

    for (seg_id, direction), group in temp_df.groupby(group_cols):
        group = group.set_index("local_datetime").sort_index()
        
        for col, metric_label in metric_cols.items():
            if col not in group.columns:
                continue
                
            # Reindex for this specific metric
            full_series = group[col].reindex(expected_range)
            is_missing = full_series.isna()
            
            if not is_missing.any():
                continue
            
            # Identify contiguous blocks
            is_missing_int = is_missing.astype(int)
            gap_id = (is_missing_int != is_missing_int.shift()).cumsum()
            
            # Group ONLY the missing blocks
            for _, block in is_missing_int[is_missing].groupby(gap_id):
                m_start = block.index[0]
                m_end = block.index[-1]
                m_intervals = len(block)
                m_hours = m_intervals * interval_hours
                
                # Alignment logic for "Full Day"
                # For hourly: start at 00:00, end at 23:00
                is_aligned = (m_start.hour == 0 and m_end.hour == 23) if freq == "1H" else True
                
                if m_hours >= 24 and is_aligned:
                    g_type = "full_days_missing"
                    aligned_val = "TRUE"
                else:
                    g_type = "partial_day_missing" if m_hours < 24 else "multi_day_partial"
                    aligned_val = "FALSE"

                results.append({
                    "segment_id": seg_id,
                    "direction": direction,
                    "metric": metric_label,
                    "missing_start": m_start.strftime("%m/%d/%Y %H:%M"),
                    "missing_end": m_end.strftime("%m/%d/%Y %H:%M"),
                    "missing_hours": int(m_hours),
                    "missing_days": round(m_hours / 24, 2),
                    "full_day_aligned": aligned_val,
                    "gap_type": g_type
                })

    return pd.DataFrame(results) if results else pd.DataFrame(columns=["segment_id", "direction", "metric", "missing_start", "missing_end", "missing_hours", "missing_days", "full_day_aligned", "gap_type"])


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
