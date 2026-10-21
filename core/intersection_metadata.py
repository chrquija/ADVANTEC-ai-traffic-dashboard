import json
import os
import re
import streamlit as st

METADATA_PATH = "reference_data/intersection_metadata.json"

def normalize_name(name):
    if not isinstance(name, str):
        return ""
    
    # 1. Lowercase and strip
    n = name.lower().strip()
    
    # 2. Standardize separators
    n = re.sub(r'\s*[&/@]\s*|\s+at\s+|\s+and\s+', ' and ', n)
    
    # 3. Replace common abbreviations
    replacements = {
        r'\bst\.?\b': 'street',
        r'\bave\.?\b': 'avenue',
        r'\bdr\.?\b': 'drive',
        r'\bblvd\.?\b': 'boulevard',
        r'\bhwy\.?\b': 'highway',
        r'\brd\.?\b': 'road',
        r'\bpkwy\.?\b': 'parkway',
        r'\bpl\.?\b': 'place',
        r'\bct\.?\b': 'court',
        r'\bln\.?\b': 'lane',
        r'\bctr\.?\b': 'center',
        r'\bwy\.?\b': 'way'
    }
    for pattern, repl in replacements.items():
        n = re.sub(pattern, repl, n)
    
    # 4. Remove extra whitespace
    n = re.sub(r'\s+', ' ', n).strip()
    
    # 5. Split, sort streets alphabetically, and join
    parts = [p.strip() for p in n.split(' and ')]
    parts.sort()
    return ' and '.join(parts)

@st.cache_data
def load_cv_sync_intersection_metadata():
    if not os.path.exists(METADATA_PATH):
        return {"Phase 1": [], "Phase 2": []}
    
    with open(METADATA_PATH, 'r') as f:
        return json.load(f)

# Manual aliases for tricky matches
MANUAL_ALIASES = {
    "Washington and Point Happy Simon": "Washington St & Simon Dr/ Point Happy Way"
}

def get_intersection_metadata(name, corridor=None):
    metadata = load_cv_sync_intersection_metadata()
    
    # Check manual aliases first
    actual_name = MANUAL_ALIASES.get(name, name)
    norm_name = normalize_name(actual_name)
    
    # First, try to find in both phases
    all_locations = metadata["Phase 1"] + metadata["Phase 2"]
    
    # 1. Exact normalized name match
    matches = [loc for loc in all_locations if loc["normalized_name"] == norm_name]
    
    # 2. If corridor provided, try to filter
    if corridor:
        corridor_norm = corridor.lower().strip()
        corridor_matches = [m for m in matches if m.get("corridor") and corridor_norm in m["corridor"].lower()]
        if corridor_matches:
            return corridor_matches[0]
            
    # 3. Return first match if unique or at least one found
    if matches:
        return matches[0]
        
    return None

def format_intersection_display(name, metadata=None):
    if metadata is None:
        metadata = get_intersection_metadata(name)
        
    if metadata:
        number = metadata.get("intersection_number")
        phase = metadata.get("phase")
        p_short = "P1" if phase == "Phase 1" else "P2"
        
        if number and phase:
            return f"{p_short} #{number} — {name}"
        elif number:
            return f"#{number} — {name}"
            
    return name
