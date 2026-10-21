import pandas as pd
import re
import json
import os

FILE_PATH = "reference_data/CV_SYNC_Acceptance_Matrix_2.0_Phase_1_Phase_2_Locations_BOSCH_CAO_v2.xlsm"
OUTPUT_PATH = "reference_data/intersection_metadata.json"

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

def load_metadata():
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found.")
        return
    
    metadata = {
        "Phase 1": [],
        "Phase 2": []
    }
    
    try:
        # Phase 1: MATRIX TOOL
        print("Parsing Phase 1 (MATRIX TOOL)...")
        df1 = pd.read_excel(FILE_PATH, sheet_name="MATRIX TOOL", header=None)
        # Headers are at row 3 (0-indexed). Data starts at row 4.
        for i in range(4, len(df1)):
            row = df1.iloc[i]
            corridor = str(row[11]).strip() if pd.notna(row[11]) else None
            def clean_number(num):
                if pd.isna(num): return None
                s = str(num).strip()
                if s.endswith('.0'): s = s[:-2]
                return s

            number = clean_number(row[15])
            name = str(row[17]).strip() if pd.notna(row[17]) else None
            
            if not name or name.lower() == 'nan':
                continue
                
            metadata["Phase 1"].append({
                "official_name": name,
                "normalized_name": normalize_name(name),
                "intersection_number": number,
                "corridor": corridor,
                "phase": "Phase 1"
            })
            
        # Phase 2: Phase 2 Intersections
        print("Parsing Phase 2 (Phase 2 Intersections)...")
        df2 = pd.read_excel(FILE_PATH, sheet_name="Phase 2 Intersections", header=3)
        # Corridor: 0, Agency: 1, Intersection No.: 2, Project Intersection: 4
        for i in range(len(df2)):
            row = df2.iloc[i]
            corridor = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else None
            agency = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else None
            number = clean_number(row.iloc[2])
            name = str(row.iloc[4]).strip() if pd.notna(row.iloc[4]) else None
            
            if not name or name.lower() == 'nan':
                continue
                
            metadata["Phase 2"].append({
                "official_name": name,
                "normalized_name": normalize_name(name),
                "intersection_number": number,
                "corridor": corridor,
                "agency": agency,
                "phase": "Phase 2"
            })
            
        with open(OUTPUT_PATH, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Metadata saved to {OUTPUT_PATH}")
        print(f"Total Phase 1: {len(metadata['Phase 1'])}")
        print(f"Total Phase 2: {len(metadata['Phase 2'])}")
        
    except Exception as e:
        print(f"Error parsing workbook: {e}")

if __name__ == "__main__":
    load_metadata()
