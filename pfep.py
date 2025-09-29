import streamlit as st
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import math
import re
import io

# --- App Configuration ---
st.set_page_config(page_title="PFEP Analyser", layout="wide")

# --- 1. MASTER TEMPLATE AND LOGIC CONSTANTS ---
BASE_TEMPLATE_COLUMNS = [
    'SR.NO', 'PARTNO', 'PART DESCRIPTION', # Placeholder for dynamic Qty/Veh cols
    'UOM', 'ST.NO', 'FAMILY', # Placeholder for dynamic Qty/Veh_Daily cols
    'NET', 'UNIT PRICE', 'PART CLASSIFICATION',
    'L-MM_Size', 'W-MM_Size', 'H-MM_Size', 'Volume (m^3)', 'SIZE CLASSIFICATION', 'VENDOR CODE',
    'VENDOR NAME', 'VENDOR TYPE', 'CITY', 'STATE', 'COUNTRY', 'PINCODE', 'PRIMARY PACK TYPE',
    'L-MM_Prim_Pack', 'W-MM_Prim_Pack', 'H-MM_Prim_Pack', 'QTY/PACK_Prim', 'PRIM. PACK LIFESPAN',
    'PRIMARY PACKING FACTOR', 'SECONDARY PACK TYPE', 'L-MM_Sec_Pack', 'W-MM_Sec_Pack',
    'H-MM_Sec_Pack', 'NO OF BOXES', 'QTY/PACK_Sec', 'SEC. PACK LIFESPAN', 'ONE WAY/ RETURNABLE',
    'DISTANCE CODE', 'INVENTORY CLASSIFICATION', 'RM IN DAYS', 'RM IN QTY',
    'RM IN INR', 'PACKING FACTOR (PF)', 'NO OF SEC. REQD.', 'NO OF SEC REQ. AS PER PF',
    'WH LOC', 'PRIMARY LOCATION ID', 'SECONDARY LOCATION ID',
    'OVER FLOW TO BE ALLOTED', 'DOCK NUMBER', 'STACKING FACTOR', 'SUPPLY TYPE', 'SUPPLY VEH SET',
    'SUPPLY STRATEGY', 'SUPPLY CONDITION', 'CONTAINER LINE SIDE', 'L-MM_Supply', 'W-MM_Supply',
    'H-MM_Supply', 'Volume_Supply', 'QTY/CONTAINER -LS -9M', 'QTY/CONTAINER -LS-12M', 'STORAGE LINE SIDE',
    'L-MM_Line', 'W-MM_Line', 'H-MM_Line', 'Volume_Line', 'CONTAINER / RACK','NO OF TRIPS/DAY', 'INVENTORY LINE SIDE'
]
PFEP_COLUMN_MAP = { 'part_id': 'PARTNO', 'description': 'PART DESCRIPTION', 'net_daily_consumption': 'NET', 'unit_price': 'UNIT PRICE', 'vendor_code': 'VENDOR CODE', 'vendor_name': 'VENDOR NAME', 'city': 'CITY', 'state': 'STATE', 'country': 'COUNTRY', 'pincode': 'PINCODE', 'length': 'L-MM_Size', 'width': 'W-MM_Size', 'height': 'H-MM_Size', 'qty_per_pack': 'QTY/PACK_Sec', 'packing_factor': 'PACKING FACTOR (PF)', 'primary_packaging_factor': 'PRIMARY PACKING FACTOR', 'qty_per_pack_prim': 'QTY/PACK_Prim', 'one_way_returnable': 'ONE WAY/ RETURNABLE', 'primary_pack_type': 'PRIMARY PACK TYPE', 'supply_condition': 'SUPPLY CONDITION'}
INTERNAL_TO_PFEP_NEW_COLS = { 'family': 'FAMILY', 'part_classification': 'PART CLASSIFICATION', 'volume_m3': 'Volume (m^3)', 'size_classification': 'SIZE CLASSIFICATION', 'wh_loc': 'WH LOC', 'inventory_classification': 'INVENTORY CLASSIFICATION', 'prim_pack_lifespan': 'PRIM. PACK LIFESPAN', 'sec_pack_lifespan': 'SEC. PACK LIFESPAN'}
FAMILY_KEYWORD_MAPPING = { "ADAPTOR": ["ADAPTOR", "ADAPTER"], "Beading": ["BEADING"], "Electrical": ["BATTERY", "HVPDU", "ELECTRICAL", "INVERTER", "SENSOR", "DC", "COMPRESSOR", "TMCS", "COOLING", "BRAKE SIGNAL", "VCU", "VEHICLE CONTROL", "EVCC", "EBS ECU", "ECU", "CONTROL UNIT", "SIGNAL", "TRANSMITTER", "TRACTION", "HV", "KWH", "EBS", "SWITCH", "HORN"], "Electronics": ["DISPLAY", "APC", "SCREEN", "MICROPHONE", "CAMERA", "SPEAKER", "DASHBOARD", "ELECTRONICS", "SSD", "WOODWARD", "FDAS", "BDC", "GEN-2", "SENSOR", "BUZZER"], "Wheels": ["WHEEL", "TYRE", "TIRE", "RIM"], "Harness": ["HARNESS", "CABLE"], "Mechanical": ["PUMP", "SHAFT", "LINK", "GEAR", "ARM"], "Hardware": ["NUT", "BOLT", "SCREW", "WASHER", "RIVET", "M5", "M22", "M12", "CLAMP", "CLIP", "CABLE TIE", "DIN", "ZFP"], "Bracket": ["BRACKET", "BRKT", "BKT", "BRCKT"], "ASSY": ["ASSY"], "Sticker": ["STICKER", "LOGO", "EMBLEM"], "Suspension": ["SUSPENSION"], "Tank": ["TANK"], "Tape": ["TAPE", "REFLECTOR", "COLOUR"], "Tool Kit": ["TOOL KIT"], "Valve": ["VALVE"], "Hose": ["HOSE"], "Insulation": ["INSULATION"], "Interior & Exterior": ["ROLLER", "FIRE", "HAMMER"], "L-angle": ["L-ANGLE"], "Lamp": ["LAMP"], "Lock": ["LOCK"], "Lubricants": ["GREASE", "LUBRICANT"], "Medical": ["MEDICAL", "FIRST AID"], "Mirror": ["MIRROR", "ORVM"], "Motor": ["MOTOR"], "Mounting": ["MOUNT", "MTG", "MNTG", "MOUNTED"], "Oil": ["OIL"], "Panel": ["PANEL"], "Pillar": ["PILLAR"], "Pipe": ["PIPE", "TUBE", "SUCTION", "TUBULAR"], "Plate": ["PLATE"], "Plywood": ["FLOORING", "PLYWOOD", "EPGC"], "Profile": ["PROFILE", "ALUMINIUM"], "Rail": ["RAIL"], "Rubber": ["RUBBER", "GROMMET", "MOULDING"], "Seal": ["SEAL"], "Seat": ["SEAT"], "ABS Cover": ["ABS COVER"], "AC": ["AC"], "ACP Sheet": ["ACP SHEET"], "Aluminium": ["ALUMINIUM", "ALUMINUM"], "AXLE": ["AXLE"], "Bush": ["BUSH"], "Chassis": ["CHASSIS"], "Dome": ["DOME"], "Door": ["DOOR"], "Filter": ["FILTER"], "Flap": ["FLAP"], "FRP": ["FRP", "FACIA"], "Glass": ["GLASS", "WINDSHIELD", "WINDSHILED"], "Handle": ["HANDLE", "HAND", "PLASTIC"], "HATCH": ["HATCH"], "HDF Board": ["HDF"] }
CATEGORY_PRIORITY_FAMILIES = {"ACP Sheet", "ADAPTOR", "Bracket", "Bush", "Flap", "Handle", "Beading", "Lubricants", "Panel", "Pillar", "Rail", "Seal", "Sticker", "Valve"}
BASE_WAREHOUSE_MAPPING = { "ABS Cover": "HRR", "ADAPTOR": "MEZ B-01(A)", "Beading": "HRR", "AXLE": "FLOOR", "Bush": "HRR", "Chassis": "FLOOR", "Dome": "MEZ C-02(B)", "Door": "MRR(C-01)", "Electrical": "HRR", "Filter": "CRL", "Flap": "MEZ C-02", "Insulation": "MEZ C-02(B)", "Interior & Exterior": "HRR", "L-angle": "MEZ B-01(A)", "Lamp": "CRL", "Lock": "CRL", "Lubricants": "HRR", "Medical": "HRR", "Mirror": "HRR", "Motor": "HRR", "Mounting": "HRR", "Oil": "HRR", "Panel": "MEZ C-02", "Pillar": "MEZ C-02", "Pipe": "HRR", "Plate": "HRR", "Profile": "HRR", "Rail": "CTR(C-01)", "Seal": "HRR", "Seat": "MRR(C-01)", "Sticker": "MEZ B-01(A)", "Suspension": "MRR(C-01)", "Tank": "HRR", "Tool Kit": "HRR", "Valve": "CRL", "Wheels": "HRR", "Hardware": "MEZ B-02(A)", "Glass": "MRR(C-01)", "Harness": "HRR", "Hose": "HRR", "Aluminium": "HRR", "ACP Sheet": "MEZ C-02(B)", "Handle": "HRR", "HATCH": "HRR", "HDF Board": "MRR(C-01)", "FRP": "CTR", "Others": "HRR" }
GEOLOCATOR = Nominatim(user_agent="inventory_distance_calculator_streamlit_v11", timeout=10)

# --- 2. CORE DATA PROCESSING FUNCTIONS ---
@st.cache_data
def get_lat_lon(pincode, country="India", city="", state="", retries=3, backoff_factor=2):
    """
    More robustly geocodes a pincode by trying multiple query formats.
    """
    pincode_str = str(pincode).strip().split('.')[0]
    if not (pincode_str.isdigit() and len(pincode_str) == 6):
        return (None, None)

    queries_to_try = [
        {'postalcode': pincode_str, 'city': city, 'state': state, 'country': country},
        {'postalcode': pincode_str, 'state': state, 'country': country},
        {'postalcode': pincode_str, 'country': country},
        f"{pincode_str}, {city}, {state}, {country}",
        f"{pincode_str}, {country}"
    ]

    for attempt in range(retries):
        time.sleep(1) 
        for query in queries_to_try:
            try:
                location = GEOLOCATOR.geocode(query, exactly_one=True, timeout=10)
                if location:
                    return (location.latitude, location.longitude)
            except Exception as e:
                print(f"Geocoding attempt {attempt+1} for query '{query}' failed with error: {e}")
                break 

        if attempt < retries - 1:
            wait_time = backoff_factor * (attempt + 1)
            time.sleep(wait_time)

    # Do not show a warning for every single failure, as it clutters the UI. 
    # The main function will handle reporting.
    return (None, None)


def get_distance_code(distance):
    if pd.isna(distance): return None
    elif distance < 50: return 1
    elif distance <= 250: return 2
    elif distance <= 750: return 3
    else: return 4

def read_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.lower().endswith('.csv'): return pd.read_csv(uploaded_file, low_memory=False)
        elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')): return pd.read_excel(uploaded_file)
        st.warning(f"Unsupported file type: {uploaded_file.name}. Please use CSV or Excel.")
        return None
    except Exception as e:
        st.error(f"Error reading file {uploaded_file.name}: {e}")
        return None

def read_pfep_file(uploaded_file):
    try:
        return pd.read_excel(uploaded_file, header=1)
    except Exception as e:
        st.error(f"Error reading PFEP file {uploaded_file.name}: {e}")
        return None

def validate_and_parse_pfep(df, is_modify_workflow=False):
    """
    Validates the structure of a PFEP file. For the 'modify' workflow,
    it performs stricter checks for columns required for recalculations.
    """
    uploaded_cols = {str(col).lower().strip() for col in df.columns}
    
    # Basic structural checks for all workflows
    core_cols = {'partno', 'part description', 'family', 'net', 'total'}
    if not core_cols.issubset(uploaded_cols):
        missing = core_cols - uploaded_cols
        st.error(f"Validation Failed: The uploaded file is missing one or more structural columns: {', '.join(missing).upper()}.")
        return None, None, None

    # Stricter checks for the modify workflow
    if is_modify_workflow:
        required_for_recalc = {'unit price', 'pincode'}
        if not required_for_recalc.issubset(uploaded_cols):
            missing = required_for_recalc - uploaded_cols
            st.error(f"Validation Failed for 'Modify' Workflow: To recalculate inventory norms, the uploaded PFEP must contain the following columns: {', '.join(missing).upper()}. Please add them and re-upload.")
            return None, None, None

    # Identify vehicle-specific columns based on their position in the template
    try:
        # Use a case-insensitive search for column indices
        part_desc_idx = [i for i, col in enumerate(df.columns) if str(col).lower().strip() == 'part description'][0]
        total_idx = [i for i, col in enumerate(df.columns) if str(col).lower().strip() == 'total'][0]
        vehicle_qty_pfep_cols = df.columns[part_desc_idx + 1 : total_idx].tolist()
    except IndexError:
        st.error("Validation Failed: Could not find 'PART DESCRIPTION' or 'TOTAL' columns to identify vehicle types.")
        return None, None, None

    all_mappings = {**PFEP_COLUMN_MAP, **INTERNAL_TO_PFEP_NEW_COLS}
    reverse_map = {v: k for k, v in all_mappings.items()}
    df.rename(columns=reverse_map, inplace=True)

    vehicle_configs, internal_qty_cols = [], []
    for i, col_name in enumerate(vehicle_qty_pfep_cols):
        internal_name = f"qty_veh_{i}"
        df.rename(columns={col_name: internal_name}, inplace=True)
        internal_qty_cols.append(internal_name)
        vehicle_configs.append({"name": col_name, "multiplier": 1.0})

    if 'part_id' in df.columns:
        df['part_id'] = df['part_id'].astype(str)

    st.success(f"✅ PFEP file validated successfully. Found {len(vehicle_configs)} vehicle types.")
    return df, sorted(internal_qty_cols), vehicle_configs


def find_and_rename_columns(df):
    rename_dict, found_keys = {}, []
    for internal_key, pfep_name in PFEP_COLUMN_MAP.items():
        for col in df.columns:
            if str(col).lower().strip() == pfep_name.lower():
                rename_dict[col] = internal_key
                found_keys.append(internal_key)
                break
    qty_veh_regex = re.compile(r'(qty|quantity)[\s_/]?p?e?r?[\s_/]?veh(icle)?', re.IGNORECASE)
    qty_veh_cols = [col for col in df.columns if qty_veh_regex.search(str(col))]
    for original_col in qty_veh_cols:
        if original_col not in rename_dict:
            temp_name = f"qty_veh_temp_{original_col}".replace(" ", "_")
            rename_dict[original_col] = temp_name
            found_keys.append(f"{temp_name} (from {original_col})")
    df.rename(columns=rename_dict, inplace=True)
    if found_keys: st.info(f"   Found and mapped columns: {found_keys}")
    else: st.warning("   Could not automatically map any standard columns.")
    return df

def _consolidate_bom_list(bom_list):
    valid_boms = [df for df in bom_list if df is not None and 'part_id' in df.columns]
    if not valid_boms: return None
    
    master = pd.concat(valid_boms, ignore_index=True)
    
    qty_cols = [col for col in master.columns if 'qty_veh_temp' in col]
    other_cols = [col for col in master.columns if col not in qty_cols and col != 'part_id']
    
    agg_dict = {col: 'sum' for col in qty_cols}
    agg_dict.update({col: 'first' for col in other_cols})
        
    master[qty_cols] = master[qty_cols].fillna(0)
    master['part_id'] = master['part_id'].astype(str)
    master = master.groupby('part_id').agg(agg_dict).reset_index()
    
    for col in qty_cols:
        master[col] = master[col].replace(0, np.nan)
        
    return master

def _merge_supplementary_df(main_df, new_df):
    if 'part_id' not in new_df.columns: return main_df
    
    main_df['part_id'] = main_df['part_id'].astype(str)
    new_df['part_id'] = new_df['part_id'].astype(str)

    if 'part_id' in main_df.columns: main_df = main_df.set_index('part_id')
    else:
        st.error("Error: 'part_id' not found in main DataFrame for merging.")
        return main_df
    new_df.drop_duplicates(subset=['part_id'], keep='first', inplace=True)
    new_df = new_df.set_index('part_id')
    
    main_df.update(new_df)
    new_cols = new_df.columns.difference(main_df.columns)
    main_df = main_df.join(new_df[new_cols])
    
    return main_df.reset_index()

def _merge_vendor_df(main_df, vendor_df):
    if 'vendor_code' not in main_df.columns or 'vendor_code' not in vendor_df.columns:
        st.warning("Skipping a vendor file merge: 'vendor_code' column not found in both the base BOM and the vendor master file.")
        return main_df

    main_df['vendor_code'] = main_df['vendor_code'].astype(str)
    vendor_df['vendor_code'] = vendor_df['vendor_code'].astype(str)
    vendor_df.drop_duplicates(subset=['vendor_code'], keep='first', inplace=True)
    
    merged_df = pd.merge(main_df, vendor_df, on='vendor_code', how='left', suffixes=('_existing', ''))
    
    for col in vendor_df.columns:
        if col == 'vendor_code': continue
        
        existing_col_name = f"{col}_existing"
        if existing_col_name in merged_df.columns:
            merged_df[col] = merged_df[col].fillna(merged_df[existing_col_name])
            merged_df.drop(columns=[existing_col_name], inplace=True)
            
    return merged_df

def load_all_files(uploaded_files):
    file_types = { "pbom": [], "mbom": [], "part_attribute": [], "packaging": [], "vendor_master": [] }
    st.session_state['source_files_for_report'] = { "pbom": [], "mbom": [], "part_attribute": [], "packaging": [], "vendor_master": [] }
    
    with st.spinner("Processing uploaded files..."):
        for key, files in uploaded_files.items():
            if not files: continue
            file_list = files if isinstance(files, list) else [files]
            for f in file_list:
                df = read_uploaded_file(f)
                if df is not None:
                    st.session_state['source_files_for_report'][key].append(df.copy())
                    processed_df = find_and_rename_columns(df)
                    
                    if key == 'mbom' and 'supply_condition' in processed_df.columns:
                        initial_count = len(processed_df)
                        supply_conditions_lower = processed_df['supply_condition'].astype(str).str.lower()
                        mask_to_remove = (supply_conditions_lower.str.contains('inhouse', na=False) |
                                          supply_conditions_lower.str.contains('make', na=False) |
                                          (supply_conditions_lower == 'e'))
                        processed_df = processed_df[~mask_to_remove]
                        
                        removed_count = initial_count - len(processed_df)
                        if removed_count > 0:
                            st.info(f"   Removed {removed_count} parts from an MBOM file marked as 'Inhouse', 'Make', or 'E'.")

                    file_types[key].append(processed_df)
    return file_types

def finalize_master_df(base_bom_df, supplementary_dfs):
    with st.spinner("Consolidating final dataset..."):
        final_df = base_bom_df
        part_attr_dfs, vendor_master_dfs, packaging_dfs = supplementary_dfs

        for df in part_attr_dfs + packaging_dfs:
            if df is not None and 'part_id' in df.columns:
                final_df = _merge_supplementary_df(final_df, df)

        for df in vendor_master_dfs:
            if df is not None:
                final_df = _merge_vendor_df(final_df, df)
        
        final_df.drop_duplicates(subset=['part_id'], keep='first', inplace=True)
        
        detected_qty_cols = sorted([col for col in final_df.columns if 'qty_veh_temp_' in str(col)])
        rename_map = {old_name: f"qty_veh_{i}" for i, old_name in enumerate(detected_qty_cols)}
        final_df.rename(columns=rename_map, inplace=True)
        final_qty_cols = sorted(rename_map.values())
        
        for col in final_qty_cols:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
            
        st.success(f"Consolidated base has {final_df['part_id'].nunique()} unique parts.")
        st.success(f"Detected {len(final_qty_cols)} unique 'Quantity per Vehicle' columns.")
        
        return final_df, final_qty_cols

# --- 3. CLASSIFICATION AND PROCESSING CLASSES ---
class PartClassificationSystem:
    def __init__(self):
        self.percentages = {'C': {'target': 60}, 'B': {'target': 25}, 'A': {'target': 12}, 'AA': {'target': 3}}
        self.calculated_ranges = {}

    def calculate_percentage_ranges(self, df, price_column):
        valid_prices = pd.to_numeric(df[price_column], errors='coerce').dropna().sort_values(ascending=False)
        if valid_prices.empty:
            st.warning("No valid prices found to calculate part classification ranges.")
            return
        
        total_valid_parts = len(valid_prices)
        st.write(f"Calculating classification ranges from {total_valid_parts} valid prices...")
        
        ranges, current_idx = {}, 0
        processing_order = ['AA', 'A', 'B', 'C']
        
        for class_name in processing_order:
            if current_idx >= total_valid_parts: break
            if class_name == 'C':
                ranges[class_name] = {'min': 0, 'max': valid_prices.iloc[current_idx]}
                continue

            details = self.percentages[class_name]
            count = round(total_valid_parts * (details['target'] / 100))
            end_idx = min(current_idx + count, total_valid_parts)
            
            if current_idx < end_idx:
                min_val = valid_prices.iloc[end_idx - 1]
                max_val = valid_prices.iloc[current_idx]
                ranges[class_name] = {'min': min_val, 'max': max_val}
            
            current_idx = end_idx
            
        self.calculated_ranges = {k: ranges.get(k, {'min': 0, 'max': 0}) for k in self.percentages.keys()}
        st.write("   Ranges calculated successfully.")

    def classify_part(self, unit_price):
        if pd.isna(unit_price): return np.nan
        if not self.calculated_ranges: return 'Unclassified'
        
        if unit_price >= self.calculated_ranges.get('AA', {'min': float('inf')})['min']: return 'AA'
        if unit_price >= self.calculated_ranges.get('A', {'min': float('inf')})['min']: return 'A'
        if unit_price >= self.calculated_ranges.get('B', {'min': float('inf')})['min']: return 'B'
        return 'C'

    def classify_all_parts(self, df, price_column):
        self.calculate_percentage_ranges(df, price_column)
        return df[price_column].apply(self.classify_part)

class ComprehensiveInventoryProcessor:
    def __init__(self, initial_data):
        self.data = initial_data.copy()
        self.rm_days_mapping = {'A1': 4, 'A2': 6, 'A3': 8, 'A4': 11, 'B1': 6, 'B2': 11, 'B3': 13, 'B4': 16, 'C1': 16, 'C2': 31}
        self.classifier = PartClassificationSystem()

    def calculate_dynamic_consumption(self, qty_cols, multipliers):
        st.subheader("Calculating Daily & Net Consumption")
        daily_cols = []
        for col in qty_cols:
            if col not in self.data.columns: self.data[col] = np.nan
        
        for i, col in enumerate(qty_cols):
            daily_col_name = f"{col}_daily"
            self.data[daily_col_name] = self.data[col] * multipliers[i]
            daily_cols.append(daily_col_name)

        self.data['TOTAL'] = self.data[qty_cols].sum(axis=1, skipna=True) if qty_cols else 0
        self.data['net_daily_consumption'] = self.data[daily_cols].sum(axis=1, skipna=True) if daily_cols else 0
        st.success("Consumption calculated.")
        return self.data

    def run_family_classification(self):
        st.subheader("(A) Family Classification")
        if 'description' not in self.data.columns:
            self.data['family'] = 'Others'
            return
        def extract_family(desc):
            if pd.isna(desc): return 'Others'
            desc_upper = str(desc).upper()
            
            def find_pos(kw):
                match = re.search(r'\b' + re.escape(kw) + r'\b', desc_upper)
                return match.start() if match else -1

            for fam in CATEGORY_PRIORITY_FAMILIES:
                if fam in FAMILY_KEYWORD_MAPPING and any(find_pos(kw) != -1 for kw in FAMILY_KEYWORD_MAPPING[fam]):
                    return fam

            matches = ((pos, fam) for fam, kws in FAMILY_KEYWORD_MAPPING.items() if fam not in CATEGORY_PRIORITY_FAMILIES
                       for kw in kws for pos in (find_pos(kw),) if pos != -1)
            
            try:
                first_match = min(matches, key=lambda x: x[0])
                return first_match[1]
            except ValueError:
                return 'Others'

        self.data['family'] = self.data['description'].apply(extract_family)
        st.success("✅ Automated family classification complete.")

    def run_size_classification(self):
        st.subheader("(B) Size Classification")
        size_cols = ['length', 'width', 'height']
        if not all(k in self.data.columns for k in size_cols):
            self.data['volume_m3'], self.data['size_classification'] = np.nan, np.nan
            return
        for col in size_cols: self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
        self.data['volume_m3'] = (self.data['length'] * self.data['width'] * self.data['height']) / 1_000_000_000
        
        def classify_size(row):
            if pd.isna(row['volume_m3']): return np.nan
            dims = [d for d in [row['length'], row['width'], row['height']] if pd.notna(d)]
            if not dims: return np.nan
            
            max_dim = max(dims)
            if row['volume_m3'] > 1.5 or max_dim > 1200: return 'XL'
            if 0.5 < row['volume_m3'] <= 1.5 or 750 < max_dim <= 1200: return 'L'
            if 0.05 < row['volume_m3'] <= 0.5 or 150 < max_dim <= 750: return 'M'
            return 'S'
        self.data['size_classification'] = self.data.apply(classify_size, axis=1)
        st.success("✅ Automated size classification complete.")

    def run_part_classification(self):
        st.subheader("(C) Part Classification")
        if 'unit_price' not in self.data.columns:
            self.data['part_classification'] = np.nan
            st.warning("'unit_price' column not found. Skipping part classification.")
            return

        self.data['unit_price'] = pd.to_numeric(self.data['unit_price'], errors='coerce')
        self.data['part_classification'] = self.classifier.classify_all_parts(self.data, 'unit_price')
        st.success("✅ Percentage-based part classification complete.")

    def run_packaging_classification(self):
        st.subheader("(D) Packaging Classification & Lifespan")
        if 'primary_pack_type' not in self.data.columns:
            self.data['one_way_returnable'] = np.nan
        else:
            returnable_keywords = ['metallic pallet', 'collapsible box', 'bucket', 'plastic bin', 'trolley', 'plastic pallet', 'bin a', 'mesh bin', 'drum']
            one_way_keywords = ['bubble wrap', 'carton box', 'gunny bag', 'polybag', 'stretch wrap', 'wooden box', 'open', 'wooden pallet', 'foam', 'plastic bag']
            def classify_pack(pack_type):
                if pd.isna(pack_type): return np.nan
                pack_type_lower = str(pack_type).lower()
                if any(keyword in pack_type_lower for keyword in returnable_keywords): return 'Returnable'
                if any(keyword in pack_type_lower for keyword in one_way_keywords): return 'One Way'
                return np.nan
            self.data['one_way_returnable'] = self.data['primary_pack_type'].apply(classify_pack)
            st.success("✅ Automated packaging type classification complete.")

        net_daily = pd.to_numeric(self.data['net_daily_consumption'], errors='coerce')
        if 'qty_per_pack_prim' in self.data.columns:
            qty_prim = pd.to_numeric(self.data['qty_per_pack_prim'], errors='coerce')
            self.data['prim_pack_lifespan'] = np.divide(qty_prim, net_daily, out=np.full_like(qty_prim, np.nan, dtype=float), where=net_daily!=0)
        if 'qty_per_pack' in self.data.columns:
            qty_sec = pd.to_numeric(self.data['qty_per_pack'], errors='coerce')
            self.data['sec_pack_lifespan'] = np.divide(qty_sec, net_daily, out=np.full_like(qty_sec, np.nan, dtype=float), where=net_daily!=0)
        st.success("✅ Package lifespan calculation complete.")

    def run_location_based_norms(self, pincode):
        st.subheader(f"(E) Distance & Inventory Norms")
        
        # --- DEFENSIVE CHECK ---
        required_cols = ['pincode', 'part_classification', 'net_daily_consumption', 'unit_price']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            st.error(f"Cannot calculate inventory norms. The data is missing the following required columns: {', '.join(missing_cols)}")
            # Fill output columns with NaN to avoid downstream errors
            for col in ['distance_km', 'DISTANCE CODE', 'inventory_classification', 'RM IN DAYS', 'RM IN QTY', 'RM IN INR', 'NO OF SEC. REQD.', 'NO OF SEC REQ. AS PER PF']:
                self.data[col] = np.nan
            return

        with st.spinner(f"Getting coordinates for your location pincode: {pincode}..."):
            current_coords = get_lat_lon(pincode, country="India")
        if current_coords == (None, None):
            st.error(f"CRITICAL: Could not find coordinates for your pincode {pincode}. Distances cannot be calculated.")
            return

        # Prepare a unique list of vendor locations to geocode
        vendor_locations = self.data[['pincode', 'city', 'state']].drop_duplicates().dropna(subset=['pincode'])
        location_cache = {}
        
        status_placeholder = st.sidebar.empty()
        st.sidebar.info("Geocoding vendor locations...")
        
        failed_geocodes = 0
        total_locations = len(vendor_locations)
        for i, row in enumerate(vendor_locations.itertuples()):
            status_placeholder.info(f"Geocoding vendor {i+1} of {total_locations}...")
            # Use tuple of location data as cache key
            cache_key = (row.pincode, row.city, row.state)
            if cache_key not in location_cache:
                coords = get_lat_lon(row.pincode, city=str(row.city), state=str(row.state))
                location_cache[cache_key] = geodesic(current_coords, coords).km if coords[0] is not None else np.nan
                if pd.isna(location_cache[cache_key]):
                    failed_geocodes += 1

        status_placeholder.empty()
        if failed_geocodes > 0:
            st.sidebar.warning(f"Geocoding failed for {failed_geocodes} out of {total_locations} vendor locations. Those parts will have blank distance norms.")
        else:
            st.sidebar.success(f"Successfully geocoded all {total_locations} vendor locations.")

        # Map the calculated distances back to the main dataframe
        def map_distance(row):
            cache_key = (row.get('pincode'), row.get('city'), row.get('state'))
            return location_cache.get(cache_key, np.nan)
            
        self.data['distance_km'] = self.data.apply(map_distance, axis=1)
        
        self.data['DISTANCE CODE'] = self.data['distance_km'].apply(get_distance_code)
        def get_inv_class(p, d):
            if pd.isna(p) or pd.isna(d): return None
            d = int(d)
            if p in ['AA', 'A']: return f"A{d}"
            if p == 'B': return f"B{d}"
            if p == 'C': return 'C1' if d in [1, 2] else 'C2'
            return None
        self.data['inventory_classification'] = self.data.apply(lambda r: get_inv_class(r.get('part_classification'), r.get('DISTANCE CODE')), axis=1)
        self.data['RM IN DAYS'] = self.data['inventory_classification'].map(self.rm_days_mapping)
        self.data['RM IN QTY'] = self.data['RM IN DAYS'] * self.data['net_daily_consumption']
        self.data['RM IN INR'] = self.data['RM IN QTY'] * pd.to_numeric(self.data['unit_price'], errors='coerce')
        
        qty_per_pack = pd.to_numeric(self.data.get('qty_per_pack'), errors='coerce').fillna(1).replace(0, 1)
        packing_factor = pd.to_numeric(self.data.get('packing_factor'), errors='coerce').fillna(1)
        self.data['NO OF SEC. REQD.'] = np.ceil(self.data['RM IN QTY'] / qty_per_pack)
        self.data['NO OF SEC REQ. AS PER PF'] = np.ceil(self.data['NO OF SEC. REQD.'] * packing_factor)
        st.success(f"✅ Inventory norms calculated.")

    def run_warehouse_location_assignment(self):
        st.subheader("(F) Warehouse Location Assignment")
        if 'family' not in self.data.columns:
            self.data['wh_loc'] = 'HRR'
        else:
            def get_wh_loc(row):
                fam, desc, vol_m3 = row.get('family', 'Others'), row.get('description', ''), row.get('volume_m3', None)
                match = lambda w: re.search(r'\b' + re.escape(w) + r'\b', str(desc).upper())
                if fam == "AC" and match("BCS"): return "OUTSIDE"
                if fam in ["ASSY", "Bracket"] and match("STEERING"): return "DIRECT FROM INSTOR"
                if fam == "Electronics" and any(match(k) for k in ["CAMERA", "APC", "MNVR", "WOODWARD"]): return "CRL"
                if fam == "Electrical" and vol_m3 is not None and (vol_m3 * 1_000_000) > 200: return "HRR"
                if fam == "Mechanical" and match("STEERING"): return "DIRECT FROM INSTOR"
                if fam == "Plywood" and not match("EDGE"): return "MRR(C-01)"
                if fam == "Rubber" and match("GROMMET"): return "MEZ B-01"
                if fam == "Tape" and not match("BUTYL"): return "MEZ B-01"
                if fam == "Wheels":
                    if match("TYRE") and match("JK"): return "OUTSIDE"
                    if match("RIM"): return "MRR(C-01)"
                return BASE_WAREHOUSE_MAPPING.get(fam, "HRR")
            self.data['wh_loc'] = self.data.apply(get_wh_loc, axis=1)
        
        loc_expansion_map = { 'HRR': 'High Rise Rack (HRR)', 'CRL': 'Carousal (CRL)', 'MEZ': 'Mezzanine (MEZ)', 'CTR': 'Cantilever (CTR)', 'MRR': 'Mid Rise Rack (MRR)' }
        for short, long in loc_expansion_map.items():
            self.data['wh_loc'] = self.data['wh_loc'].astype(str).str.replace(short, long, regex=False)
        st.success("✅ Automated warehouse location assignment complete.")

# --- 4. UI AND REPORTING FUNCTIONS ---
def create_formatted_excel_output(df, vehicle_configs, source_files_dict=None):
    st.subheader("(G) Generating Formatted Excel Report")
    
    final_df = df.copy()
    rename_map = {**PFEP_COLUMN_MAP, **INTERNAL_TO_PFEP_NEW_COLS, 'TOTAL': 'TOTAL'}
    
    qty_veh_cols, qty_veh_daily_cols = [], []
    vehicle_configs = vehicle_configs if isinstance(vehicle_configs, list) else []
    daily_consumption_values = {}
    for i, config in enumerate(vehicle_configs):
        rename_map[f"qty_veh_{i}"] = config['name']
        rename_map[f"qty_veh_{i}_daily"] = f"{config['name']}_Daily"
        qty_veh_cols.append(config['name'])
        qty_veh_daily_cols.append(f"{config['name']}_Daily")
        daily_consumption_values[config['name']] = config.get('multiplier', 0)

    final_df.rename(columns={k: v for k, v in rename_map.items() if k in final_df.columns}, inplace=True)

    template = [col for col in BASE_TEMPLATE_COLUMNS if '#' not in col]
    part_desc_idx = template.index('PART DESCRIPTION')
    family_idx = template.index('FAMILY')
    
    final_template = (template[:part_desc_idx + 1] + qty_veh_cols + ['TOTAL'] +
                      template[part_desc_idx + 1:family_idx + 1] + qty_veh_daily_cols +
                      template[family_idx + 1:])

    for col in final_template:
        if col not in final_df.columns: final_df[col] = np.nan
    final_df = final_df[final_template]
    final_df['SR.NO'] = range(1, len(final_df) + 1)

    part_class_counts = df['part_classification'].value_counts().reindex(['AA', 'A', 'B', 'C']).fillna(0)
    total_classified = part_class_counts.sum()
    part_class_percentages = (part_class_counts / total_classified) if total_classified > 0 else part_class_counts
    
    size_counts = df['size_classification'].value_counts().reindex(['XL', 'L', 'M', 'S']).fillna(0)
    packaging_counts = df['one_way_returnable'].value_counts().reindex(['One Way', 'Returnable']).fillna(0)
    wh_loc_counts = df['wh_loc'].value_counts()
    
    with st.spinner("Creating the final Excel workbook with all source files..."):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet('Master Data Sheet')
            
            h_gray = workbook.add_format({'bold': True, 'text_wrap': True, 'valign': 'top', 'align': 'center', 'fg_color': '#D9D9D9', 'border': 1})
            s_orange = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#FDE9D9', 'border': 1})
            s_blue = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'fg_color': '#DCE6F1', 'border': 1})
            header_title_format = workbook.add_format({'bold': True, 'align': 'center', 'valign': 'vcenter', 'border': 1, 'fg_color': '#D9D9D9'})
            header_label_format = workbook.add_format({'bold': True, 'align': 'center', 'border': 1})
            header_value_format = workbook.add_format({'align': 'center', 'border': 1})
            percentage_format = workbook.add_format({'align': 'center', 'border': 1, 'num_format': '0.0%'})
            
            worksheet.merge_range('J5:K5', 'Daily consumption', header_title_format)
            veh_names = list(daily_consumption_values.keys())
            if len(veh_names) >= 2:
                worksheet.write('J6', veh_names[0], header_label_format); worksheet.write('K6', veh_names[1], header_label_format)
                worksheet.write('J7', daily_consumption_values.get(veh_names[0], 0), header_value_format); worksheet.write('K7', daily_consumption_values.get(veh_names[1], 0), header_value_format)
            elif len(veh_names) == 1:
                 worksheet.write('J6', veh_names[0], header_label_format); worksheet.write('K6', "", header_label_format)
                 worksheet.write('J7', daily_consumption_values.get(veh_names[0], 0), header_value_format); worksheet.write('K7', "", header_value_format)

            worksheet.merge_range('M4:Q4', 'Part Classification Assumption', header_title_format)
            worksheet.write('M5', 'Class', header_label_format); worksheet.write('N5', 'Calculated Range (INR)', header_label_format)
            worksheet.write('O5', '% Target', header_label_format); worksheet.write('P5', 'Count', header_label_format)
            worksheet.write('Q5', 'Actual %', header_label_format)
            
            calculated_ranges = st.session_state.processor.classifier.calculated_ranges
            def format_range(class_name):
                if not calculated_ranges or class_name not in calculated_ranges: return "N/A"
                range_data = calculated_ranges[class_name]
                min_val, max_val = range_data.get('min', 0), range_data.get('max', 0)
                if class_name == 'C': return f"< {max_val:,.2f}"
                if class_name == 'AA' and max_val == min_val: return f">= {min_val:,.2f}"
                return f"{min_val:,.2f} - {max_val:,.2f}"

            row_map = {'AA': 6, 'A': 7, 'B': 8, 'C': 9}
            for code, row_num in row_map.items():
                row_idx = row_num - 1
                worksheet.write(row_idx, 12, code, header_label_format); worksheet.write(row_idx, 13, format_range(code), header_value_format)
                worksheet.write(row_idx, 14, st.session_state.processor.classifier.percentages[code]['target'] / 100, percentage_format)
                worksheet.write(row_idx, 15, part_class_counts.get(code, 0), header_value_format); worksheet.write(row_idx, 16, part_class_percentages.get(code, 0), percentage_format)
            
            worksheet.write('P10', total_classified, header_value_format); worksheet.write('Q10', 1 if total_classified > 0 else 0, percentage_format)

            worksheet.merge_range('S4:T4', 'Size Classification Count', header_title_format)
            worksheet.write('S5', 'Size', header_label_format); worksheet.write('T5', 'Count', header_label_format)
            size_row_map = {'XL': 6, 'L': 7, 'M': 8, 'S': 9}
            for size, row_num in size_row_map.items():
                row_idx = row_num - 1
                worksheet.write(row_idx, 18, size, header_label_format); worksheet.write(row_idx, 19, int(size_counts.get(size, 0)), header_value_format)
            worksheet.write('T10', int(size_counts.sum()), header_value_format)

            worksheet.merge_range('V4:W4', 'Packaging Type Count', header_title_format)
            worksheet.write('V5', 'Type', header_label_format); worksheet.write('W5', 'Count', header_label_format)
            pack_row_map = {'One Way': 6, 'Returnable': 7}
            for pack_type, row_num in pack_row_map.items():
                row_idx = row_num - 1
                worksheet.write(row_idx, 21, pack_type, header_label_format); worksheet.write(row_idx, 22, int(packaging_counts.get(pack_type, 0)), header_value_format)

            worksheet.merge_range('Y4:Z4', 'Warehouse Location Count', header_title_format)
            worksheet.write('Y5', 'Location', header_label_format); worksheet.write('Z5', 'Count', header_label_format)
            current_row = 6
            for location, count in wh_loc_counts.items():
                if current_row > 11:
                    worksheet.write(current_row - 1, 24, '...and more', header_value_format); break
                row_idx = current_row - 1
                worksheet.write(row_idx, 24, location, header_value_format); worksheet.write(row_idx, 25, count, header_value_format)
                current_row += 1

            final_df.to_excel(writer, sheet_name='Master Data Sheet', startrow=12, header=False, index=False)
            
            final_columns_list = final_df.columns.tolist()
            first_daily_col = qty_veh_daily_cols[0] if qty_veh_daily_cols else 'NET'
            
            headers_info = [
                {'title': 'PART DETAILS', 'start': 'SR.NO', 'end': 'FAMILY', 'style': h_gray}, {'title': 'Daily consumption', 'start': first_daily_col, 'end': 'NET', 'style': s_orange},
                {'title': 'PRICE & CLASSIFICATION', 'start': 'UNIT PRICE', 'end': 'PART CLASSIFICATION', 'style': s_orange}, {'title': 'Size & Classification', 'start': 'L-MM_Size', 'end': 'SIZE CLASSIFICATION', 'style': s_orange},
                {'title': 'VENDOR DETAILS', 'start': 'VENDOR CODE', 'end': 'PINCODE', 'style': s_blue}, {'title': 'PACKAGING DETAILS', 'start': 'PRIMARY PACK TYPE', 'end': 'ONE WAY/ RETURNABLE', 'style': s_orange},
                {'title': 'INVENTORY NORM', 'start': 'DISTANCE CODE', 'end': 'NO OF SEC REQ. AS PER PF', 'style': s_blue}, {'title': 'WH STORAGE', 'start': 'WH LOC', 'end': 'STACKING FACTOR', 'style': s_orange},
                {'title': 'SUPPLY SYSTEM', 'start': 'SUPPLY TYPE', 'end': 'SUPPLY CONDITION', 'style': s_blue}, {'title': 'LINE SIDE STORAGE', 'start': 'CONTAINER LINE SIDE', 'end': 'INVENTORY LINE SIDE', 'style': h_gray}
            ]

            for header in headers_info:
                try:
                    start_idx = final_columns_list.index(header['start']); end_idx = final_columns_list.index(header['end'])
                    if start_idx <= end_idx:
                        if start_idx == end_idx: worksheet.write(11, start_idx, header['title'], header['style'])
                        else: worksheet.merge_range(11, start_idx, 11, end_idx, header['title'], header['style'])
                except ValueError:
                    st.warning(f"A column for header '{header['title']}' was not found. Skipping header.")

            for col_num, value in enumerate(final_columns_list):
                worksheet.write(12, col_num, value, h_gray)
            worksheet.set_column('A:A', 6); worksheet.set_column('B:C', 22); worksheet.set_column('D:ZZ', 18)

            if source_files_dict:
                for file_category, df_list in source_files_dict.items():
                    if not df_list: continue
                    if len(df_list) == 1:
                        sheet_name = f"Source_{file_category.replace('_', ' ').title()}"[:31]
                        df_list[0].to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        for i, source_df in enumerate(df_list):
                            sheet_name = f"Source_{file_category.title()}_{i+1}"[:31]
                            source_df.to_excel(writer, sheet_name=sheet_name, index=False)

        processed_data = output.getvalue()
    st.success(f"✅ Successfully created formatted Excel file with all source data!")
    return processed_data

def render_review_step(step_name, internal_key, next_stage):
    st.markdown("---")
    st.header(f"Manual Review: {step_name}")
    st.info(f"The automated {step_name.lower()} is complete. Review, download, modify, and upload to override.")
    
    pfep_name = INTERNAL_TO_PFEP_NEW_COLS.get(internal_key, PFEP_COLUMN_MAP.get(internal_key, internal_key))
    
    review_cols = ['part_id', 'description']
    if internal_key == 'one_way_returnable': review_cols.append('primary_pack_type')
    review_cols.append(internal_key)

    existing_cols = [c for c in review_cols if c in st.session_state.master_df.columns]
    review_df = st.session_state.master_df[existing_cols].copy()
    
    display_rename_map = {
        internal_key: pfep_name, 'part_id': 'PARTNO', 'description': 'PART DESCRIPTION',
        'primary_pack_type': 'PRIMARY PACK TYPE'
    }
    review_df.rename(columns=display_rename_map, inplace=True)
    
    st.dataframe(review_df.head(20))
    
    csv_data = review_df.to_csv(index=False).encode('utf-8')
    st.download_button(label=f"📥 Download {step_name} Data for Review", data=csv_data, file_name=f"manual_review_{internal_key}.csv", mime='text/csv')
    
    st.markdown("---")
    uploaded_file = st.file_uploader(f"Upload Modified {step_name} File Here", type=['csv', 'xlsx'], key=f"upload_{internal_key}")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"Apply Changes & Continue", disabled=not uploaded_file, type="primary", key=f"apply_{internal_key}"):
            modified_df = read_uploaded_file(uploaded_file)
            if modified_df is not None and 'PARTNO' in modified_df.columns and pfep_name in modified_df.columns:
                modified_df.rename(columns={pfep_name: internal_key, 'PARTNO': 'part_id'}, inplace=True)
                st.session_state.master_df = _merge_supplementary_df(st.session_state.master_df, modified_df[['part_id', internal_key]])
                st.success(f"✅ Manual changes for {step_name} applied successfully!")
                st.session_state.app_stage = next_stage
                st.rerun()
            else:
                st.error(f"Upload failed. File must contain 'PARTNO' and '{pfep_name}' columns.")
    with col2:
        if st.button(f"Skip & Continue", key=f"skip_{internal_key}"):
            st.session_state.app_stage = next_stage
            st.rerun()

# --- 5. MAIN APPLICATION WORKFLOW ---
def main():
    st.title("🏭 PFEP (Plan For Each Part) ANALYSER")

    for key in ['app_stage', 'master_df', 'qty_cols', 'final_report', 'processor', 'all_files', 'vehicle_configs', 'pincode', 'workflow_mode', 'source_files_for_report']:
        if key not in st.session_state:
            st.session_state[key] = None if key != 'app_stage' else 'welcome'
    
    if st.session_state.app_stage == "welcome":
        st.header("What would you like to do?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Create New PFEP", use_container_width=True, type="primary"):
                st.session_state.workflow_mode = 'create'
                st.session_state.app_stage = 'upload'
                st.rerun()
        with col2:
            if st.button("🔄 Modify Existing PFEP", use_container_width=True):
                st.session_state.workflow_mode = 'modify'
                st.session_state.app_stage = 'modify_upload'
                st.rerun()

    elif st.session_state.app_stage == "modify_upload":
        st.header("Step 1: Upload and Validate Existing PFEP")
        st.info("Upload your existing PFEP Excel file. The application will check if the headers match the standard structure before proceeding.")
        
        uploaded_pfep = st.file_uploader("Upload PFEP file", type=['xlsx'], label_visibility="collapsed")
        pincode = st.text_input("Enter your location's pincode for distance calculations", value="411001")

        if uploaded_pfep and st.button("Validate and Proceed", type="primary"):
            with st.spinner("Validating PFEP file structure..."):
                uploaded_pfep_bytes = uploaded_pfep.getvalue()
                pfep_df_for_process = read_pfep_file(io.BytesIO(uploaded_pfep_bytes))
                
                try:
                    raw_pfep_df = pd.read_excel(io.BytesIO(uploaded_pfep_bytes))
                    st.session_state['source_files_for_report'] = {'Original_PFEP': [raw_pfep_df]}
                except Exception as e:
                    st.warning(f"Could not store the original PFEP for the final report due to a reading error: {e}")

                if pfep_df_for_process is not None:
                    # Pass the flag to enable stricter validation for the modify workflow
                    parsed_df, qty_cols, vehicle_configs = validate_and_parse_pfep(pfep_df_for_process, is_modify_workflow=True)
                    if parsed_df is not None:
                        st.session_state.master_df = parsed_df
                        st.session_state.qty_cols = qty_cols
                        st.session_state.vehicle_configs = vehicle_configs
                        st.session_state.pincode = pincode
                        st.session_state.app_stage = "configure"
                        st.rerun()

    elif st.session_state.app_stage == "upload":
        st.header("Step 1: Upload Data Files to Create a New PFEP")
        uploaded_files = {}
        file_options = [ 
            ("PBOM", "pbom", True), ("MBOM", "mbom", True), 
            ("Part Attribute", "part_attribute", True), ("Vendor Master", "vendor_master", False), 
            ("Packaging Details", "packaging", True)
        ]
        for display_name, key_name, is_multiple in file_options:
            with st.expander(f"Upload {display_name} File(s)"):
                uploaded_files[key_name] = st.file_uploader(f"Upload", type=['csv', 'xlsx'], accept_multiple_files=is_multiple, key=f"upload_{key_name}", label_visibility="collapsed")
        pincode = st.text_input("Enter your location's pincode for distance calculations", value="411001")

        if st.button("Process Uploaded Files"):
            if not (uploaded_files['pbom'] or uploaded_files['mbom']):
                st.error("You must upload at least one PBOM or MBOM file.")
            else:
                st.session_state.all_files = load_all_files(uploaded_files)
                st.session_state.pincode = pincode
                if st.session_state.all_files['pbom'] and st.session_state.all_files['mbom']:
                    st.session_state.app_stage = "bom_selection"
                else:
                    bom_dfs = st.session_state.all_files['pbom'] + st.session_state.all_files['mbom']
                    base_bom = _consolidate_bom_list(bom_dfs)
                    if base_bom is not None:
                        supp_dfs = [st.session_state.all_files[k] for k in ['part_attribute', 'vendor_master', 'packaging']]
                        st.session_state.master_df, st.session_state.qty_cols = finalize_master_df(base_bom, supp_dfs)
                        st.session_state.app_stage = "configure"
                    else:
                        st.error("Failed to consolidate BOM data. Please check your files.")
                st.rerun()

    elif st.session_state.app_stage == "bom_selection":
        st.header("Step 1.5: BOM Base Selection")
        st.info("You uploaded both PBOM and MBOM files. Choose the base for the PFEP analysis.")
        with st.spinner("Analyzing differences..."):
            all_files = st.session_state.all_files
            master_pbom = _consolidate_bom_list(all_files['pbom'])
            master_mbom = _consolidate_bom_list(all_files['mbom'])
            pbom_parts = set(master_pbom['part_id']) if master_pbom is not None else set()
            mbom_parts = set(master_mbom['part_id']) if master_mbom is not None else set()
        st.subheader("BOM Comparison")
        col1, col2, col3 = st.columns(3)
        col1.metric("Parts Unique to PBOM", len(pbom_parts - mbom_parts))
        col2.metric("Parts Unique to MBOM", len(mbom_parts - pbom_parts))
        col3.metric("Parts Common to Both", len(pbom_parts.intersection(mbom_parts)))
        
        bom_choice = st.radio("Select the BOM base:", ('Use PBOM as base', 'Use MBOM as base', 'Combine both PBOM and MBOM'), horizontal=True)
        if st.button("Confirm Selection and Continue"):
            base_bom_df = { 'Use PBOM as base': master_pbom, 'Use MBOM as base': master_mbom, 'Combine both PBOM and MBOM': _consolidate_bom_list([master_pbom, master_mbom]) }[bom_choice]
            if base_bom_df is not None:
                supp_dfs = [all_files[k] for k in ['part_attribute', 'vendor_master', 'packaging']]
                st.session_state.master_df, st.session_state.qty_cols = finalize_master_df(base_bom_df, supp_dfs)
                st.session_state.app_stage = "configure"
                st.rerun()

    elif st.session_state.app_stage == "configure":
        header_text = "Step 2: Configure Daily Production"
        button_text = "🚀 Run Full Analysis"
        st.header(header_text)
        
        if not st.session_state.qty_cols:
            st.warning("No 'Quantity per Vehicle' columns detected. Consumption will be zero.")
            st.session_state.qty_cols = []
        else:
            st.info("Provide a name and daily production for each detected vehicle type.")
        
        if st.session_state.vehicle_configs is None:
            st.session_state.vehicle_configs = [{"name": f"Vehicle Type {i+1}", "multiplier": 1.0} for i, _ in enumerate(st.session_state.qty_cols)]

        vehicle_configs_input = []
        for i, _ in enumerate(st.session_state.qty_cols):
            default_config = st.session_state.vehicle_configs[i] if i < len(st.session_state.vehicle_configs) else {"name": f"Vehicle Type {i+1}", "multiplier": 1.0}
            st.markdown(f"**Detected Column/Vehicle Type #{i+1}**")
            cols = st.columns([2, 1])
            name = cols[0].text_input("Vehicle Name", default_config['name'], key=f"name_{i}")
            multiplier = cols[1].number_input("Daily Production", min_value=0.0, value=default_config.get('multiplier', 1.0), step=0.1, key=f"mult_{i}")
            vehicle_configs_input.append({"name": name, "multiplier": multiplier})
        
        if st.button(button_text):
            st.session_state.vehicle_configs = vehicle_configs_input
            processor = ComprehensiveInventoryProcessor(st.session_state.master_df)
            
            st.session_state.master_df = processor.calculate_dynamic_consumption(st.session_state.qty_cols, [c.get('multiplier', 0) for c in vehicle_configs_input])
            st.session_state.processor = processor

            # UNIFIED WORKFLOW: Both 'create' and 'modify' now start the full processing pipeline here.
            st.session_state.app_stage = "process_family" 
            st.rerun()

    elif st.session_state.app_stage.startswith(("process_", "review_")):
        processing_steps = [
            {"process_stage": "process_family", "review_stage": "review_family", "method": "run_family_classification", "key": "family", "name": "Family Classification"},
            {"process_stage": "process_size", "review_stage": "review_size", "method": "run_size_classification", "key": "size_classification", "name": "Size Classification"},
            {"process_stage": "process_part", "review_stage": "review_part", "method": "run_part_classification", "key": "part_classification", "name": "Part Classification"},
            {"process_stage": "process_packaging", "review_stage": "review_packaging", "method": "run_packaging_classification", "key": "one_way_returnable", "name": "Packaging Classification"},
            {"process_stage": "process_norms", "review_stage": "review_norms", "method": "run_location_based_norms", "key": "inventory_classification", "name": "Inventory Norms"},
            {"process_stage": "process_wh", "review_stage": "review_wh", "method": "run_warehouse_location_assignment", "key": "wh_loc", "name": "Warehouse Location"},
        ]
        
        current_step_index = -1
        for i, step in enumerate(processing_steps):
            if st.session_state.app_stage in [step['process_stage'], step['review_stage']]:
                current_step_index = i
                break
        
        if current_step_index != -1:
            step = processing_steps[current_step_index]
            next_stage = processing_steps[current_step_index + 1]['process_stage'] if current_step_index + 1 < len(processing_steps) else "generate_report"
            
            if st.session_state.app_stage == step['process_stage']:
                st.header(f"Step 3: Automated Processing")
                with st.spinner(f"Running {step['name']}..."):
                    processor = st.session_state.processor
                    # The pincode is now passed correctly from the session state
                    if step['method'] == 'run_location_based_norms':
                        getattr(processor, step['method'])(st.session_state.pincode)
                    else:
                        getattr(processor, step['method'])()
                    st.session_state.master_df = processor.data
                    st.session_state.app_stage = step['review_stage']
                    st.rerun()

            elif st.session_state.app_stage == step['review_stage']:
                render_review_step(step['name'], step['key'], next_stage)

    elif st.session_state.app_stage == "generate_report":
        report_data = create_formatted_excel_output(st.session_state.master_df, st.session_state.vehicle_configs, source_files_dict=st.session_state.get('source_files_for_report'))
        st.session_state.final_report = report_data
        st.balloons()
        st.success("🎉 End-to-end process complete!")
        st.session_state.app_stage = "download"
        st.rerun()

    elif st.session_state.app_stage == "download":
        st.header("Step 4: Download Final Report")
        st.download_button(label="📥 Download Structured Inventory Data Final.xlsx", data=st.session_state.final_report, file_name='structured_inventory_data_final.xlsx', mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        if st.button("Start Over"):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
