import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import math

# Page configuration
st.set_page_config(
    page_title="PFEP Automation Tool",
    page_icon="🏭",
    layout="wide"
)

# Initialize session state
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1

if 'pfep_data' not in st.session_state:
    st.session_state.pfep_data = pd.DataFrame()

if 'pbom_data' not in st.session_state:
    st.session_state.pbom_data = pd.DataFrame()

if 'mbom_data' not in st.session_state:
    st.session_state.mbom_data = pd.DataFrame()

if 'vendor_data' not in st.session_state:
    st.session_state.vendor_data = pd.DataFrame()

# Helper functions
def calculate_part_classification(value):
    """Calculate part classification based on ABC analysis"""
    if value >= 60:
        return 'AA'
    elif value >= 25:
        return 'A'
    elif value >= 12:
        return 'B'
    else:
        return 'C'

def get_inventory_classification(part_class):
    """Get inventory classification based on part classification"""
    classifications = {
        'AA': ['A1', 'A2', 'A3', 'A4'],
        'A': ['A1', 'A2', 'A3', 'A4'],
        'B': ['B1', 'B2', 'B3', 'B4'],
        'C': ['C1', 'C2']
    }
    return classifications.get(part_class, ['C1', 'C2'])

def calculate_rm_days(inventory_class):
    """Calculate RM in Days based on inventory classification"""
    rm_days = {
        'A1': 7, 'A2': 10, 'A3': 15, 'A4': 20,
        'B1': 15, 'B2': 20, 'B3': 25, 'B4': 30,
        'C1': 30, 'C2': 45
    }
    return rm_days.get(inventory_class, 30)

def get_family_keywords():
    """Return predefined family keywords for location selection"""
    return {
        'Engine': ['engine', 'motor', 'piston'],
        'Body': ['body', 'panel', 'door'],
        'Electrical': ['wire', 'cable', 'connector'],
        'Chassis': ['frame', 'axle', 'suspension'],
        'Interior': ['seat', 'dashboard', 'trim']
    }

# Main application
def main():
    st.title("🏭 PFEP Automation Tool")
    st.markdown("---")
    
    # Progress bar
    progress = st.session_state.current_step / 8
    st.progress(progress)
    st.write(f"Step {st.session_state.current_step} of 8")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Previous") and st.session_state.current_step > 1:
            st.session_state.current_step -= 1
            st.rerun()
    
    with col3:
        if st.button("Next →") and st.session_state.current_step < 8:
            st.session_state.current_step += 1
            st.rerun()
    
    # Step routing
    if st.session_state.current_step == 1:
        step1_initial_setup()
    elif st.session_state.current_step == 2:
        step2_upload_data()
    elif st.session_state.current_step == 3:
        step3_product_configuration()
    elif st.session_state.current_step == 4:
        step4_calculations()
    elif st.session_state.current_step == 5:
        step5_storage_supply()
    elif st.session_state.current_step == 6:
        step6_container_analysis()
    elif st.session_state.current_step == 7:
        step7_visualization()
    elif st.session_state.current_step == 8:
        step8_final_output()

def step1_initial_setup():
    st.header("Step 1: Initial Setup")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Drag Box Configuration")
        drag_box_qty = st.number_input("Drag Box Quantity", min_value=1, max_value=10, value=5)
        
        st.subheader("Station Configuration")
        station_required = st.checkbox("Station Number Required?")
        
        if station_required:
            st.info("MBOM upload will be required in the next step")
    
    with col2:
        st.subheader("Product Configuration")
        product_wise = st.checkbox("Product-wise Analysis", value=True)
        
        st.subheader("Daily Consumption")
        daily_consumption = st.number_input("Daily Consumption", min_value=1, value=100)
    
    # Store in session state
    st.session_state.drag_box_qty = drag_box_qty
    st.session_state.station_required = station_required
    st.session_state.product_wise = product_wise
    st.session_state.daily_consumption = daily_consumption

def step2_upload_data():
    st.header("Step 2: Upload Data Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("PBOM Upload")
        pbom_file = st.file_uploader("Upload PBOM", type=['csv', 'xlsx'], key="pbom")
        
        if pbom_file:
            try:
                if pbom_file.name.endswith('.csv'):
                    st.session_state.pbom_data = pd.read_csv(pbom_file)
                else:
                    st.session_state.pbom_data = pd.read_excel(pbom_file)
                st.success(f"PBOM uploaded: {len(st.session_state.pbom_data)} rows")
                st.dataframe(st.session_state.pbom_data.head())
            except Exception as e:
                st.error(f"Error uploading PBOM: {e}")
        
        if st.session_state.get('station_required', False):
            st.subheader("MBOM Upload")
            mbom_file = st.file_uploader("Upload MBOM", type=['csv', 'xlsx'], key="mbom")
            
            if mbom_file:
                try:
                    if mbom_file.name.endswith('.csv'):
                        st.session_state.mbom_data = pd.read_csv(mbom_file)
                    else:
                        st.session_state.mbom_data = pd.read_excel(mbom_file)
                    st.success(f"MBOM uploaded: {len(st.session_state.mbom_data)} rows")
                    st.dataframe(st.session_state.mbom_data.head())
                except Exception as e:
                    st.error(f"Error uploading MBOM: {e}")
    
    with col2:
        st.subheader("Vendor Data Upload")
        vendor_file = st.file_uploader("Upload Vendor Data", type=['csv', 'xlsx'], key="vendor")
        
        if vendor_file:
            try:
                if vendor_file.name.endswith('.csv'):
                    st.session_state.vendor_data = pd.read_csv(vendor_file)
                else:
                    st.session_state.vendor_data = pd.read_excel(vendor_file)
                st.success(f"Vendor data uploaded: {len(st.session_state.vendor_data)} rows")
                st.dataframe(st.session_state.vendor_data.head())
            except Exception as e:
                st.error(f"Error uploading vendor data: {e}")
        
        st.subheader("Size Classification")
        size_options = st.multiselect(
            "Select Size Categories",
            ['S', 'M', 'L', 'XL'],
            default=['S', 'M', 'L', 'XL']
        )
        st.session_state.size_options = size_options

def step3_product_configuration():
    st.header("Step 3: Product Configuration & Classification")
    
    if st.session_state.pbom_data.empty:
        st.warning("Please upload PBOM data in Step 2")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Part Classification Setup")
        st.info("Classification Percentages: AA-60%, A-25%, B-12%, C-3%")
        
        # Sample data for demonstration
        sample_parts = ['Part_A', 'Part_B', 'Part_C', 'Part_D', 'Part_E']
        sample_values = [70, 30, 15, 8, 2]
        
        classification_data = pd.DataFrame({
            'Part_Name': sample_parts,
            'Value_Percentage': sample_values,
            'Classification': [calculate_part_classification(v) for v in sample_values]
        })
        
        st.dataframe(classification_data)
        
        st.subheader("Family Keywords Configuration")
        family_keywords = get_family_keywords()
        
        selected_family = st.selectbox("Select Family", list(family_keywords.keys()))
        keywords = st.text_input(
            "Keywords (comma-separated)", 
            value=', '.join(family_keywords[selected_family])
        )
    
    with col2:
        st.subheader("Daily Consumption Calculation")
        
        qty_per_vehicle = st.number_input("Quantity per Vehicle", min_value=1, value=2)
        daily_consumption = st.session_state.get('daily_consumption', 100)
        
        total_daily_consumption = daily_consumption * qty_per_vehicle
        st.metric("Total Daily Consumption", total_daily_consumption)
        
        st.subheader("Inventory Classification")
        
        inventory_df = pd.DataFrame({
            'Part_Classification': ['AA', 'A', 'B', 'C'],
            'Inventory_Options': [
                ', '.join(get_inventory_classification('AA')),
                ', '.join(get_inventory_classification('A')),
                ', '.join(get_inventory_classification('B')),
                ', '.join(get_inventory_classification('C'))
            ]
        })
        
        st.dataframe(inventory_df)

def step4_calculations():
    st.header("Step 4: PFEP Calculations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Raw Material Calculations")
        
        # Sample calculation data
        sample_data = {
            'Part_Name': ['Part_A', 'Part_B', 'Part_C'],
            'Inventory_Class': ['A1', 'B2', 'C1'],
            'Unit_Price': [100, 50, 25],
            'Daily_Consumption': [10, 20, 5]
        }
        
        calc_df = pd.DataFrame(sample_data)
        
        # Calculate RM in Days
        calc_df['RM_Days'] = calc_df['Inventory_Class'].apply(calculate_rm_days)
        
        # Calculate RM in Qty
        calc_df['RM_Qty'] = calc_df['Daily_Consumption'] * calc_df['RM_Days']
        
        # Calculate RM in INR
        calc_df['RM_INR'] = calc_df['Unit_Price'] * calc_df['RM_Qty']
        
        st.dataframe(calc_df)
        
        st.subheader("Packing Calculations")
        
        packing_factor = st.number_input("Packing Factor (PF)", min_value=1, value=10)
        secondary_qty_pack = st.number_input("Secondary Qty Pack", min_value=1, value=50)
        
        # Calculate number of secondary packs
        calc_df['Sec_Pack_Required'] = calc_df['RM_Qty'].apply(
            lambda x: math.ceil(x / secondary_qty_pack)
        )
        
        calc_df['Sec_Pack_per_PF'] = calc_df['Sec_Pack_Required'].apply(
            lambda x: math.ceil(packing_factor * x)
        )
        
        st.dataframe(calc_df[['Part_Name', 'Sec_Pack_Required', 'Sec_Pack_per_PF']])
    
    with col2:
        st.subheader("Cost Analysis")
        
        # Cost visualization
        fig = px.bar(calc_df, x='Part_Name', y='RM_INR', 
                     title='Raw Material Cost by Part')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Quantity Analysis")
        
        fig2 = px.scatter(calc_df, x='Daily_Consumption', y='RM_Qty',
                         size='RM_INR', color='Inventory_Class',
                         title='Consumption vs RM Quantity')
        st.plotly_chart(fig2, use_container_width=True)

def step5_storage_supply():
    st.header("Step 5: Storage & Supply Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Store Location")
        
        location_options = ['HRR', 'CRL', 'MEZ']
        selected_location = st.selectbox("Primary Storage Location", location_options)
        
        jit_items = st.checkbox("Mark as JIT Items")
        
        if st.checkbox("Storage Full?"):
            alternative_location = st.selectbox(
                "Alternative Location", 
                [loc for loc in location_options if loc != selected_location]
            )
        
        st.subheader("Dock & Stacking")
        dock_number = st.number_input("Dock Number", min_value=1, value=1)
        stacking_factor = st.number_input("Stacking Factor", min_value=1, value=5)
        
        st.subheader("Supply Classification")
        supply_types = ['Direct', 'KIT trolley', 'Repacking']
        supply_type = st.selectbox("Supply Type", supply_types)
        
        packaging_type = st.selectbox(
            "Primary Packaging Type",
            ['Box', 'Bag', 'Container', 'Pallet']
        )
    
    with col2:
        st.subheader("Container Line Side")
        
        st.write("Container Dimensions")
        length = st.number_input("Length (L)", min_value=1.0, value=10.0)
        width = st.number_input("Width (W)", min_value=1.0, value=8.0)
        height = st.number_input("Height (H)", min_value=1.0, value=6.0)
        
        volume = length * width * height
        st.metric("Container Volume (V)", f"{volume:.2f} cubic units")
        
        st.subheader("Trolley/Bin Configuration")
        container_config = st.selectbox(
            "Container Configuration",
            ['Trolley', 'Bin', 'Rack']
        )
        
        st.subheader("Supply Condition Check")
        trolley_type = st.selectbox(
            "Trolley Type",
            ['Engineering Trolley', 'Standard Trolley']
        )
        
        # Quantity calculations
        qty_web = st.number_input("Qty/Web", min_value=1, value=10)
        supply_web_set = st.number_input("Supply Web Set", min_value=1, value=2)
        qty_cont = qty_web + supply_web_set
        
        st.metric("Qty/Container", qty_cont)

def step6_container_analysis():
    st.header("Step 6: Container & Storage Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Storage Line Side Configuration")
        
        storage_options = st.radio(
            "Storage Type",
            ['Trolley (Engineering)', 'Bin Flow (Rack)', 'Bin-Trolley']
        )
        
        if storage_options == 'Trolley (Engineering)':
            st.info("Using Engineering Trolley configuration")
        elif storage_options == 'Bin Flow (Rack)':
            st.info("Using Rack-based bin flow system")
        else:
            st.info("Using Bin-Trolley hybrid system")
        
        st.subheader("Container/Rack Upload")
        container_file = st.file_uploader(
            "Upload Container/Rack Configuration", 
            type=['csv', 'xlsx']
        )
        
        st.subheader("Trip Calculations")
        trips_per_day = st.number_input("Number of Trips per Day", min_value=1, value=4)
        trip_capacity = st.number_input("Trip Capacity", min_value=1, value=100)
        
        total_capacity = trips_per_day * trip_capacity
        st.metric("Total Daily Capacity", total_capacity)
    
    with col2:
        st.subheader("Inventory Line Side")
        
        # Sample inventory data
        inventory_data = {
            'Location': ['HRR-A1', 'CRL-B2', 'MEZ-C1'],
            'Current_Stock': [150, 200, 75],
            'Max_Capacity': [200, 300, 100],
            'Utilization_%': [75, 67, 75]
        }
        
        inventory_df = pd.DataFrame(inventory_data)
        st.dataframe(inventory_df)
        
        # Utilization chart
        fig = px.bar(inventory_df, x='Location', y='Utilization_%',
                     title='Storage Utilization by Location',
                     color='Utilization_%',
                     color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Space Analysis")
        space_analysis = pd.DataFrame({
            'Area': ['CRL', 'HRR', 'MEZ'],
            'Available_Space': [1000, 800, 600],
            'Used_Space': [750, 600, 450],
            'Remaining_Space': [250, 200, 150]
        })
        
        st.dataframe(space_analysis)

def step7_visualization():
    st.header("Step 7: Visualization & Analysis")
    
    # Part Classification Visualization
    st.subheader("Part Classification Analysis")
    
    # Sample data for visualization
    classification_data = {
        'Classification': ['AA', 'A', 'B', 'C'],
        'Percentage': [60, 25, 12, 3],
        'Count': [12, 25, 48, 115],
        'Value': [1200000, 500000, 240000, 60000]
    }
    
    class_df = pd.DataFrame(classification_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart for classification distribution
        fig1 = px.pie(class_df, values='Percentage', names='Classification',
                     title='Part Classification Distribution')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Bar chart for count
        fig2 = px.bar(class_df, x='Classification', y='Count',
                     title='Part Count by Classification',
                     color='Classification')
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Value analysis
        fig3 = px.bar(class_df, x='Classification', y='Value',
                     title='Value by Classification',
                     color='Value',
                     color_continuous_scale='viridis')
        st.plotly_chart(fig3, use_container_width=True)
        
        # Space analysis visualization
        space_data = {
            'Area': ['CRL', 'HRR', 'MEZ'],
            'Used': [75, 67, 82],
            'Available': [25, 33, 18]
        }
        
        fig4 = px.bar(space_data, x='Area', y=['Used', 'Available'],
                     title='Space Utilization Analysis',
                     barmode='stack')
        st.plotly_chart(fig4, use_container_width=True)
    
    st.subheader("Key Metrics Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Parts", "200", "5%")
    
    with col2:
        st.metric("Total Value", "₹2,000,000", "12%")
    
    with col3:
        st.metric("Storage Utilization", "74%", "-3%")
    
    with col4:
        st.metric("Active Vendors", "45", "2")

def step8_final_output():
    st.header("Step 8: Final PFEP Output")
    
    st.subheader("Generated PFEP Data")
    
    # Create comprehensive PFEP output
    pfep_output = {
        'Part_Number': ['P001', 'P002', 'P003', 'P004', 'P005'],
        'Description': ['Engine Component', 'Body Panel', 'Electrical Wire', 'Chassis Part', 'Interior Trim'],
        'Classification': ['AA', 'A', 'B', 'B', 'C'],
        'Daily_Consumption': [10, 15, 25, 8, 5],
        'RM_Days': [7, 10, 20, 20, 30],
        'RM_Qty': [70, 150, 500, 160, 150],
        'Unit_Price': [1000, 500, 100, 200, 50],
        'RM_Value': [70000, 75000, 50000, 32000, 7500],
        'Vendor': ['V001', 'V002', 'V003', 'V001', 'V004'],
        'Storage_Location': ['HRR', 'CRL', 'MEZ', 'HRR', 'CRL'],
        'Supply_Type': ['Direct', 'KIT trolley', 'Repacking', 'Direct', 'Repacking'],
        'Container_Type': ['Trolley', 'Bin', 'Rack', 'Trolley', 'Bin']
    }
    
    pfep_df = pd.DataFrame(pfep_output)
    st.dataframe(pfep_df)
    
    # Download options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Convert to Excel
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            pfep_df.to_excel(writer, sheet_name='PFEP_Output', index=False)
        
        st.download_button(
            label="📊 Download PFEP Excel",
            data=output.getvalue(),
            file_name="PFEP_Output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    with col2:
        # Convert to CSV
        csv = pfep_df.to_csv(index=False)
        st.download_button(
            label="📄 Download PFEP CSV",
            data=csv,
            file_name="PFEP_Output.csv",
            mime="text/csv"
        )
    
    with col3:
        if st.button("🔄 Reset Application"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    st.subheader("Summary Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**PFEP Configuration Summary:**")
        st.write(f"- Total Parts Processed: {len(pfep_df)}")
        st.write(f"- Total Value: ₹{pfep_df['RM_Value'].sum():,}")
        st.write(f"- Storage Locations Used: {len(pfep_df['Storage_Location'].unique())}")
        st.write(f"- Vendors Involved: {len(pfep_df['Vendor'].unique())}")
    
    with col2:
        st.write("**Classification Breakdown:**")
        class_summary = pfep_df['Classification'].value_counts()
        for class_type, count in class_summary.items():
            st.write(f"- {class_type} Class: {count} parts")
    
    st.success("✅ PFEP automation completed successfully!")
    st.balloons()

if __name__ == "__main__":
    main()
