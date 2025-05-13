"""
Demand Estimation Script
This script estimates police demand for burglary prevention at the ward level in London.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import folium
import os
from pathlib import Path
import time

def load_burglary_data():
    """Load burglary data from Excel file and transform to DataFrame"""
    print("\n=== Loading Data from Excel File ===")
    
    start_time = time.time()
    
    # Read from Excel file
    input_file = Path('data') / 'transformed_data.xlsx'
    if not input_file.exists():
        raise FileNotFoundError(f"Excel file not found at {input_file}")
    
    try:
        df = pd.read_excel(input_file, header=1)  # Skip the first row and use the second row as headers
        
        # Check for required columns
        required_columns = ['LSOA code', 'Ward name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        total_time = time.time() - start_time
        print(f"\nDone! Total records found: {len(df)}")
        print(f"Data loaded from {input_file}")
        print(f"Total loading time: {total_time:.2f} seconds")
        
        return df
    except Exception as e:
        print(f"Failed to read {input_file}: {e}")
        raise

def load_lsoa_boundaries():
    """Load LSOA boundaries from ESRI shapefile"""
    print("\n=== Loading LSOA Boundaries ===")
    lsoa_path = Path('statistical-gis-boundaries-london/ESRI/LSOA_2011_London_gen_MHW.shp')
    lsoa_boundaries = gpd.read_file(lsoa_path)
    lsoa_boundaries = lsoa_boundaries.to_crs('EPSG:27700')  # British National Grid
    print(f"Loaded {len(lsoa_boundaries)} LSOA boundaries")
    return lsoa_boundaries

def load_geographic_boundaries():
    """Load London's geographic boundaries and return ward boundaries as GeoDataFrame"""
    print("\n=== Loading Geographic Boundaries ===")
    
    # Load ward boundaries from ESRI shapefile
    ward_path = Path('statistical-gis-boundaries-london/ESRI/London_Ward_CityMerged.shp')
    ward_boundaries = gpd.read_file(ward_path)
    ward_boundaries = ward_boundaries.to_crs('EPSG:27700')  # British National Grid
    
    print(f"Loaded {len(ward_boundaries)} London wards")
    return ward_boundaries

def map_crimes_to_wards(df, ward_boundaries, lsoa_boundaries):
    """Map crimes to ward boundaries using LSOA boundaries for accurate spatial join"""
    print("\nMapping crimes to ward boundaries...")
    
    if 'LSOA code' not in df.columns:
        raise ValueError("Crime data must include 'LSOA code' column")
    
    print(f"Total crimes in dataset: {len(df)}")
    print(f"Unique LSOA codes: {df['LSOA code'].nunique()}")
    
    # First, join crimes to LSOA boundaries
    lsoa_crimes = df.groupby('LSOA code').size().reset_index(name='burglary_count')
    lsoa_boundaries = lsoa_boundaries.merge(lsoa_crimes, 
                                          left_on='LSOA11CD', 
                                          right_on='LSOA code', 
                                          how='left')
    lsoa_boundaries['burglary_count'] = lsoa_boundaries['burglary_count'].fillna(0)
    
    # Perform spatial join between LSOAs and wards
    lsoa_in_wards = gpd.sjoin(lsoa_boundaries, ward_boundaries, how='inner', predicate='within')
    
    # Aggregate crimes by ward
    ward_crimes = lsoa_in_wards.groupby('GSS_CODE').agg({
        'burglary_count': 'sum'
    }).reset_index()
    
    # Merge with ward boundaries
    ward_boundaries = ward_boundaries.merge(ward_crimes, 
                                          left_on='GSS_CODE', 
                                          right_on='GSS_CODE', 
                                          how='left')
    ward_boundaries['burglary_count'] = ward_boundaries['burglary_count'].fillna(0)
    
    # Add ward names from the original data
    ward_names = df.groupby('Ward name')['LSOA code'].nunique().reset_index()
    ward_boundaries = ward_boundaries.merge(ward_names, 
                                          left_on='NAME', 
                                          right_on='Ward name', 
                                          how='left')
    
    print(f"\nWards with no crimes: {len(ward_boundaries[ward_boundaries['burglary_count'] == 0])}")
    print(f"Total crimes mapped: {ward_boundaries['burglary_count'].sum()}")
    
    # Calculate area and density
    ward_boundaries['area_km2'] = ward_boundaries.geometry.area / 1e6
    ward_boundaries['crime_density'] = ward_boundaries['burglary_count'] / ward_boundaries['area_km2']
    
    # Convert back to WGS84 for visualization
    ward_boundaries = ward_boundaries.to_crs('EPSG:4326')
    print(f"Final ward boundaries shape: {ward_boundaries.shape}")
    return ward_boundaries

def create_boundary_visualizations(ward_boundaries):
    """Visualize burglary density on static and interactive maps"""
    print("\nCreating boundary visualizations...")
    output_dir = Path('visualizations/boundaries')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print summary statistics
    print("\nCrime density statistics:")
    print(ward_boundaries['crime_density'].describe())
    
    # Create bins for better visualization
    max_density = ward_boundaries['crime_density'].max()
    bins = [0, max_density/5, max_density/2.5, max_density/1.67, max_density/1.25, max_density]
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    ward_boundaries['density_category'] = pd.cut(ward_boundaries['crime_density'], 
                                               bins=bins, 
                                               labels=labels, 
                                               include_lowest=True)

    # Create static map
    fig, ax = plt.subplots(figsize=(15, 10))
    ward_boundaries.plot(column='density_category', 
                        cmap='YlOrRd', 
                        linewidth=0.8, 
                        edgecolor='0.8',
                        ax=ax, 
                        legend=True,
                        categorical=True,
                        legend_kwds={'title': "Burglary Density"})
    
    ax.set_title('Burglary Density by Ward in London')
    ax.set_axis_off()
    plt.savefig(output_dir / 'london_burglary_density.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create interactive map
    m = folium.Map(location=[51.5074, -0.1278], zoom_start=10)

    # Add choropleth layer
    folium.Choropleth(
        geo_data=ward_boundaries.__geo_interface__,
        name='Burglary Density',
        data=ward_boundaries,
        columns=['GSS_CODE', 'burglary_count'],
        key_on='feature.properties.GSS_CODE',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name='Number of Burglaries'
    ).add_to(m)

    # Add tooltip layer
    style_function = lambda x: {'fillColor': '#ffffff', 
                              'color': '#000000', 
                              'fillOpacity': 0.1, 
                              'weight': 0.1}
    highlight_function = lambda x: {'fillColor': '#000000', 
                                  'color': '#000000', 
                                  'fillOpacity': 0.50, 
                                  'weight': 0.1}
    
    NIL = folium.features.GeoJson(
        ward_boundaries,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=['GSS_CODE', 'burglary_count', 'crime_density'],
            aliases=['Ward Code:', 'Total Burglaries:', 'Burglaries per km²:'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
        )
    )
    m.add_child(NIL)
    m.keep_in_front(NIL)

    # Add Layer Control
    folium.LayerControl().add_to(m)

    m.save(output_dir / 'london_burglary_interactive.html')
    print("Boundary visualizations created and saved.")


def main():
    """Run full demand estimation analysis"""
    # Load data from Excel file
    df = load_burglary_data()
    
    # Load boundaries
    lsoa_boundaries = load_lsoa_boundaries()
    ward_boundaries = load_geographic_boundaries()
    ward_boundaries = map_crimes_to_wards(df, ward_boundaries, lsoa_boundaries)

    create_boundary_visualizations(ward_boundaries)

if __name__ == "__main__":
    main() 