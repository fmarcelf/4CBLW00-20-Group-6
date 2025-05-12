import pandas as pd
import geopandas as gpd
from pathlib import Path

def verify_numbers():
    print("\n=== Verifying Numbers ===")
    
    # 1. Check burglary cases data
    print("\n1. Checking burglary cases data:")
    df = pd.read_csv('output_csv_files/burglary_cases.csv')
    df['Month'] = pd.to_datetime(df['Month'])
    df['Year'] = df['Month'].dt.year
    
    print(f"Total burglary cases: {len(df)}")
    print(f"Time period: {df['Year'].min()} to {df['Year'].max()}")
    print(f"Unique months: {df['Month'].nunique()}")
    print(f"Unique LSOA codes: {df['LSOA code'].nunique()}")
    
    # 2. Check ward boundaries and crime mapping
    print("\n2. Checking ward boundaries and crime mapping:")
    ward_boundaries = gpd.read_file('statistical-gis-boundaries-london/ESRI/London_Ward_CityMerged.shp')
    lsoa_boundaries = gpd.read_file('statistical-gis-boundaries-london/ESRI/LSOA_2011_London_gen_MHW.shp')
    
    print(f"Total wards: {len(ward_boundaries)}")
    print(f"Total LSOAs: {len(lsoa_boundaries)}")
    
    # 3. Check crime density in visualizations
    print("\n3. Checking crime density in visualizations:")
    ward_crimes = df.groupby('LSOA code').size().reset_index(name='burglary_count')
    lsoa_boundaries = lsoa_boundaries.merge(ward_crimes, 
                                          left_on='LSOA11CD', 
                                          right_on='LSOA code', 
                                          how='left')
    lsoa_boundaries['burglary_count'] = lsoa_boundaries['burglary_count'].fillna(0)
    
    # Perform spatial join
    lsoa_in_wards = gpd.sjoin(lsoa_boundaries, ward_boundaries, how='inner', predicate='within')
    ward_crimes = lsoa_in_wards.groupby('GSS_CODE').agg({
        'burglary_count': 'sum'
    }).reset_index()
    
    # Merge with ward boundaries
    ward_boundaries = ward_boundaries.merge(ward_crimes, 
                                          left_on='GSS_CODE', 
                                          right_on='GSS_CODE', 
                                          how='left')
    ward_boundaries['burglary_count'] = ward_boundaries['burglary_count'].fillna(0)
    
    # Calculate area and density
    ward_boundaries['area_km2'] = ward_boundaries.geometry.area / 1e6
    ward_boundaries['crime_density'] = ward_boundaries['burglary_count'] / ward_boundaries['area_km2']
    
    print(f"Wards with no crimes: {len(ward_boundaries[ward_boundaries['burglary_count'] == 0])}")
    print(f"Total crimes mapped: {ward_boundaries['burglary_count'].sum()}")
    print("\nCrime density statistics:")
    print(ward_boundaries['crime_density'].describe())
    
    # 4. Check visualization files
    print("\n4. Checking visualization files:")
    vis_dir = Path('visualizations/boundaries')
    if vis_dir.exists():
        print("Visualization directory exists")
        print(f"Files in directory: {list(vis_dir.glob('*'))}")
    else:
        print("Visualization directory not found")

if __name__ == "__main__":
    verify_numbers() 