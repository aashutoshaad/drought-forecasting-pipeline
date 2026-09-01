"""
Earth Engine Data Pipeline for Hydrological & Climate Variables
Extracts time-series climate data (2000-2025) and static soil properties for station coordinates.
Outputs to multi-sheet Excel for downstream sequential machine learning (LSTM/RF) processing.
"""

import ee
import pandas as pd
import numpy as np
import time
import re
from google.colab import files

# --- Configuration ---
PROJECT_ID = 'root-engine-497416-h4'
STATION_FILE = 'Station_Metadata.csv'
OUTPUT_FILE = 'Station_Climate_Data_2000_2025.xlsx'
START_YEAR = 2000
END_YEAR = 2025

def initialize_gee():
    try:
        ee.Initialize(project=PROJECT_ID)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

def load_and_clean_stations(filepath):
    df = pd.read_csv(filepath)
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Filter bounds to Nepal region
    df = df[(df['Latitude'] >= 26) & (df['Latitude'] <= 31) & 
            (df['Longitude'] >= 80) & (df['Longitude'] <= 89)]
    
    features = [ee.Feature(ee.Geometry.Point([row['Longitude'], row['Latitude']]), 
                {'Station': str(row['Station']), 'Latitude': row['Latitude'], 'Longitude': row['Longitude']}) 
                for _, row in df.iterrows()]
    
    return ee.FeatureCollection(features)

def extract_static_variables(station_fc):
    dem = ee.Image('USGS/SRTMGL1_003').select('elevation')
    sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select(['b0'], ['sand'])
    clay = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02").select(['b0'], ['clay'])
    
    static_data = dem.addBands(sand).addBands(clay).reduceRegions(
        collection=station_fc, reducer=ee.Reducer.first(), crs='EPSG:4326', scale=1000
    ).getInfo()

    static_dict = {}
    for feat in static_data['features']:
        props = feat['properties']
        station = str(props.get('Station'))
        sand_val = props.get('sand', np.nan)
        clay_val = props.get('clay', np.nan)
        
        # Pedotransfer functions for Field Capacity and Wilting Point
        fc_val = 0.332 - (0.0007251 * sand_val) + (0.001276 * clay_val) if pd.notnull(sand_val) else np.nan
        wp_val = 0.0394 + (0.00714 * clay_val) if pd.notnull(clay_val) else np.nan
        
        static_dict[station] = {'DEM': props.get('elevation', np.nan), 'Sand_Ratio': sand_val, 
                                'Clay_Ratio': clay_val, 'FC': fc_val, 'WP': wp_val}
    return static_dict

def extract_time_series(station_fc, static_dict):
    data_records = []
    
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            try:
                start_date = ee.Date.fromYMD(year, month, 1)
                end_date = start_date.advance(1, 'month')

                climate = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate(start_date, end_date).first().select(['pr', 'tmmx', 'tmmn', 'srad', 'vs', 'soil', 'vap'])
                ndvi = ee.ImageCollection("MODIS/061/MOD13A3").filterDate(start_date, end_date)
                refl = ee.ImageCollection("MODIS/061/MOD09A1").filterDate(start_date, end_date)

                combined_img = climate
                if ndvi.size().getInfo() > 0:
                    combined_img = combined_img.addBands(ndvi.first().select(['NDVI']))
                if refl.size().getInfo() > 0:
                    combined_img = combined_img.addBands(refl.first().select(['sur_refl_b01', 'sur_refl_b02'], ['Red', 'NIR']))

                sampled = combined_img.reduceRegions(
                    collection=station_fc, reducer=ee.Reducer.first(), crs='EPSG:4326', scale=1000
                ).getInfo()

                for feat in sampled['features']:
                    props = feat['properties']
                    station = str(props.get('Station'))
                    row = {
                        'Station': station, 'Latitude': props.get('Latitude'), 'Longitude': props.get('Longitude'),
                        'Year': year, 'Month': month, 'pr': props.get('pr'), 'tmmx': props.get('tmmx'),
                        'tmmn': props.get('tmmn'), 'srad': props.get('srad'), 'vs': props.get('vs'),
                        'soil': props.get('soil'), 'vap': props.get('vap'), 
                        'NDVI': props.get('NDVI', np.nan), 'Red': props.get('Red', np.nan), 'NIR': props.get('NIR', np.nan)
                    }
                    row.update(static_dict.get(station, {}))
                    data_records.append(row)
                    
                time.sleep(0.1)  # Throttle API requests

            except Exception as e:
                print(f"Data missing or API error for {year}-{month:02d}. Skipping.")

    return pd.DataFrame(data_records)

def process_and_export(df):
    # Apply scale factors
    df['Total_Precipitation_mm'] = df['pr']
    df['Max_Temperature_C'] = df['tmmx'] * 0.1
    df['Min_Temperature_C'] = df['tmmn'] * 0.1
    df['Mean_Temperature_C'] = (df['Max_Temperature_C'] + df['Min_Temperature_C']) / 2
    df['Solar_Radiation_Wm2'] = df['srad'] * 0.1
    df['Wind_Speed_ms'] = df['vs'] * 0.01
    df['RootZone_SoilMoisture_mm'] = df['soil'] * 0.1
    df['Vapor_Pressure_kPa'] = df['vap'] * 0.001
    df['NDVI_Value'] = df['NDVI'] * 0.0001
    df['Red_Reflectance'] = df['Red'] * 0.0001
    df['NIR_Reflectance'] = df['NIR'] * 0.0001

    # Cyclical embeddings & Humidity
    df['e_s_kPa'] = 0.611 * np.exp((17.27 * df['Mean_Temperature_C']) / (df['Mean_Temperature_C'] + 237.3))
    df['Relative_Humidity_Pct'] = ((df['Vapor_Pressure_kPa'] / df['e_s_kPa']) * 100).clip(upper=100)
    df['Month_Sine'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_Cosine'] = np.cos(2 * np.pi * df['Month'] / 12)

    keep_cols = ['Station', 'Latitude', 'Longitude', 'Year', 'Month', 'Total_Precipitation_mm', 
                 'Mean_Temperature_C', 'Max_Temperature_C', 'Min_Temperature_C', 'Solar_Radiation_Wm2', 
                 'Wind_Speed_ms', 'Relative_Humidity_Pct', 'NIR_Reflectance', 'Red_Reflectance', 
                 'NDVI_Value', 'RootZone_SoilMoisture_mm', 'Sand_Ratio', 'Clay_Ratio', 'FC', 'WP', 
                 'DEM', 'Month_Sine', 'Month_Cosine']
    
    df = df[[c for c in keep_cols if c in df.columns]]

    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        for station_name, group_df in df.groupby('Station'):
            clean_sheet = re.sub(r'[\\/*?:\[\]]', '_', str(station_name))[:30]
            group_df.to_excel(writer, sheet_name=clean_sheet, index=False)
            
    files.download(OUTPUT_FILE)

if __name__ == "__main__":
    print("Initializing workflow...")
    initialize_gee()
    
    print("Loading station metadata...")
    stations = load_and_clean_stations(STATION_FILE)
    
    print("Extracting DEM and soil parameters...")
    static_vars = extract_static_variables(stations)
    
    print("Executing time-series extraction (2000-2025). This will take time...")
    raw_df = extract_time_series(stations, static_vars)
    
    print("Processing variables and compiling Excel workbook...")
    process_and_export(raw_df)
    print("Execution complete.")
