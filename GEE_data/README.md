# 🌍 Google Earth Engine (GEE) Satellite Datasets

This directory contains multi-source Earth Observation (EO) satellite products and spatial rasters downloaded via Google Earth Engine (GEE). The dataset blends hydro-meteorological, vegetation, soil, topographical, and cyclic temporal variables for machine learning-based drought modeling.

## 📋 Feature Variable Dictionary

| Category | Variable Name | Description |
| :--- | :--- | :--- |
| **Spatial Identifiers** | `Station` | Meteorological station identifier |
| | `Latitude`, `Longitude` | Geographic spatial coordinates (Decimal Degrees) |
| **Temporal Identifiers** | `Year`, `Month` | Calendar temporal attributes |
| **Meteorological Variables** | `Precipitation` | Total monthly precipitation |
| | `Temperature` | Mean surface temperature |
| | `Temperature_Min` | Monthly minimum temperature ($T_{min}$) |
| | `Temperature_Max` | Monthly maximum temperature ($T_{max}$) |
| | `Radiation` | Solar radiation flux |
| | `Wind_Speed` | Near-surface wind velocity |
| | `Relative_Humidity` | Near-surface relative humidity |
| **Remote Sensing & Vegetation** | `Reflectance_Band_X` | Satellite spectral band reflectances |
| | `NDVI_Value` | Normalized Difference Vegetation Index |
| | `SoilMoisture` | Volumetric soil moisture content |
| **Soil Physical Properties** | `Sand_Ratio`, `Clay_Ratio` | Soil texture composition percentages |
| | `FC` | Field Capacity |
| | `WP` | Wilting Point |
| **Topography & Cycles** | `DEM` | Digital Elevation Model (Elevation in meters) |
| | `Month_Sine`, `Month_Cosine` | Cyclic encoding for seasonal periodicity |

---

## 📊 Dataset Overview
* **Data Sources:** Multi-satellite observations (CHIRPS, ERA5-Land, MODIS, SRTM DEM, SoilGrids).
* **Target Output:** Standardized Precipitation Evapotranspiration Index (SPEI) and drought severity classification.
* **Usage:** Publicly available for model training, testing, and spatial feature engineering.

## 🔄 Reproduction
To re-run the automated extraction pipeline directly via the GEE Python API, execute:

```bash
python data_extraction/gee_data_download_code.py
