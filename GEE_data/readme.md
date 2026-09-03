# 🌍 Google Earth Engine (GEE) Satellite Datasets

This folder contains pre-processed multi-temporal meteorological and hydro-climatic data extracted from Google Earth Engine (GEE) remote sensing products.

## 📊 Dataset Overview
* **Data Source:** Publicly accessible satellite observation products (e.g., CHIRPS, ERA5-Land, MODIS).
* **Variables:** Precipitation ($P$), Minimum Temperature ($T_{min}$), Maximum Temperature ($T_{max}$), Sunshine duration, and station spatial coordinates.
* **Coverage:** Complete multi-station climate records spanning 2000–202X.
* **Usage:** Openly available for pipeline input, feature engineering, and model training.

## 🔄 Reproduction
If you wish to re-download or update these satellite metrics directly via GEE API, execute the extraction script from the repository root:

```bash
python data_extraction/gee_data_download_code.py
