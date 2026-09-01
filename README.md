# Drought Forecasting and Earth System Modeling Pipeline

An end-to-end automated research pipeline for multi-temporal meteorological drought forecasting, geospatial data extraction, and physical index calculation. This repository is structured to handle large-scale climate data extraction, standardized drought index computations, and deep learning-driven predictive modeling.

---

## Repository Architecture

```text
drought-forecasting-pipeline/
├── assets/
│   ├── GEE_downloaded_dataa.png
│   └── Train_data_output.png
├── spei_processing/
│   └── spei_calculator.py
├── gee_data_download_code.py
├── requirements.txt
└── README.md 
```
## Research Workflow & Methodology

[ Google Earth Engine (GEE) ] 
       │
       ▼ (Extracts meteorological variables: P, Tmin, Tmax, Sunshine, Coordinates)
[ Geospatial Data Pipeline (`gee_data_download_code.py`) ]
       │
       ▼ (Computes climatic water balance WB = P - PET & multi-horizon indices)
[ Multi-Temporal SPEI Processing (`spei_processing/spei_calculator.py`) ]
       │
       ▼ (Feeds multi-scale temporal sequences into deep learning models)
[ Deep Learning Forecasting Architecture (LSTMs & Transformers) ]
       │
       ▼ (Executes non-linear mapping and ensemble predictions)
[ Random Forest (RF) & Advanced Regressors ]
       │
       ▼ (Generates localized, multi-horizon drought severity forecasts & spatial risk maps)
[ Final Drought Forecasting & Performance Evaluation ]

## Pipeline Execution & Results Preview
### 1. Extracted Google Earth Engine Data
Raw extracted meteorological variables, station coordinates, and hydro-climatic features processed from remote sensing sources:

![GEE Downloaded Data](assets/GEE_downloaded_dataa.png)

### 2. Multi-Temporal SPEI Processing, Deep Learning & Final Forecasting Output
Processed climatic water balance, PET calculations, multi-horizon SPEI indices, sequential LSTM/Transformer training metrics, and final Random Forest-driven drought severity forecasts:

![Train Data Output](assets/Train_data_output.png)

Installation & Reproducibility
To set up the environment and run the pipeline locally:

git clone https://github.com/aashutoshaad/drought-forecasting-pipeline.git
cd drought-forecasting-pipeline
pip install -r requirements.txt



