# Aircraft Engine Predictive Maintenance — New Streamlit Software

## What this app does

Raw engine time-series CSV
-> feature engineering
-> 65 model features
-> saved XGBoost model
-> predicted RUL

The app predicts RUL for the latest cycle of a selected engine.

## Folder structure

Aircraft_RUL_Software/
├── app.py
├── requirements.txt
├── README.md
├── models/
│   ├── final_rul_xgboost_model.pkl   # copy your saved model here
│   └── rul_feature_names.pkl         # copy your saved feature list here
└── utils/
    └── preprocessing.py

## Setup

Open a terminal in this folder:

    python -m pip install -r requirements.txt

Then:

    streamlit run app.py

## Important

Do NOT manually type a single sensor row for prediction.

The trained model uses rolling and delta features, so it needs time-series rows for an engine. Upload a CSV with all available cycles for an engine (or multiple engines) and select the engine.

The app supports either model filename:
- final_rul_xgboost_model.pkl
- xgboost_model.pkl

and either feature-name filename:
- rul_feature_names.pkl
- feature_names.pkl

## Required CSV columns

engine_id
cycle
op_setting_1
op_setting_2
op_setting_3
sensor_1 ... sensor_21

## Current project metrics

Validation:
MAE  = 23.58 cycles
RMSE = 30.97 cycles
R2   = 0.778

Final test-point:
MAE  = 19.71 cycles
RMSE = 27.68 cycles
R2   = 0.556

R2 is not classification accuracy. 0.556 means the model explains about 55.6% of the variance on that test-point evaluation.

## Model files

Copy the model files from your existing project into:

    models/

The software does not retrain the model.
