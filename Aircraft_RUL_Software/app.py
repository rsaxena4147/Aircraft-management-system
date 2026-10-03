import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from utils.preprocessing import create_features

st.set_page_config(
    page_title="Aircraft Engine Predictive Maintenance",
    page_icon="✈️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

MODEL_CANDIDATES = [
    MODEL_DIR / "final_rul_xgboost_model.pkl",
    MODEL_DIR / "xgboost_model.pkl",
]
FEATURE_CANDIDATES = [
    MODEL_DIR / "rul_feature_names.pkl",
    MODEL_DIR / "feature_names.pkl",
]

@st.cache_resource
def load_model():
    for p in MODEL_CANDIDATES:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError(
        "Model file not found. Put final_rul_xgboost_model.pkl "
        "or xgboost_model.pkl inside the models folder."
    )

@st.cache_resource
def load_feature_names():
    for p in FEATURE_CANDIDATES:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError(
        "Feature-name file not found. Put rul_feature_names.pkl "
        "or feature_names.pkl inside the models folder."
    )

st.title("✈️ Aircraft Engine Predictive Maintenance")
st.caption("XGBoost-based Remaining Useful Life (RUL) Prediction")

st.info(
    "Upload a time-series CSV containing one or more engines. "
    "The app recreates the same 65-feature structure used by the trained model "
    "and predicts RUL for the latest cycle of each selected engine."
)

uploaded_file = st.file_uploader(
    "Upload engine sensor data (.csv)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)

        st.subheader("1. Uploaded Data")
        st.write(f"Rows: **{len(raw_df):,}** | Columns: **{len(raw_df.columns)}**")
        st.dataframe(raw_df.head(10), use_container_width=True)

        if "engine_id" not in raw_df.columns:
            st.error("CSV must contain an `engine_id` column.")
            st.stop()

        required_raw = [
            "engine_id", "cycle",
            "op_setting_1", "op_setting_2", "op_setting_3"
        ] + [f"sensor_{i}" for i in range(1, 22)]

        missing = [c for c in required_raw if c not in raw_df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        engines = sorted(raw_df["engine_id"].unique())
        selected_engine = st.selectbox("Select Engine", engines)

        engine_df = raw_df[raw_df["engine_id"] == selected_engine].copy()
        engine_df = engine_df.sort_values("cycle").reset_index(drop=True)

        st.write(
            f"Engine **{selected_engine}** has **{len(engine_df)} cycles** "
            f"(latest cycle: **{engine_df['cycle'].iloc[-1]}**)."
        )

        if st.button("🔮 Predict RUL", type="primary"):
            with st.spinner("Creating features and predicting..."):
                engineered = create_features(engine_df)

                feature_names = load_feature_names()
                model = load_model()

                # Keep exactly the same feature set and order as training.
                X = engineered.reindex(columns=feature_names)

                # XGBoost can handle the NaNs created by first-cycle
                # delta/rolling calculations. Any completely absent
                # feature indicates an integration mismatch.
                absent = [c for c in feature_names if c not in engineered.columns]
                if absent:
                    st.error(f"Feature-engineering mismatch. Missing: {absent}")
                    st.stop()

                prediction = float(model.predict(X.tail(1))[0])
                prediction = max(0.0, prediction)

            st.subheader("2. Prediction")

            c1, c2, c3 = st.columns(3)
            c1.metric("Engine", str(selected_engine))
            c2.metric("Current Cycle", int(engine_df["cycle"].iloc[-1]))
            c3.metric("Predicted RUL", f"{prediction:.1f} cycles")

            if prediction <= 20:
                st.error("🔴 Critical — maintenance should be considered soon.")
            elif prediction <= 50:
                st.warning("🟠 Warning — monitor the engine closely.")
            else:
                st.success("🟢 Healthy — predicted remaining life is relatively high.")

            result = pd.DataFrame({
                "engine_id": [selected_engine],
                "current_cycle": [int(engine_df["cycle"].iloc[-1])],
                "predicted_RUL": [round(prediction, 2)]
            })

            st.subheader("3. Result")
            st.dataframe(result, use_container_width=True)

            st.download_button(
                "Download Prediction CSV",
                result.to_csv(index=False),
                file_name=f"engine_{selected_engine}_rul_prediction.csv",
                mime="text/csv"
            )

            st.subheader("4. Sensor / RUL Context")
            chart_df = engine_df[["cycle"]].copy()
            chart_df["Predicted RUL trend"] = np.nan

            # Predict every available cycle to visualize the model's behavior.
            all_X = engineered.reindex(columns=feature_names)
            all_pred = np.maximum(0, model.predict(all_X))
            chart_df["Predicted RUL trend"] = all_pred

            st.line_chart(
                chart_df.set_index("cycle")[["Predicted RUL trend"]]
            )

    except Exception as e:
        st.exception(e)
else:
    st.subheader("How to use")
    st.markdown("""
    1. Put the trained `.pkl` files in the `models` folder.
    2. Upload a CSV containing engine time-series sensor data.
    3. Select an engine.
    4. Click **Predict RUL**.
    5. The app creates the engineered features and sends the latest cycle to XGBoost.
    """)

st.divider()
st.caption("Model: XGBoost Regression | Target: Remaining Useful Life (RUL)")


# streamlit run app.py