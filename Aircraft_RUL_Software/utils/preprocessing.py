import pandas as pd

SELECTED_SENSORS = [
    2, 3, 4, 7, 11, 12, 15, 17, 20, 21
]

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recreate the feature-engineering structure used by the trained model.

    Input:
        Raw engine time-series data with engine_id, cycle,
        3 operating settings and 21 sensors.

    Output:
        Raw model features + delta + rolling mean/std features.

    Important:
        The model expects the final column order to be supplied by
        rul_feature_names.pkl. The Streamlit app handles that ordering.
    """
    df = df.copy()

    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    for s in SELECTED_SENSORS:
        col = f"sensor_{s}"

        # Cycle-to-cycle change
        df[f"{col}_delta"] = (
            df.groupby("engine_id")[col].diff()
        )

        # Rolling statistics using only current/past observations.
        # min_periods=1 matches the behavior seen in the notebook output.
        df[f"{col}_roll_mean_5"] = (
            df.groupby("engine_id")[col]
              .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        )

        df[f"{col}_roll_mean_10"] = (
            df.groupby("engine_id")[col]
              .transform(lambda x: x.rolling(window=10, min_periods=1).mean())
        )

        df[f"{col}_roll_std_10"] = (
            df.groupby("engine_id")[col]
              .transform(lambda x: x.rolling(window=10, min_periods=1).std())
        )

    return df
