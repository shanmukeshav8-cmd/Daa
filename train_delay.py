from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "flight_delay.csv"
MODEL_PATH = ROOT / "models" / "delay_model.pkl"
FEATURES = [
    "airline",
    "origin",
    "destination",
    "scheduled_hour",
    "day_of_week",
    "weather",
    "traffic_level",
]
TARGET = "delay_minutes"
CATEGORICAL = ["airline", "origin", "destination", "weather", "traffic_level"]


def main():
    data = pd.read_csv(DATA_PATH)
    missing = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing:
        raise ValueError(f"{DATA_PATH.name} is missing required columns: {', '.join(missing)}")
    if data.empty or data[FEATURES + [TARGET]].isna().any().any():
        raise ValueError(f"{DATA_PATH.name} must contain non-empty rows without missing feature values.")

    preprocessor = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)],
        remainder="passthrough",
    )
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=1)),
        ]
    )
    model.fit(data[FEATURES], data[TARGET])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Trained delay estimator with {len(data)} sample records.")
    print("Saved model to models/delay_model.pkl")


if __name__ == "__main__":
    main()
