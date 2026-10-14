from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "queue_data.csv"
MODEL_PATH = ROOT / "models" / "queue_model.pkl"
FEATURES = ["terminal", "hour", "day_of_week", "passenger_volume", "open_lanes"]
TARGET = "queue_minutes"


def main():
    data = pd.read_csv(DATA_PATH)
    missing = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing:
        raise ValueError(f"{DATA_PATH.name} is missing required columns: {', '.join(missing)}")
    if data.empty or data[FEATURES + [TARGET]].isna().any().any():
        raise ValueError(f"{DATA_PATH.name} must contain non-empty rows without missing feature values.")

    model = Pipeline(
        [
            ("preprocessor", ColumnTransformer([("terminal", OneHotEncoder(handle_unknown="ignore"), ["terminal"])], remainder="passthrough")),
            ("regressor", RandomForestRegressor(n_estimators=160, random_state=42, min_samples_leaf=1)),
        ]
    )
    model.fit(data[FEATURES], data[TARGET])

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Trained queue estimator with {len(data)} sample records.")
    print("Saved model to models/queue_model.pkl")


if __name__ == "__main__":
    main()
