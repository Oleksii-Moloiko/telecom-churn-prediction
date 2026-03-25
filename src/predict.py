import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    imputer = joblib.load(os.path.join(MODELS_DIR, "imputer.pkl"))
    features = joblib.load(os.path.join(MODELS_DIR, "features.pkl"))
    return model, scaler, imputer, features


def predict(data):
    model, scaler, imputer, features = load_artifacts()

    if len(data) != len(features):
        raise ValueError(f"Очікується {len(features)} ознак, отримано {len(data)}")

    data_df = pd.DataFrame([data], columns=features)

    data_imputed = imputer.transform(data_df)
    data_scaled = scaler.transform(data_imputed)

    proba = model.predict_proba(data_scaled)[0][1]
    result = "high" if proba > 0.5 else "low"

    return result, proba


if __name__ == "__main__":
    sample = [
        1,
        1,
        12,
        50,
        6,
        0,
        20,
        5,
        0
    ]

    result, proba = predict(sample)
    print("Рівень ризику:", result)
    print(f"Ймовірність відтоку: {proba:.2%}")