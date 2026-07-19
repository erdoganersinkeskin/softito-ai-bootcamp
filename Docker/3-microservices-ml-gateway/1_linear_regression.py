"""
ALGORITMA 1: Linear Regression (Regresyon) - Flask Servisi
Gorev: Bir Steam oyununun fiyatini (price) tahmin etmek
Dataset: steam_games.csv (prepare_dataset.py ile Kaggle'dan uretilir)

ÖĞRENME NOTU (yönlendirme):
  Bu dosyanın amacı, Flask mikroservis + Linear Regression hattıyla bir
  Steam oyununun fiyatını tahmin etmektir.
"""
import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 5001))

model = None
encoders = {}
feature_columns = []


def train():
    global model, encoders, feature_columns
    df = pd.read_csv("steam_games.csv")

    if "AppID" in df.columns:
        df = df.drop("AppID", axis=1)
    if "Name" in df.columns:
        df = df.drop("Name", axis=1)
    if len(df) > 10000:
        df = df.sample(n=10000, random_state=42).reset_index(drop=True)
    df = df.dropna()

    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop("Price", axis=1)
    y = df["Price"]
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=" * 40)
    print("LINEAR REGRESSION SONUCLARI")
    print("=" * 40)
    print(f"R2 Skoru : {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE     : {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"MAE      : {mean_absolute_error(y_test, y_pred):.2f}")


def build_input_row(data):
    """Gelen JSON'i model girdisine cevirir."""
    row = []
    for col in feature_columns:
        value = data.get(col)
        if col in encoders:
            le = encoders[col]
            value = int(le.transform([value])[0]) if value in le.classes_ else 0
        row.append(value)
    return [row]


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "linear", "model_ready": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model not trained"}), 503
    data = request.get_json(force=True)
    prediction = float(model.predict(build_input_row(data))[0])
    return jsonify({"predicted_price": round(prediction, 2)})


if __name__ == "__main__":
    train()                                 # aciliste modeli egit
    app.run(host="0.0.0.0", port=PORT)      # sonra ayakta kal, istek bekle
