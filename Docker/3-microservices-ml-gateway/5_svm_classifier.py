"""
ALGORITMA 5: SVM Classifier (Siniflandirma) - Flask Servisi
Gorev: Oyunu "pahali (1)" / "ucuz (0)" olarak siniflandirmak
Dataset: steam_games.csv (prepare_dataset.py ile Kaggle'dan uretilir)
"""
import os
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 5005))

model = None
scaler = None
encoders = {}
feature_columns = []
price_threshold = 0


def train():
    global model, scaler, encoders, feature_columns, price_threshold
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

    price_threshold = df["Price"].median()
    df["expensive"] = (df["Price"] > price_threshold).astype(int)

    X = df.drop(["Price", "expensive"], axis=1)
    y = df["expensive"]
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel="rbf", random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("=" * 40)
    print("SVM CLASSIFIER SONUCLARI")
    print("=" * 40)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Ucuz", "Pahali"]))


def build_input_row(data):
    row = []
    for col in feature_columns:
        value = data.get(col)
        if col in encoders:
            le = encoders[col]
            value = int(le.transform([value])[0]) if value in le.classes_ else 0
        row.append(value)
    return scaler.transform([row])


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "svm", "model_ready": model is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "model not trained"}), 503
    data = request.get_json(force=True)
    prediction = int(model.predict(build_input_row(data))[0])
    return jsonify({"expensive": prediction, "label": "expensive" if prediction == 1 else "cheap",
                     "price_threshold": round(float(price_threshold), 2)})


if __name__ == "__main__":
    train()
    app.run(host="0.0.0.0", port=PORT)
