FROM python:3.13-slim

WORKDIR /app

# XGBoost'un ihtiyac duydugu sistem kutuphanesi
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY vgsales.csv .
COPY ml_xgboost.py .
COPY ml_logistic_regression.py .
COPY ml_kmeans.py .

RUN mkdir -p output
